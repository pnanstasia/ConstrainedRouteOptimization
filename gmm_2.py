'''
module for GMM clustering
'''

from typing import List, Optional, Dict, Any
from copy import deepcopy

import numpy as np
from sklearn.mixture import GaussianMixture


class InfeasibleError(Exception):
    '''
    Raising the error when task has no solutions
    '''


def extract_submatrix(matrix, indices):
    '''
    get submatrix from the large matrix
    '''
    return [[matrix[i][j] for j in indices] for i in indices]


def route_summary(routes):
    '''
    general metrics for routes
    '''
    total_distance = sum(float(r.distance_km) for r in routes)
    total_duration = sum(float(r.end_time - r.start_time) for r in routes)
    total_served = sum(int(r.served_count) for r in routes)

    return {
        "n_routes": len(routes),
        "total_distance_km": total_distance,
        "total_duration_min": total_duration,
        "total_served": total_served,
    }


def extract_gmm_features(day_stops: List):
    '''
    build feature matrix from stop coordinates
    '''
    if len(day_stops) == 0:
        return np.zeros((0, 2), dtype=float)

    return np.array([[stop.lat, stop.lon] for stop in day_stops], dtype=float)


def build_initial_clusters_from_labels(labels, n_clusters):
    '''
    put points into lists of clusters
    '''
    clusters = {cluster_id: [] for cluster_id in range(n_clusters)}

    for point_idx, cluster_id in enumerate(labels):
        clusters[int(cluster_id)].append(point_idx)

    return clusters


def fit_gmm_model(
    X,
    n_clusters: int,
    random_state: int = 42,
    covariance_type: str = "full",
    n_init: int = 5,
    max_iter: int = 200):
    '''
    fit GMM model
    '''
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    model.fit(X)

    probabilities = model.predict_proba(X)
    labels = np.argmax(probabilities, axis=1)

    return model, probabilities, labels


def get_candidate_clusters_for_point(point_probs):
    '''
    take max probability and keep all clusters with prob >= max_prob / 2
    '''
    max_prob = float(np.max(point_probs))
    threshold = max_prob / 2
    candidate_clusters = np.where(point_probs >= threshold)[0].tolist()
    return candidate_clusters


def identify_ambiguous_points(probabilities):
    '''
    find points that may belong to more than one cluster
    '''
    ambiguous_info = {}

    for point_idx in range(probabilities.shape[0]):
        point_probs = probabilities[point_idx]
        candidate_clusters = get_candidate_clusters_for_point(point_probs)

        if len(candidate_clusters) > 1:
            ambiguous_info[point_idx] = {
                "probs": point_probs,
                "candidate_clusters": candidate_clusters
            }

    return ambiguous_info


def run_algorithm_on_cluster(
    cluster_stops: List,
    depo,
    dist_km_global: List[List[Optional[float]]],
    time_min_global: List[List[Optional[float]]],
    global_customer_indices_zero_based: List[int],
    build_routes_gmm,
    build_routes_kwargs):
    '''
    run routing algorithm for one cluster
    '''
    global_matrix_indices = [0] + [idx + 1 for idx in global_customer_indices_zero_based]

    dist_sub = extract_submatrix(dist_km_global, global_matrix_indices)
    time_sub = extract_submatrix(time_min_global, global_matrix_indices)

    routes = build_routes_gmm(
        day_stops=cluster_stops,
        depo=depo,
        dist_km=dist_sub,
        times=time_sub,
        **build_routes_kwargs,
    )
    return routes


def evaluate_cluster_candidate(
    candidate_indices: List[int],
    day_stops: List,
    depo,
    dist_km,
    time_min,
    build_routes_gmm,
    build_routes_kwargs):
    '''
    Evaluate one candidate cluster by routing result
    '''
    if len(candidate_indices) == 0:
        return {
            "feasible": False,
            "total_duration_min": np.inf,
            "total_distance_km": np.inf,
            "routes": []
        }

    cluster_stops = [day_stops[i] for i in candidate_indices]

    try:
        routes = run_algorithm_on_cluster(
            cluster_stops=cluster_stops,
            depo=depo,
            dist_km_global=dist_km,
            time_min_global=time_min,
            global_customer_indices_zero_based=candidate_indices,
            build_routes_gmm=build_routes_gmm,
            build_routes_kwargs=build_routes_kwargs,
        )

        summary = route_summary(routes)

        return {
            "feasible": True,
            "total_duration_min": summary["total_duration_min"],
            "total_distance_km": summary["total_distance_km"],
            "routes": routes
        }

    except Exception:
        return {
            "feasible": False,
            "total_duration_min": np.inf,
            "total_distance_km": np.inf,
            "routes": []
        }


def compare_cluster_evaluations(eval_a, eval_b):
    '''
    Compare 2 cluster evaluations
    '''
    if eval_a["feasible"] and not eval_b["feasible"]:
        return -1
    if eval_b["feasible"] and not eval_a["feasible"]:
        return 1

    if eval_a["total_duration_min"] < eval_b["total_duration_min"]:
        return -1
    if eval_b["total_duration_min"] < eval_a["total_duration_min"]:
        return 1

    if eval_a["total_distance_km"] < eval_b["total_distance_km"]:
        return -1
    if eval_b["total_distance_km"] < eval_a["total_distance_km"]:
        return 1

    return 0


def find_current_cluster_of_point(clusters, point_idx):
    '''
    Find current cluster of point
    '''
    for cluster_id, points in clusters.items():
        if point_idx in points:
            return cluster_id
    return None


def resolve_ambiguous_points(
    day_stops: List,
    depo,
    clusters,
    ambiguous_info,
    dist_km,
    time_min,
    build_routes_gmm,
    build_routes_kwargs):
    '''
    choose the best cluster for ambiguous points
    '''
    final_clusters = deepcopy(clusters)
    reassignment_log = []

    ambiguous_items = list(ambiguous_info.items())
    ambiguous_items.sort(key=lambda x: len(x[1]["candidate_clusters"]), reverse=True)

    for point_idx, point_info in ambiguous_items:
        candidate_clusters = point_info["candidate_clusters"]
        current_cluster = find_current_cluster_of_point(final_clusters, point_idx)

        best_eval = None
        best_cluster = current_cluster

        for candidate_cluster in candidate_clusters:
            temp_clusters = deepcopy(final_clusters)

            current_points = temp_clusters[current_cluster]

            if candidate_cluster != current_cluster and len(current_points) <= 1:
                continue

            if candidate_cluster != current_cluster:
                temp_clusters[current_cluster].remove(point_idx)
                temp_clusters[candidate_cluster].append(point_idx)

            candidate_eval = evaluate_cluster_candidate(
                candidate_indices=temp_clusters[candidate_cluster],
                day_stops=day_stops,
                depo=depo,
                dist_km=dist_km,
                time_min=time_min,
                build_routes_gmm=build_routes_gmm,
                build_routes_kwargs=build_routes_kwargs
            )

            if best_eval is None:
                best_eval = candidate_eval
                best_cluster = candidate_cluster
            else:
                cmp_res = compare_cluster_evaluations(candidate_eval, best_eval)
                if cmp_res == -1:
                    best_eval = candidate_eval
                    best_cluster = candidate_cluster

        moved = False
        if best_eval is not None and best_eval["feasible"] and best_cluster != current_cluster:
            if len(final_clusters[current_cluster]) > 1:
                final_clusters[current_cluster].remove(point_idx)
                final_clusters[best_cluster].append(point_idx)
                moved = True

        reassignment_log.append({
            "point_idx": point_idx,
            "old_cluster": current_cluster,
            "new_cluster": best_cluster if best_eval is not None else current_cluster,
            "moved": moved,
            "candidate_clusters": candidate_clusters,
            "best_duration_min": None if best_eval is None else best_eval["total_duration_min"],
            "best_distance_km": None if best_eval is None else best_eval["total_distance_km"]
        })

    return final_clusters, reassignment_log


def build_cluster_details_and_routes(
    day_stops: List,
    depo,
    clusters,
    dist_km,
    time_min,
    build_routes_gmm,
    build_routes_kwargs):
    '''
    Build final routes and details for all clusters
    '''
    all_routes = []
    cluster_details = []
    crew_offset = 0

    for cluster_id in sorted(clusters.keys()):
        idxs = clusters[cluster_id]

        if not idxs:
            cluster_details.append({
                "cluster_id": int(cluster_id),
                "n_customers": 0,
                "customer_indices": [],
                "routes": [],
                "summary": route_summary([]),
            })
            continue

        cluster_stops = [day_stops[i] for i in idxs]

        routes = run_algorithm_on_cluster(
            cluster_stops=cluster_stops,
            depo=depo,
            dist_km_global=dist_km,
            time_min_global=time_min,
            global_customer_indices_zero_based=idxs,
            build_routes_gmm=build_routes_gmm,
            build_routes_kwargs=build_routes_kwargs,
        )

        for r in routes:
            r.crew_id += crew_offset

        if routes:
            current_max_id = max(r.crew_id for r in routes)
            crew_offset = current_max_id + 1

        cluster_details.append({
            "cluster_id": int(cluster_id),
            "n_customers": len(cluster_stops),
            "customer_indices": idxs,
            "routes": routes,
            "summary": route_summary(routes),
        })

        all_routes.extend(routes)

    return all_routes, cluster_details


def run_gmm_pipeline(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    n_clusters: int,
    build_routes_gmm,
    build_routes_kwargs: Dict[str, Any],
    random_state: int = 42,
    covariance_type: str = "full",
    n_init: int = 5,
    max_iter: int = 200):
    '''
    run gmm clustering pipeline
    '''
    n_points = len(day_stops)
    if n_points == 0:
        return {
            "model": None,
            "labels": np.array([], dtype=int),
            "probabilities": np.zeros((0, 0)),
            "routes": [],
            "summary": route_summary([]),
            "cluster_details": [],
            "clusters_initial": {},
            "clusters_final": {},
            "ambiguous_info": {},
            "reassignment_log": [],
            "n_clusters_used": n_clusters,
            "bic": np.nan,
            "aic": np.nan,
            "log_likelihood": np.nan,
            "converged": np.nan,
            "n_iter_gmm": np.nan,
        }

    X = extract_gmm_features(day_stops)

    model, probabilities, labels = fit_gmm_model(
        X=X,
        n_clusters=n_clusters,
        random_state=random_state,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter
    )

    clusters_initial = build_initial_clusters_from_labels(labels, n_clusters)

    ambiguous_info = identify_ambiguous_points(probabilities)

    clusters_final, reassignment_log = resolve_ambiguous_points(
        day_stops=day_stops,
        depo=depo,
        clusters=clusters_initial,
        ambiguous_info=ambiguous_info,
        dist_km=dist_km,
        time_min=time_min,
        build_routes_gmm=build_routes_gmm,
        build_routes_kwargs=build_routes_kwargs
    )

    all_routes, cluster_details = build_cluster_details_and_routes(
        day_stops=day_stops,
        depo=depo,
        clusters=clusters_final,
        dist_km=dist_km,
        time_min=time_min,
        build_routes_gmm=build_routes_gmm,
        build_routes_kwargs=build_routes_kwargs
    )

    result_summary = route_summary(all_routes)
    result_summary["n_clusters_final"] = len([c for c in clusters_final.values() if len(c) > 0])

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "routes": all_routes,
        "summary": result_summary,
        "cluster_details": cluster_details,
        "clusters_initial": clusters_initial,
        "clusters_final": clusters_final,
        "ambiguous_info": ambiguous_info,
        "reassignment_log": reassignment_log,
        "n_clusters_used": n_clusters,
        "bic": model.bic(X),
        "aic": model.aic(X),
        "log_likelihood": model.score(X),
        "converged": model.converged_,
        "n_iter_gmm": model.n_iter_,
    }


def find_optimal_n_clusters_gmm(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    max_crews: int,
    build_routes_gmm,
    build_routes_kwargs: Dict[str, Any],
    random_state: int = 42,
    covariance_type: str = "full",
    n_init: int = 5,
    max_iter: int = 200):
    '''
    returns final solution
    '''
    n_points = len(day_stops)
    if n_points == 0:
        return "No point to cluster"

    best_result = None
    best_key = None

    for n_clusters in range(1, max_crews + 1):
        try:
            current_res = run_gmm_pipeline(
                day_stops=day_stops,
                depo=depo,
                dist_km=dist_km,
                time_min=time_min,
                n_clusters=n_clusters,
                build_routes_gmm=build_routes_gmm,
                build_routes_kwargs=build_routes_kwargs,
                random_state=random_state,
                covariance_type=covariance_type,
                n_init=n_init,
                max_iter=max_iter,
            )

            summary = current_res["summary"]
            actual_routes_count = summary["n_routes"]

            if actual_routes_count > max_crews:
                continue

            current_key = (
                current_res["bic"],
                summary["total_duration_min"],
                summary["total_distance_km"]
            )

            if best_result is None or current_key < best_key:
                best_key = current_key
                best_result = current_res

        except Exception:
            continue

    if best_result is None:
        raise InfeasibleError

    return best_result
