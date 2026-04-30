'''
module for agglomerative claterization
'''

from typing import List, Optional, Dict, Any
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class InfeasibleError(Exception):
    '''
    Raising the error when task has no solutions
    '''


def find_max(values):
    '''
    find max value in the matrix
    '''
    vals = values[np.isfinite(values)]
    return float(np.max(vals))


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

def check_points(stop_i, stop_j, travel_ij, travel_ji):
    '''
    Check whether 2 points may be in one claster
    '''
    a_i = stop_i.tw_start
    b_i = stop_i.tw_end
    a_j = stop_j.tw_start
    b_j = stop_j.tw_end

    if (a_i is None or b_i is None) or (a_j is None or b_j is None):
        return 0.0

    service_i = float(stop_i.service)
    service_j = float(stop_j.service)

    feasible_slacks = []
    violations = []

    if travel_ij is not None:
        slack_ij = b_j - (a_i + service_i + float(travel_ij))
        if slack_ij >= 0:
            feasible_slacks.append(slack_ij)
        else:
            violations.append(abs(slack_ij))

    if travel_ji is not None:
        slack_ji = b_i - (a_j + service_j + float(travel_ji))
        if slack_ji >= 0:
            feasible_slacks.append(slack_ji)
        else:
            violations.append(abs(slack_ji))

    if feasible_slacks:
        best_slack = max(feasible_slacks)
        return 1.0 / (1.0 + best_slack)

    if violations:
        return 1.0 + min(violations)

    return 1.0


def build_agglomerative_distance_matrix(
    day_stops: List,
    time_min: List[List[Optional[float]]]):
    '''
    Build matrix for clarterization of points
    '''
    n = len(day_stops)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    time_arr = np.full((n, n), np.nan, dtype=float)
    tw_arr = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                time_arr[i, j] = 0.0
                tw_arr[i, j] = 0.0
                continue

            tij = time_min[i + 1][j + 1]
            tji = time_min[j + 1][i + 1]

            if tij is not None:
                time_arr[i, j] = float(tij)

            tw_arr[i, j] = check_points(
                stop_i=day_stops[i],
                stop_j=day_stops[j],
                travel_ij=tij,
                travel_ji=tji,
            )

    time_scale = find_max(time_arr)
    time_filled = np.where(np.isfinite(time_arr), time_arr, time_scale)
    time_term = time_filled / time_scale

    score = time_term + tw_arr
    sym_score_matrix = (score+ score.T) / 2.0
    np.fill_diagonal(sym_score_matrix, 0.0)

    return sym_score_matrix

def run_algorithm_on_cluster(
    cluster_stops: List,
    depo,
    dist_km_global: List[List[Optional[float]]],
    time_min_global: List[List[Optional[float]]],
    global_customer_indices_zero_based: List[int],
    build_routes_ag,
    build_routes_kwargs: Dict[str, Any]):
    '''
    run the algorithm for a specific cluster
    '''
    global_matrix_indices = [0] + [idx + 1 for idx in global_customer_indices_zero_based]

    dist_sub = extract_submatrix(dist_km_global, global_matrix_indices)
    time_sub = extract_submatrix(time_min_global, global_matrix_indices)

    routes = build_routes_ag(
        day_stops=cluster_stops,
        depo=depo,
        dist_km=dist_sub,
        times=time_sub,
        **build_routes_kwargs,
    )
    return routes

def run_agglomerative_pipeline(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    n_clusters: int,
    build_routes_ag,
    build_routes_kwargs,
    matrix_score,
    linkage_mode):
    '''
    returns all routes that were created
    '''

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage_mode,
    )
    labels = model.fit_predict(matrix_score)

    all_routes = []
    cluster_details = []
    crew_offset = 0

    for cluster_id in sorted(set(labels)):
        idxs = [i for i, lab in enumerate(labels) if lab == cluster_id]
        cluster_stops = [day_stops[i] for i in idxs]

        routes = run_algorithm_on_cluster(
            cluster_stops=cluster_stops,
            depo=depo,
            dist_km_global=dist_km,
            time_min_global=time_min,
            global_customer_indices_zero_based=idxs,
            build_routes_ag=build_routes_ag,
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

    return {
        "labels": labels,
        "routes": all_routes,
        "summary": route_summary(all_routes),
        "cluster_details": cluster_details,
        "cluster_distance_matrix": matrix_score,
    }

def find_optimal_n_clusters(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    max_crews: int,
    linkage_mode: str,
    build_routes_ag,
    build_routes_kwargs: Dict[str, Any]):
    '''
    returns final solution
    '''
    n_points = len(day_stops)
    if n_points == 0:
        return "No point to cluster"

    matrix_score = build_agglomerative_distance_matrix(day_stops, time_min)

    best_result = None
    best_key = None


    for n in range(1, max_crews + 1):
        try:
            current_res = run_agglomerative_pipeline(
                day_stops=day_stops,
                depo=depo,
                dist_km=dist_km,
                time_min=time_min,
                n_clusters=n,
                matrix_score=matrix_score,
                build_routes_ag=build_routes_ag,
                build_routes_kwargs=build_routes_kwargs,
                linkage_mode=linkage_mode
            )

            summary = current_res['summary']
            actual_routes_count = summary['n_routes']

            if actual_routes_count > max_crews:
                continue

            current_key = (
                summary['total_duration_min'],
                summary['total_distance_km']
            )

            if best_result is None or current_key < best_key:
                best_key = current_key
                best_result = current_res

        except Exception as e:
            continue

    if best_result is None:
        raise InfeasibleError

    return best_result
