'''
module for k-medoids claterization
'''

from typing import List, Optional, Dict, Any
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
    if vals.size == 0:
        return 1.0
    max_val = float(np.max(vals))
    return max_val if max_val > 0 else 1.0


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
    Check whether 2 points may be in one claster based on time wondows
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


def build_kmedoids_distance_matrix(
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
    sym_score_matrix = (score + score.T) / 2.0
    np.fill_diagonal(sym_score_matrix, 0.0)

    return sym_score_matrix


def initialize_medoids(distance_matrix, n_clusters, random_state):
    '''
    choose medoids
    '''
    n_points = distance_matrix.shape[0]
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive")
    if n_clusters > n_points:
        raise ValueError("Number of clusters cannot exceed number of points")


    rng = np.random.default_rng(random_state)

    medoids = [int(rng.integers(0, n_points))]

    while len(medoids) < n_clusters:
        distances = np.min(distance_matrix[:, medoids], axis=1)
        distances[medoids] = -1.0
        next_medoid = int(np.argmax(distances))
        medoids.append(next_medoid)

    return medoids


def assign_points_to_medoids(distance_matrix, medoid_indices):
    '''
    Assign points to the medoids
    '''
    distances_to_medoids = distance_matrix[:, medoid_indices]
    labels = np.argmin(distances_to_medoids, axis=1)
    return labels


def update_medoids(distance_matrix, labels, n_clusters):
    '''
    Update medoids inside each cluster
    '''
    new_medoids = []

    for cluster_id in range(n_clusters):
        cluster_points = np.where(labels == cluster_id)[0]

        if len(cluster_points) == 0:
            continue

        cluster_submatrix = distance_matrix[np.ix_(cluster_points, cluster_points)]
        point_costs = np.sum(cluster_submatrix, axis=1)
        best_local_idx = int(np.argmin(point_costs))
        best_global_idx = int(cluster_points[best_local_idx])
        new_medoids.append(best_global_idx)

    return new_medoids

def compute_total_cost(distance_matrix, medoid_indices, labels):
    '''
    Compute total clustering cost
    '''
    total_cost = 0.0
    for i in range(distance_matrix.shape[0]):
        medoid_idx = medoid_indices[labels[i]]
        total_cost += float(distance_matrix[i, medoid_idx])
    return total_cost


def fit_kmedoids(
    distance_matrix,
    n_clusters: int,
    max_iter: int = 100,
    random_state: Optional[int] = 42):
    '''
    find the best medoids
    '''
    n_points = distance_matrix.shape[0]
    medoid_indices = initialize_medoids(distance_matrix, n_clusters, random_state=random_state)

    num_of_iters = 0
    for _ in range(max_iter):
        num_of_iters+=1
        labels = assign_points_to_medoids(distance_matrix, medoid_indices)
        new_medoids = update_medoids(distance_matrix, labels, n_clusters)

        if len(new_medoids) < n_clusters:
            missing = n_clusters - len(new_medoids)
            available = [i for i in range(n_points) if i not in new_medoids]
            new_medoids.extend(available[:missing])

        if medoid_indices == new_medoids:
            break

        medoid_indices = new_medoids

    labels = assign_points_to_medoids(distance_matrix, medoid_indices)
    total_cost = compute_total_cost(distance_matrix, medoid_indices, labels)

    return {
        "labels": labels,
        "medoid_indices": medoid_indices,
        "costs": total_cost,
        "iterations": num_of_iters
    }


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


def run_kmedoids_pipeline(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    n_clusters: int,
    build_routes_ag,
    build_routes_kwargs,
    matrix_score,
    random_state: Optional[int] = 42,
    max_iter: int = 100):
    '''
    returns all routes that were created
    '''
    model_result = fit_kmedoids(
        distance_matrix=matrix_score,
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
    )

    labels = model_result["labels"]
    medoid_indices = model_result["medoid_indices"]

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
            "medoid_index": int(medoid_indices[cluster_id]) if cluster_id < len(medoid_indices) else None,
            "routes": routes,
            "summary": route_summary(routes),
            "num_iterations": model_result["iterations"],
        })
        all_routes.extend(routes)

    return {
        "labels": labels,
        "medoid_indices": medoid_indices,
        "routes": all_routes,
        "summary": route_summary(all_routes),
        "cluster_details": cluster_details,
        "cluster_distance_matrix": matrix_score,
        "costs": model_result["costs"],
        "num_iterations": model_result["iterations"],
    }


def find_optimal_n_clusters_kmedoids(
    day_stops: List,
    depo,
    dist_km: List[List[Optional[float]]],
    time_min: List[List[Optional[float]]],
    max_crews: int,
    build_routes_ag,
    build_routes_kwargs: Dict[str, Any],
    random_state: Optional[int] = 42,
    max_iter: int = 100):
    '''
    returns final solution
    '''
    n_points = len(day_stops)
    if n_points == 0:
        return "No point to cluster"

    matrix_score = build_kmedoids_distance_matrix(day_stops, time_min)

    best_result = None
    best_key = None


    for n in range(1, max_crews + 1):
        try:
            current_res = run_kmedoids_pipeline(
                day_stops=day_stops,
                depo=depo,
                dist_km=dist_km,
                time_min=time_min,
                n_clusters=n,
                matrix_score=matrix_score,
                build_routes_ag=build_routes_ag,
                build_routes_kwargs=build_routes_kwargs,
                random_state=random_state,
                max_iter=max_iter,
            )

            summary = current_res["summary"]
            actual_routes_count = summary["n_routes"]

            if actual_routes_count > max_crews:
                continue

            current_key = (
                summary["total_duration_min"],
                summary["total_distance_km"]
            )

            if best_result is None or current_key < best_key:
                best_key = current_key
                best_result = current_res

        except Exception as e:
            print(f"Error occured - {e}")
            continue

    if best_result is None:
        raise InfeasibleError

    return best_result
