'''
module for vns algorithm
'''
from dataclasses import dataclass
from typing import List, Optional, Dict
import copy
import random


class InfeasibleError(Exception):
    '''
    Raising the error when task has no solutions
    '''


@dataclass
class EvaluatedRoute:
    '''
    class for results of constructed route
    '''
    route_nodes: List[int]
    start_time: float
    end_time: float
    distance_km: float
    served_count: int
    feasible: bool


class GlobalVNS:
    '''
    vns class with necessary methods
    '''

    def __init__(
        self,
        day_stops: List,
        depo,
        dist_km: List[List[Optional[float]]],
        time_min: List[List[Optional[float]]],
        max_route_duration_min: int,
        random_state: int = 42):
        self.day_stops = day_stops
        self.depo = depo
        self.dist_km = dist_km
        self.time_min = time_min
        self.max_route_duration_min = max_route_duration_min
        self.rng = random.Random(random_state)
        self.stop_id_to_node: Dict[str, int] = {
            stop.stop_id: i for i, stop in enumerate(day_stops, start=1)
        }

    def route_to_nodes(self, route):
        '''
        Convert route id into indexes
        '''
        nodes = []
        for stop_id in route.stop_sequence:
            if stop_id == self.depo.stop_id:
                nodes.append(0)
            else:
                if stop_id not in self.stop_id_to_node:
                    raise KeyError(f'Unknown stop_id {stop_id}')
                nodes.append(self.stop_id_to_node[stop_id])
        return nodes

    def routes_to_solution(self, routes: List):
        '''
        cope of the information for experiments
        '''
        solution = []
        for r in routes:
            solution.append({
                "crew_id": r.crew_id,
                "start_time": float(r.start_time),
                "workers_used": r.workers_used,
                "nodes": self.route_to_nodes(r),
            })
        return solution

    def solution_to_routes(self, solution, route_cls):
        '''
        convert routes into coherent result
        '''
        routes_res = []

        for item in solution:
            evaluation = self.evaluate_route(
                route_nodes=item["nodes"],
                route_start=item["start_time"]
            )
            if not evaluation.feasible:
                raise InfeasibleError("infeasible route")

            stop_sequence = [
                self.depo.stop_id if node == 0 else self.day_stops[node - 1].stop_id
                for node in item["nodes"]
            ]

            routes_res.append(route_cls(
                crew_id=item["crew_id"],
                stop_sequence=stop_sequence,
                start_time=int(round(item["start_time"])),
                end_time=int(round(evaluation.end_time)),
                distance_km=float(evaluation.distance_km),
                workers_used=item["workers_used"],
                served_count=evaluation.served_count,
            ))
        return routes_res

    def evaluate_route(self, route_nodes, route_start) -> EvaluatedRoute:
        '''
        check the route whether it's accepted or not
        '''
        if len(route_nodes) < 2 or route_nodes[0] != 0 or route_nodes[-1] != 0:
            return EvaluatedRoute(route_nodes, route_start, float("inf"), float("inf"), 0, False)

        current_time = float(route_start)
        route_deadline = float(route_start) + self.max_route_duration_min
        total_distance = 0.0
        served_count = 0

        for i in range(len(route_nodes) - 1):
            a = route_nodes[i]
            b = route_nodes[i + 1]

            time_ab = self.time_min[a][b]
            dist_ab = self.dist_km[a][b]

            if time_ab is None or dist_ab is None:
                return EvaluatedRoute(route_nodes, route_start, float("inf"), float("inf"), 0, False)

            current_time += float(time_ab)
            total_distance += float(dist_ab)

            if b != 0:
                stop = self.day_stops[b - 1]

                if stop.tw_start is not None:
                    current_time = max(current_time, float(stop.tw_start))

                if stop.tw_end is not None and current_time > float(stop.tw_end):
                    return EvaluatedRoute(route_nodes, route_start, float("inf"), float("inf"), 0, False)

                current_time += float(stop.service)
                served_count += 1

        if current_time > route_deadline:
            return EvaluatedRoute(route_nodes, route_start, float("inf"), float("inf"), 0, False)

        return EvaluatedRoute(
            route_nodes=route_nodes,
            start_time=route_start,
            end_time=current_time,
            distance_km=total_distance,
            served_count=served_count,
            feasible=True
        )

    def evaluate_solution(self, solution):
        '''
        returns total duration and distance for accepted routes
        '''
        total_duration = 0.0
        total_distance = 0.0

        for item in solution:
            result = self.evaluate_route(item["nodes"], item["start_time"])
            if not result.feasible:
                return float("inf"), float("inf")

            total_duration += float(result.end_time - result.start_time)
            total_distance += float(result.distance_km)
        return total_duration, total_distance

    def intra_swap(self, solution):
        '''
        swap 2 random points in a route
        '''
        new_sol = copy.deepcopy(solution)
        candidate_routes = [r for r in new_sol if len(r["nodes"]) > 4]
        if not candidate_routes:
            return new_sol

        route = self.rng.choice(candidate_routes)
        i, j = self.rng.sample(range(1, len(route["nodes"]) - 1), 2)
        route["nodes"][i], route["nodes"][j] = route["nodes"][j], route["nodes"][i]
        return new_sol

    def two_opt(self, solution):
        '''
        reverse random part of route
        '''
        new_sol = copy.deepcopy(solution)
        candidate_routes = [r for r in new_sol if len(r["nodes"]) > 4]
        if not candidate_routes:
            return new_sol

        route = self.rng.choice(candidate_routes)
        i, j = sorted(self.rng.sample(range(1, len(route["nodes"]) - 1), 2))
        route["nodes"][i:j + 1] = reversed(route["nodes"][i:j + 1])
        return new_sol


    def intra_relocate(self, solution):
        '''
        randomly change point poisition in the route
        '''
        new_sol = copy.deepcopy(solution)
        candidate_routes = [r for r in new_sol if len(r["nodes"]) > 4]
        if not candidate_routes:
            return new_sol

        route = self.rng.choice(candidate_routes)
        nodes = route["nodes"]

        remove_idx = self.rng.randint(1, len(nodes) - 2)
        customer = nodes.pop(remove_idx)

        insert_positions = list(range(1, len(nodes)))
        insert_idx = self.rng.choice(insert_positions)
        nodes.insert(insert_idx, customer)
        return new_sol

    def inter_relocate(self, solution):
        '''
        change point between 2 routes
        '''
        new_sol = copy.deepcopy(solution)
        source_routes = [r for r in new_sol if len(r["nodes"]) >= 3]
        if len(new_sol) < 2 or not source_routes:
            return new_sol

        from_route = self.rng.choice(source_routes)
        to_candidates = [r for r in new_sol if r is not from_route]
        if not to_candidates:
            return new_sol

        to_route = self.rng.choice(to_candidates)
        remove_idx = self.rng.randint(1, len(from_route["nodes"]) - 2)
        customer = from_route["nodes"].pop(remove_idx)

        insert_idx = self.rng.randint(1, len(to_route["nodes"]) - 1)
        to_route["nodes"].insert(insert_idx, customer)
        return new_sol

    def inter_swap(self, solution: List[Dict]) -> List[Dict]:
        '''
        swap 1 point from one route to another one and vice versa
        '''
        new_sol = copy.deepcopy(solution)
        candidate_routes = [r for r in new_sol if len(r["nodes"]) >= 3]
        if len(candidate_routes) < 2:
            return new_sol

        r1, r2 = self.rng.sample(candidate_routes, 2)
        i = self.rng.randint(1, len(r1["nodes"]) - 2)
        j = self.rng.randint(1, len(r2["nodes"]) - 2)
        r1["nodes"][i], r2["nodes"][j] = r2["nodes"][j], r1["nodes"][i]
        return new_sol

    def shake(self, solution, k):
        '''
        choose shake method based on k
        '''
        if k == 1:
            return self.intra_swap(solution)
        elif k == 2:
            return self.two_opt(solution)
        elif k == 3:
            return self.intra_relocate(solution)
        elif k == 4:
            return self.inter_relocate(solution)
        elif k == 5:
            return self.inter_swap(solution)

    def local_search(self, solution, max_iter):
        '''
        first-improvement local search
        '''
        current = copy.deepcopy(solution)
        current_value = self.evaluate_solution(current)

        moves = [
            self.intra_swap,
            self.two_opt,
            self.intra_relocate,
            self.inter_relocate,
            self.inter_swap,
        ]

        for _ in range(max_iter):
            improved = False

            for move in moves:
                candidate = move(current)
                candidate_value = self.evaluate_solution(candidate)

                if candidate_value < current_value:
                    current = candidate
                    current_value = candidate_value
                    improved = True
                    break

            if not improved:
                break

        return current

    def remove_empty_routes(self, solution):
        '''
        remove empthy routes
        '''
        cleaned = []
        for item in solution:
            nodes = item["nodes"]
            if nodes == [0, 0]:
                continue
            cleaned.append(item)
        return cleaned

    def main_vns(
        self,
        initial_routes,
        route_cls,
        max_iterations: int = 100,
        k_max: int = 5,
        local_search_max_iter: int = 50):
        '''
        main function to run vns
        '''
        current_solution = self.routes_to_solution(initial_routes)
        current_value = self.evaluate_solution(current_solution)

        if current_value[0] == float("inf"):
            raise InfeasibleError("time is inf, VNS can't be applied")

        best_solution = copy.deepcopy(current_solution)
        best_value = current_value

        for _ in range(max_iterations):
            k = 1

            while k <= k_max:
                shaken = self.shake(current_solution, k)
                shaken = self.remove_empty_routes(shaken)

                improved = self.local_search(shaken, max_iter=local_search_max_iter)
                improved = self.remove_empty_routes(improved)

                improved_value = self.evaluate_solution(improved)

                if improved_value < current_value:
                    current_solution = improved
                    current_value = improved_value

                    if improved_value < best_value:
                        best_solution = copy.deepcopy(improved)
                        best_value = improved_value

                    k = 1
                else:
                    k += 1

        best_routes = self.solution_to_routes(best_solution, route_cls=route_cls)
        return best_routes, best_value
