'''
module for alns algorithm
'''
from dataclasses import dataclass
from typing import List, Optional
import copy
import random
import math


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


class GlobalALNS:
    '''
    ALNS class with necessary methods
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
        self.stop_id_to_node = {
            stop.stop_id: i for i, stop in enumerate(day_stops, start=1)
        }
        self.node_to_stop = {
            i: stop for i, stop in enumerate(day_stops, start=1)
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

    def remove_empty_routes(self, solution):
        '''
        remove empty routes
        '''
        cleaned = []
        for item in solution:
            nodes = item["nodes"]
            if nodes == [0, 0]:
                continue
            cleaned.append(item)
        return cleaned

    def get_all_points(self, solution):
        '''
        get all points needs to be served (without depot)
        '''
        customers = []
        for item in solution:
            for node in item["nodes"]:
                if node != 0:
                    customers.append(node)
        return customers

    def count_points(self, solution):
        '''
        count all served customers
        '''
        return len(self.get_all_points(solution))



    def acceptance_probability(self, current_value, candidate_value, temperature):
        '''
        calc acceptance prob for worse routes
        '''
        current_duration, current_distance = current_value
        candidate_duration, candidate_distance = candidate_value

        if candidate_duration == float("inf") or candidate_distance == float("inf"):
            return 0.0
        if candidate_duration < current_duration:
            return 1.0
        if candidate_duration == current_duration and candidate_distance < current_distance:
            return 1.0
        if temperature <= 1e-12:
            return 0.0

        delta = candidate_duration - current_duration
        return math.exp(-delta / temperature)

    def accept_solution(self, current_value, candidate_value, temperature):
        '''
        accept solution
        '''
        current_duration, current_distance = current_value
        candidate_duration, candidate_distance = candidate_value

        if candidate_duration == float("inf") or candidate_distance == float("inf"):
            return False
        if candidate_duration < current_duration:
            return True
        if candidate_duration == current_duration and candidate_distance < current_distance:
            return True

        prob = self.acceptance_probability(current_value, candidate_value, temperature)
        return self.rng.random() < prob

    def choose_pt_remove(self, total_points, q_min, q_max) :
        '''
        choose number of points to remove
        '''
        if total_points <= 0:
            return 0
        q_max = min(q_max, total_points)
        return self.rng.randint(q_min, q_max)

    def random_removal(self, solution, q):
        '''
        randomly remove customers for one dataset
        '''
        new_sol = copy.deepcopy(solution)
        all_positions = []

        for route_idx, item in enumerate(new_sol):
            for pos_idx, node in enumerate(item["nodes"]):
                if node != 0:
                    all_positions.append((route_idx, pos_idx, node))

        if not all_positions:
            return new_sol, []

        q = min(q, len(all_positions))
        selected = self.rng.sample(all_positions, q)

        selected_set = set()
        removed_customers = []
        for r_idx, p_idx, node in selected:
            selected_set.add((r_idx, p_idx))
            removed_customers.append(node)

        for route_idx, item in enumerate(new_sol):
            new_nodes = []
            for pos_idx, node in enumerate(item["nodes"]):
                if (route_idx, pos_idx) in selected_set:
                    continue
                new_nodes.append(node)
            item["nodes"] = new_nodes

        new_sol = self.remove_empty_routes(new_sol)
        return new_sol, removed_customers

    def removal_changes(self, route_nodes, route_start, position):
        '''
        returns duration and distance of new route after removing
        '''
        if position <= 0 or position >= len(route_nodes) - 1:
            return (-float("inf"), -float("inf"))

        old_eval = self.evaluate_route(route_nodes, route_start)
        if not old_eval.feasible:
            return (-float("inf"), -float("inf"))

        new_nodes = route_nodes[:position] + route_nodes[position + 1:]
        new_eval = self.evaluate_route(new_nodes, route_start)
        if not new_eval.feasible:
            return (-float("inf"), -float("inf"))

        duration_gain = (old_eval.end_time - old_eval.start_time) - (new_eval.end_time - new_eval.start_time)
        distance_gain = old_eval.distance_km - new_eval.distance_km

        return (float(duration_gain), float(distance_gain))

    def worst_removal(self, solution, q):
        '''
        remove customers with the best route changes in one dataset
        '''
        new_sol = copy.deepcopy(solution)
        candidates = []

        for route_idx, item in enumerate(new_sol):
            nodes = item["nodes"]
            route_start = item["start_time"]

            for loc in range(1, len(nodes) - 1):
                node = nodes[loc]
                gain = self.removal_changes(nodes, route_start, loc)
                candidates.append((gain, route_idx, loc, node))

        if not candidates:
            return new_sol, []

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = candidates[:min(q, len(candidates))]
        selected_set = set()
        removed_customers = []
        for _, r_idx, p_idx, node in selected:
            selected_set.add((r_idx, p_idx))
            removed_customers.append(node)

        for route_idx, item in enumerate(new_sol):
            new_nodes = []
            for pos_idx, node in enumerate(item["nodes"]):
                if (route_idx, pos_idx) in selected_set:
                    continue
                new_nodes.append(node)
            item["nodes"] = new_nodes

        new_sol = self.remove_empty_routes(new_sol)
        return new_sol, removed_customers

    def find_best_insertion(self, solution, point):
        '''
        find best insertion place for point
        '''
        best_solution = None
        best_value = (float("inf"), float("inf"))

        for route_idx, item in enumerate(solution):
            nodes = item["nodes"]

            for insert_pos in range(1, len(nodes)):
                candidate = copy.deepcopy(solution)
                candidate[route_idx]["nodes"].insert(insert_pos, point)

                route_eval = self.evaluate_route(
                    route_nodes=candidate[route_idx]["nodes"],
                    route_start=candidate[route_idx]["start_time"]
                )
                if not route_eval.feasible:
                    continue

                candidate_value = self.evaluate_solution(candidate)
                if best_value > candidate_value:
                    best_solution = candidate
                    best_value = candidate_value

        return best_solution, best_value

    def greedy_insertion(self, partial_solution, removed_poits):
        '''
        insert removed points in routes
        '''
        current = copy.deepcopy(partial_solution)
        points_to_insert = removed_poits[:]
        self.rng.shuffle(points_to_insert)

        while points_to_insert:
            best_global_solution = None
            best_global_value = (float("inf"), float("inf"))
            best_customer = None

            for customer in points_to_insert:
                cand_sol, cand_value = self.find_best_insertion(current, customer)
                if cand_sol is not None and cand_value < best_global_value:
                    best_global_solution = cand_sol
                    best_global_value = cand_value
                    best_customer = customer

            if best_global_solution is None:
                return partial_solution, False

            current = best_global_solution
            points_to_insert.remove(best_customer)

        current = self.remove_empty_routes(current)
        return current, True

    def get_best_two_insertions(self, solution, point):
        '''
        get two best insertion places
        '''
        insertion_values = []
        for route_idx, item in enumerate(solution):
            nodes = item["nodes"]
            for insert_pos in range(1, len(nodes)):
                candidate = copy.deepcopy(solution)
                candidate[route_idx]["nodes"].insert(insert_pos, point)

                route_eval = self.evaluate_route(
                    route_nodes=candidate[route_idx]["nodes"],
                    route_start=candidate[route_idx]["start_time"]
                )
                if not route_eval.feasible:
                    continue

                candidate_value = self.evaluate_solution(candidate)
                insertion_values.append(
                    (candidate_value, candidate, route_idx, insert_pos)
                )
        insertion_values.sort(key=lambda x: x[0])
        return insertion_values[:2]

    def regret_value(self, first_value, second_value):
        '''
        comparison of first and second places
        '''
        first_duration, first_distance = first_value
        second_duration, second_distance = second_value

        return (
            float(second_duration) - float(first_duration),
            float(second_distance) - float(first_distance)
        )

    def regret_insertion(self, partial_solution, removed_points):
        '''
        regret-2 insertion
        '''
        current = copy.deepcopy(partial_solution)
        points_to_insert = removed_points[:]

        while points_to_insert:
            best_point = None
            best_candidate_solution = None
            best_regret = (-float("inf"), -float("inf"))

            for customer in points_to_insert:
                best_two = self.get_best_two_insertions(current, customer)

                if len(best_two) == 0:
                    continue

                first_value, first_solution, _, _ = best_two[0]

                if len(best_two) == 1:
                    regret = (float("inf"), float("inf"))
                else:
                    second_value, _, _, _ = best_two[1]
                    regret = self.regret_value(first_value, second_value)

                if regret > best_regret:
                    best_regret = regret
                    best_point = customer
                    best_candidate_solution = first_solution

            if best_candidate_solution is None:
                return partial_solution, False

            current = best_candidate_solution
            points_to_insert.remove(best_point)

        current = self.remove_empty_routes(current)
        return current, True

    def roulette_select(self, weights):
        '''
        select operator by roulette wheel
        '''
        total = sum(weights.values())
        if total <= 0:
            return self.rng.choice(list(weights.keys()))

        r = self.rng.random() * total
        cumulative = 0.0

        for name, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return name

        return list(weights.keys())[-1]

    def update_operator_weights(self, weights, scores, counts,reaction_factor):
        '''
        weight update
        '''
        new_weights = {}

        for op_name in weights:
            old_weight = weights[op_name]
            usage = counts[op_name]

            if usage > 0:
                avg_score = scores[op_name] / usage
                new_weight = (1 - reaction_factor) * old_weight + reaction_factor * avg_score
            else:
                new_weight = old_weight

            new_weights[op_name] = max(0.1, float(new_weight))

        return new_weights

    def apply_destroy(self, operator_name, solution, q):
        '''
        call destroy operator
        '''
        if operator_name == "random_removal":
            return self.random_removal(solution, q)
        elif operator_name == "worst_removal":
            return self.worst_removal(solution, q)
        else:
            raise ValueError(f"wrong destroy operator{operator_name}")

    def apply_repair(self, operator_name, partial_solution, removed_points):
        '''
        call repair operator
        '''
        if operator_name == "greedy_insertion":
            return self.greedy_insertion(partial_solution, removed_points)
        elif operator_name == "regret_insertion":
            return self.regret_insertion(partial_solution, removed_points)
        else:
            raise ValueError(f"wrong repair operator {operator_name}")

    def is_complete_solution(self, solution) -> bool:
        '''
        check if every customer is served exactly once
        '''
        customers = self.get_all_points(solution)
        expected = set(range(1, len(self.day_stops) + 1))
        return set(customers) == expected and len(customers) == len(expected)

    def main_alns(
        self,
        initial_routes,
        route_cls,
        max_iterations: int = 300,
        q_min: int = 1,
        q_max: int = 5,
        initial_temperature: float = 50.0,
        cooling_rate: float = 0.995,
        reaction_factor: float = 0.2,
        segment_length: int = 25,
        score_global_best: float = 5.0,
        score_improved_current: float = 3.0,
        score_accepted_worse: float = 1.0):
        '''
        main function to run ALNS
        '''
        current_solution = self.routes_to_solution(initial_routes)
        current_solution = self.remove_empty_routes(current_solution)
        current_value = self.evaluate_solution(current_solution)

        if current_value[0] == float("inf"):
            raise InfeasibleError("time is inf, ALNS can't be applied")

        if not self.is_complete_solution(current_solution):
            raise InfeasibleError("initial solution is incomplete")

        best_solution = copy.deepcopy(current_solution)
        best_value = current_value

        temperature = float(initial_temperature)

        destroy_weights = {
            "random_removal": 1.0,
            "worst_removal": 1.0,
        }

        repair_weights = {
            "greedy_insertion": 1.0,
            "regret_insertion": 1.0,
        }

        destroy_scores = {name: 0.0 for name in destroy_weights}
        destroy_counts = {name: 0 for name in destroy_weights}

        repair_scores = {name: 0.0 for name in repair_weights}
        repair_counts = {name: 0 for name in repair_weights}

        for iteration in range(1, max_iterations + 1):
            total_customers = self.count_points(current_solution)
            if total_customers == 0:
                break

            q = self.choose_pt_remove(total_customers, q_min=q_min, q_max=q_max)

            destroy_name = self.roulette_select(destroy_weights)
            repair_name = self.roulette_select(repair_weights)

            destroy_counts[destroy_name] += 1
            repair_counts[repair_name] += 1

            partial_solution, removed_points = self.apply_destroy(
                operator_name=destroy_name,
                solution=current_solution,
                q=q
            )

            candidate_solution, success = self.apply_repair(
                operator_name=repair_name,
                partial_solution=partial_solution,
                removed_points=removed_points
            )

            if not success:
                temperature *= cooling_rate
                if iteration % segment_length == 0:
                    destroy_weights = self.update_operator_weights(
                        destroy_weights, destroy_scores, destroy_counts, reaction_factor
                    )
                    repair_weights = self.update_operator_weights(
                        repair_weights, repair_scores, repair_counts, reaction_factor
                    )
                    destroy_scores = {name: 0.0 for name in destroy_weights}
                    destroy_counts = {name: 0 for name in destroy_weights}
                    repair_scores = {name: 0.0 for name in repair_weights}
                    repair_counts = {name: 0 for name in repair_weights}
                continue

            if not self.is_complete_solution(candidate_solution):
                temperature *= cooling_rate
                continue

            candidate_value = self.evaluate_solution(candidate_solution)

            if candidate_value[0] == float("inf"):
                temperature *= cooling_rate
                continue

            if candidate_value < best_value:
                best_solution = copy.deepcopy(candidate_solution)
                best_value = candidate_value
                destroy_scores[destroy_name] += score_global_best
                repair_scores[repair_name] += score_global_best

            accepted = self.accept_solution(
                current_value=current_value,
                candidate_value=candidate_value,
                temperature=temperature
            )

            if accepted:
                if candidate_value < current_value:
                    destroy_scores[destroy_name] += score_improved_current
                    repair_scores[repair_name] += score_improved_current
                else:
                    destroy_scores[destroy_name] += score_accepted_worse
                    repair_scores[repair_name] += score_accepted_worse

                current_solution = candidate_solution
                current_value = candidate_value

            temperature *= cooling_rate

            if iteration % segment_length == 0:
                destroy_weights = self.update_operator_weights(
                    destroy_weights, destroy_scores, destroy_counts, reaction_factor
                )
                repair_weights = self.update_operator_weights(
                    repair_weights, repair_scores, repair_counts, reaction_factor
                )

                destroy_scores = {name: 0.0 for name in destroy_weights}
                destroy_counts = {name: 0 for name in destroy_weights}
                repair_scores = {name: 0.0 for name in repair_weights}
                repair_counts = {name: 0 for name in repair_weights}

        best_routes = self.solution_to_routes(best_solution, route_cls=route_cls)
        return best_routes, best_value
