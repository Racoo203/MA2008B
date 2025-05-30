import numpy as np
from tqdm import tqdm
import random

class VRP_Agent:
    def __init__(self, nodes, distances, demands, capacity, n_ants=10, n_iter=100, alpha=1.0, beta=2.0, rho=0.5, Q=100):
        self.nodes = nodes  # List of node indices
        self.distances = distances  # Distance matrix
        self.demands = demands  # Dict: {(node, species): demand}
        self.capacity = capacity  # Truck capacity
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone = np.ones_like(distances)
        self.best_solution = None
        self.best_cost = float('inf')

    def _select_next_node(self, current_node, visited, remaining_capacity):
        probabilities = []
        total = 0
        for j in self.nodes:
            if j in visited or self.demands.get(j, 0) > remaining_capacity:
                probabilities.append(0)
                continue
            tau = self.pheromone[current_node][j] ** self.alpha
            eta = (1 / self.distances[current_node][j]) ** self.beta
            prob = tau * eta
            probabilities.append(prob)
            total += prob

        if total == 0:
            return None

        probabilities = [p / total for p in probabilities]
        return random.choices(self.nodes, weights=probabilities)[0]

    def _construct_solution(self):
        routes = []
        for _ in range(self.n_ants):
            route = []
            current_node = 0  # Start at depot
            visited = set([0])
            capacity = self.capacity
            sub_route = [0]

            while True:
                next_node = self._select_next_node(current_node, visited, capacity)
                if next_node is None:
                    sub_route.append(0)  # Return to depot
                    routes.append(sub_route)
                    visited = set([0])
                    capacity = self.capacity
                    sub_route = [0]
                    current_node = 0
                    if len(visited) == len(self.nodes):
                        break
                    continue
                sub_route.append(next_node)
                visited.add(next_node)
                capacity -= self.demands.get(next_node, 0)
                current_node = next_node

            if sub_route:
                routes.append(sub_route)

        return routes

    def _calculate_cost(self, routes):
        total_distance = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += self.distances[route[i]][route[i+1]]
        return total_distance

    def _update_pheromones(self, routes, cost):
        self.pheromone *= (1 - self.rho)
        for route in routes:
            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i+1]] += self.Q / cost

    def run(self):
        for _ in tqdm(range(self.n_iter), desc="VRP Ant Colony Optimization Progress"):
            routes = self._construct_solution()
            cost = self._calculate_cost(routes)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = routes
            self._update_pheromones(routes, cost)
        return self.best_solution, self.best_cost