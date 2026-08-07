import random
import math
import copy
import time

class LocalSearchAlgorithms:
    def __init__(self, env):
        self.env = env
        self.history = []

    def _get_neighbors(self, state, step_size=5):
        neighbors = []
        for i in range(self.env.num_intersections):
            for delta in [-step_size, step_size]:
                new_state = list(state)
                new_state[i] = (new_state[i] + delta) % self.env.cycle_time
                neighbors.append(new_state)
        return neighbors

    def _get_random_neighbor(self, state, step_size=15):
        new_state = list(state)
        idx = random.randint(0, self.env.num_intersections - 1)
        delta = random.choice([-step_size, step_size])
        new_state[idx] = (new_state[idx] + delta) % self.env.cycle_time
        return new_state

    def steepest_ascent_hill_climbing(self, max_iter=1000):
        current_state = self.env.generate_random_state()
        current_cost = self.env.objective_function(current_state)
        self.history = [(current_state, current_cost)]
        
        start_time = time.time()
        for iteration in range(max_iter):
            neighbors = self._get_neighbors(current_state)
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in neighbors:
                cost = self.env.objective_function(neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
                    
            if best_cost >= current_cost:
                break
                
            current_state = best_neighbor
            current_cost = best_cost
            self.history.append((current_state, current_cost))
            
        return current_state, current_cost, time.time() - start_time, len(self.history)

    def sideways_move_hill_climbing(self, max_iter=1000, max_sideways=50):
        current_state = self.env.generate_random_state()
        current_cost = self.env.objective_function(current_state)
        self.history = [(current_state, current_cost)]
        
        sideways_count = 0
        start_time = time.time()
        for iteration in range(max_iter):
            neighbors = self._get_neighbors(current_state)
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in neighbors:
                cost = self.env.objective_function(neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
                    
            if best_cost > current_cost:
                break
            elif best_cost == current_cost:
                sideways_count += 1
                if sideways_count > max_sideways:
                    break
            else:
                sideways_count = 0
                
            current_state = best_neighbor
            current_cost = best_cost
            self.history.append((current_state, current_cost))
            
        return current_state, current_cost, time.time() - start_time, len(self.history)

    def stochastic_hill_climbing(self, max_iter=1000):
        current_state = self.env.generate_random_state()
        current_cost = self.env.objective_function(current_state)
        self.history = [(current_state, current_cost)]
        
        start_time = time.time()
        for iteration in range(max_iter):
            neighbors = self._get_neighbors(current_state)
            better_neighbors = []
            
            for neighbor in neighbors:
                cost = self.env.objective_function(neighbor)
                if cost < current_cost:
                    better_neighbors.append((neighbor, cost))
                    
            if not better_neighbors:
                break
                
            selected = random.choice(better_neighbors)
            current_state = selected[0]
            current_cost = selected[1]
            self.history.append((current_state, current_cost))
            
        return current_state, current_cost, time.time() - start_time, len(self.history)

    def random_restart_hill_climbing(self, max_restarts=10, max_iter_per_restart=100):
        best_overall_state = None
        best_overall_cost = float('inf')
        total_history = []
        
        start_time = time.time()
        for restart in range(max_restarts):
            state, cost, _, _ = self.steepest_ascent_hill_climbing(max_iter_per_restart)
            total_history.extend(self.history)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_state = state
                
        self.history = total_history
        return best_overall_state, best_overall_cost, time.time() - start_time, len(self.history)

    def simulated_annealing(self, initial_temp=10000.0, cooling_rate=0.99, min_temp=0.1, max_iter_per_temp=20):
        current_state = self.env.generate_random_state()
        current_cost = self.env.objective_function(current_state)
        best_state = list(current_state)
        best_cost = current_cost
        self.history = [(current_state, current_cost, initial_temp)]
        
        temp = initial_temp
        start_time = time.time()
        
        while temp > min_temp:
            for _ in range(max_iter_per_temp):
                neighbor = self._get_random_neighbor(current_state)
                neighbor_cost = self.env.objective_function(neighbor)
                
                delta_e = neighbor_cost - current_cost
                
                if delta_e < 0:
                    current_state = neighbor
                    current_cost = neighbor_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_state = list(current_state)
                else:
                    probability = math.exp(-delta_e / temp)
                    if random.random() < probability:
                        current_state = neighbor
                        current_cost = neighbor_cost
                        
                self.history.append((current_state, current_cost, temp))
            temp *= cooling_rate
            
        return best_state, best_cost, time.time() - start_time, len(self.history)