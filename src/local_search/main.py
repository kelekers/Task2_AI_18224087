import random
import time
import sys
import json
from environment import GreenWaveEnvironment
from algorithms import LocalSearchAlgorithms

class GeneticAlgorithm:
    def __init__(self, env, pop_size=100, mutation_rate=0.15, crossover_rate=0.85):
        self.env = env
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.history = []

    def _initialize_population(self):
        return [self.env.generate_random_state() for _ in range(self.pop_size)]

    def _calculate_fitness(self, population):
        costs = [self.env.objective_function(ind) for ind in population]
        max_cost = max(costs) if costs else 1
        fitnesses = [(max_cost - cost) + 1e-6 for cost in costs]
        total_fitness = sum(fitnesses)
        return costs, [fit / total_fitness for fit in fitnesses]

    def _tournament_selection(self, population, costs, k=5):
        selected = random.sample(list(zip(population, costs)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return list(parent1), list(parent2)
        
        point = random.randint(1, self.env.num_intersections - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual):
        for i in range(self.env.num_intersections):
            if random.random() < self.mutation_rate:
                mutation_step = random.randint(-15, 15)
                individual[i] = (individual[i] + mutation_step) % self.env.cycle_time
        return individual

    def run(self, max_generations=200):
        population = self._initialize_population()
        best_overall_state = None
        best_overall_cost = float('inf')
        
        start_time = time.time()
        for generation in range(max_generations):
            costs, fitnesses = self._calculate_fitness(population)
            
            min_cost = min(costs)
            min_idx = costs.index(min_cost)
            if min_cost < best_overall_cost:
                best_overall_cost = min_cost
                best_overall_state = list(population[min_idx])
                
            self.history.append((list(best_overall_state), best_overall_cost))
            
            new_population = [list(best_overall_state)]
            
            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(population, costs)
                p2 = self._tournament_selection(population, costs)
                
                c1, c2 = self._crossover(p1, p2)
                
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population.extend([c1, c2])
                
            population = new_population[:self.pop_size]
            
        return best_overall_state, best_overall_cost, time.time() - start_time, len(self.history)

def run_cli():
    env = GreenWaveEnvironment(num_intersections=20, cycle_time=90, green_duration=45)
    search_algs = LocalSearchAlgorithms(env)
    ga = GeneticAlgorithm(env)

    print("="*50)
    print("CITY-WIDE GREEN WAVE OPTIMIZATION - LOCAL SEARCH")
    print("="*50)
    print("1. Steepest-Ascent Hill Climbing")
    print("2. Sideways Move Hill Climbing")
    print("3. Stochastic Hill Climbing")
    print("4. Random Restart Hill Climbing")
    print("5. Simulated Annealing")
    print("6. Genetic Algorithm")
    print("0. Exit")
    print("="*50)
    
    choice = input("Select algorithm (0-6): ")
    
    if choice == '0':
        sys.exit()
        
    print("\nExecuting algorithm... Please wait.\n")
    
    if choice == '1':
        state, cost, duration, iters = search_algs.steepest_ascent_hill_climbing()
    elif choice == '2':
        state, cost, duration, iters = search_algs.sideways_move_hill_climbing()
    elif choice == '3':
        state, cost, duration, iters = search_algs.stochastic_hill_climbing()
    elif choice == '4':
        state, cost, duration, iters = search_algs.random_restart_hill_climbing()
    elif choice == '5':
        state, cost, duration, iters = search_algs.simulated_annealing()
    elif choice == '6':
        state, cost, duration, iters = ga.run()
    else:
        print("Invalid choice.")
        return

    print(f"--- RESULTS ---")
    print(f"Final Objective Cost : {cost:.4f}")
    print(f"Execution Time       : {duration:.4f} seconds")
    print(f"Total Iterations     : {iters}")
    print(f"Final State (Offsets): {state}")
    
    if hasattr(search_algs, 'history') and choice in ['1','2','3','4','5']:
        history_data = search_algs.history
        print(f"Initial Cost         : {history_data[0][1]:.4f}")
    elif choice == '6':
        history_data = ga.history
        print(f"Initial Cost         : {history_data[0][1]:.4f}")
    else:
        history_data = []

    if history_data:
        log_filename = "experiment_log.json"
        try:
            with open(log_filename, "w") as f:
                json.dump(history_data, f)
            print(f"\n[INFO] Log eksperimen berhasil disimpan ke '{log_filename}'")
        except Exception as e:
            print(f"\n[ERROR] Gagal menyimpan log: {e}")

if __name__ == "__main__":
    run_cli()