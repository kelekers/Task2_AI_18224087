import random
import math

class GreenWaveEnvironment:
    def __init__(self, num_intersections=20, cycle_time=90, green_duration=45):
        self.num_intersections = num_intersections
        self.cycle_time = cycle_time
        self.green_duration = green_duration
        self.red_duration = cycle_time - green_duration
        self.routes = self._generate_complex_routes()
        self.departures = self._generate_traffic_volume()

    def _generate_complex_routes(self):
        return [
            [(0, 0), (1, 45), (2, 25), (3, 60), (4, 30), (5, 40), (6, 55)],
            [(7, 0), (8, 35), (2, 50), (9, 20), (10, 45), (11, 30), (12, 40)],
            [(13, 0), (1, 55), (14, 25), (15, 35), (9, 40), (16, 50), (17, 35)],
            [(17, 0), (10, 30), (3, 40), (1, 60), (0, 20), (18, 45), (19, 25)],
            [(6, 0), (4, 30), (9, 45), (14, 30), (13, 20), (7, 50)],
            [(12, 0), (2, 40), (1, 35), (7, 50), (19, 30), (15, 25)],
            [(18, 0), (5, 35), (3, 25), (8, 45), (11, 55), (16, 20)]
        ]

    def _generate_traffic_volume(self):
        return {
            0: [t for t in range(0, 7200, 30)],
            1: [t for t in range(15, 7200, 45)],
            2: [t for t in range(5, 7200, 20)],
            3: [t for t in range(20, 7200, 60)],
            4: [t for t in range(10, 7200, 40)],
            5: [t for t in range(0, 7200, 75)],
            6: [t for t in range(25, 7200, 35)]
        }

    def generate_random_state(self):
        return [random.randint(0, self.cycle_time - 1) for _ in range(self.num_intersections)]

    def objective_function(self, state):
        total_cost = 0.0
        for route_idx, route in enumerate(self.routes):
            departure_times = self.departures.get(route_idx, [])
            for start_time in departure_times:
                current_time = start_time
                consecutive_stops = 0
                route_cost = 0.0
                for intersection_id, travel_time in route:
                    current_time += travel_time
                    offset = state[intersection_id]
                    relative_time = (current_time - offset) % self.cycle_time
                    
                    if relative_time >= self.green_duration:
                        wait_time = self.cycle_time - relative_time
                        queue_delay = math.log1p(wait_time) * 2.5
                        acceleration_penalty = 12.0
                        total_wait = wait_time + queue_delay + acceleration_penalty
                        
                        route_cost += total_wait
                        current_time += total_wait
                        consecutive_stops += 1
                        
                        if consecutive_stops > 1:
                            route_cost += (consecutive_stops ** 2) * 4.0
                    else:
                        consecutive_stops = max(0, consecutive_stops - 1)
                
                total_cost += route_cost
        return total_cost

    def evaluate_bottlenecks(self, state):
        intersection_wait_times = {i: 0 for i in range(self.num_intersections)}
        for route_idx, route in enumerate(self.routes):
            departure_times = self.departures.get(route_idx, [])
            for start_time in departure_times:
                current_time = start_time
                for intersection_id, travel_time in route:
                    current_time += travel_time
                    offset = state[intersection_id]
                    relative_time = (current_time - offset) % self.cycle_time
                    
                    if relative_time >= self.green_duration:
                        wait_time = self.cycle_time - relative_time
                        intersection_wait_times[intersection_id] += wait_time
                        current_time += wait_time
        return intersection_wait_times