# genetic_algorithm.py
# --- MODIFIED FOR EXPERIMENTS ---
# Added 'mode' for switching between 'lstm' and 'standard' mutation.
# Modified 'evaluate_fitness' to return a dictionary of costs for logging.

import numpy as np
from typing import List, Dict, Optional, Union

class Chromosome:
    def __init__(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self.fitness = 0.0
        # Store detailed costs for analysis
        self.costs = {'distance': 0, 'turning': 0, 'collision': 0, 'penalty': 0}

class GeneticAlgorithm:
    def __init__(self, ga_params: dict, mode: str = 'lstm'):
        self.population_size = ga_params.get('population_size', 100)
        self.num_waypoints = ga_params.get('num_waypoints', 8)
        self.mutation_rate = ga_params.get('mutation_rate', 0.1)
        self.elite_ratio = ga_params.get('elite_ratio', 0.1)
        self.mode = mode # 'lstm' or 'standard'

        self.bounds_min = np.array([0, 0, 50])
        self.bounds_max = np.array([1000, 1000, 500])
        
        self.population = self._initialize_population()
        print(f"[GA] Initialized in '{self.mode}' mode.")

    def _initialize_population(self):
        pop = []
        for _ in range(self.population_size):
            waypoints = np.random.rand(self.num_waypoints, 3)
            waypoints = waypoints * (self.bounds_max - self.bounds_min) + self.bounds_min
            pop.append(Chromosome(waypoints=waypoints))
        return pop

    def standard_mutate(self, chromosome: Chromosome) -> Chromosome:
        """A standard mutation operator that adds random noise."""
        mutated_waypoints = chromosome.waypoints.copy()
        for i in range(1, len(mutated_waypoints) - 1): # Don't mutate start/end
            if np.random.rand() < 0.1: # 10% chance to mutate a given waypoint
                mutation_vector = (np.random.rand(3) - 0.5) * 50.0 # Random nudge
                new_position = mutated_waypoints[i] + mutation_vector
                mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
        return Chromosome(waypoints=mutated_waypoints)

    def predictive_mutate(self, chromosome: Chromosome, lstm_predictions: dict) -> Chromosome:
        """Predictive mutation using LSTM forecasts (sources 34-57)"""
        mutated_waypoints = chromosome.waypoints.copy()
        predicted_obstacles = lstm_predictions.get('predicted_obstacles', [])
        confidence_scores = lstm_predictions.get('confidence_scores', [])
        
        for i, waypoint in enumerate(mutated_waypoints[1:-1], 1):
            max_threat = 0.0
            threat_direction = np.zeros(3)
            
            for j, pred_obstacle in enumerate(predicted_obstacles):
                pred_pos = np.array([pred_obstacle['x'], pred_obstacle['y'], pred_obstacle['z']])
                distance = np.linalg.norm(waypoint - pred_pos)
                safety_radius = 50.0
                
                if distance < safety_radius and distance > 1e-6:
                    threat_level = (safety_radius - distance) / safety_radius
                    confidence = confidence_scores[j] if j < len(confidence_scores) else 0.5
                    
                    if threat_level * confidence > max_threat:
                        max_threat = threat_level * confidence
                        threat_direction = (waypoint - pred_pos) / distance
            
            if max_threat > 0.3:
                avoidance_vector = threat_direction * max_threat * 100.0
                new_position = waypoint + avoidance_vector
                mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
                
        return Chromosome(waypoints=mutated_waypoints)

    def evaluate_fitness(self, chromosome: Chromosome, obstacles: List[dict], 
                         lstm_predictions: Optional[dict] = None) -> (float, dict):
        """Returns total fitness and a dictionary of cost components."""
        waypoints = chromosome.waypoints
        
        total_distance = sum(np.linalg.norm(waypoints[i+1] - waypoints[i]) for i in range(len(waypoints) - 1))
        distance_cost = total_distance / 1000.0
        
        turning_cost = 0.0
        for i in range(1, len(waypoints) - 1):
            v1 = waypoints[i] - waypoints[i-1]
            v2 = waypoints[i+1] - waypoints[i]
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                turning_cost += angle ** 2
        
        collision_cost = 0.0
        for obstacle in obstacles:
            obs_pos = np.array([obstacle['x'], obstacle['y'], obstacle['z']])
            for waypoint in waypoints:
                distance = np.linalg.norm(waypoint - obs_pos)
                safety_radius = 30.0 + obstacle.get('radius', 20.0)
                if distance < safety_radius:
                    collision_cost += (safety_radius - distance) ** 2
        
        prediction_penalty = 0.0
        if lstm_predictions:
             prediction_penalty = lstm_predictions.get('confidence_penalty', 0.0)
        
        costs = {
            'distance': distance_cost,
            'turning': turning_cost,
            'collision': collision_cost,
            'penalty': prediction_penalty
        }

        total_fitness = (0.4 * costs['distance'] + 0.2 * costs['turning'] + 
                         0.3 * costs['collision'] + 0.1 * costs['penalty'])
        
        return total_fitness, costs

    def evolve_generation(self, obstacles: List[dict], lstm_predictions: dict):
        """Runs one generation of the GA."""
        for chrom in self.population:
            chrom.fitness, chrom.costs = self.evaluate_fitness(chrom, obstacles, lstm_predictions)
        
        self.population.sort(key=lambda x: x.fitness)
        elite_size = int(self.population_size * self.elite_ratio)
        new_population = self.population[:elite_size]
        
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(self.population[:elite_size], 2, replace=False)
            crossover_point = np.random.randint(1, self.num_waypoints - 1)
            child_waypoints = np.vstack((parent1.waypoints[:crossover_point], parent2.waypoints[crossover_point:]))
            child = Chromosome(child_waypoints)
            
            if np.random.rand() < self.mutation_rate:
                if self.mode == 'lstm':
                    child = self.predictive_mutate(child, lstm_predictions)
                else: # 'standard' mode
                    child = self.standard_mutate(child)
            
            new_population.append(child)
            
        self.population = new_population