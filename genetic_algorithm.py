# genetic_algorithm.py
# GA with predictive mutation (sources 34-57, 66-96)

import numpy as np
from typing import List, Dict, Optional

# Placeholder Chromosome Class (implied by sources 34, 57, 66)
class Chromosome:
    def __init__(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self.fitness = 0.0

# GA Class (implied by source 215)
class GeneticAlgorithm:
    def __init__(self, ga_params: dict):
        self.population_size = ga_params.get('population_size', 100)
        self.num_waypoints = ga_params.get('num_waypoints', 8)
        self.mutation_rate = ga_params.get('mutation_rate', 0.1)
        self.elite_ratio = ga_params.get('elite_ratio', 0.1)
        
        # Define simulation bounds (placeholder)
        self.bounds_min = np.array([0, 0, 50])
        self.bounds_max = np.array([1000, 1000, 500])
        
        self.population = self._initialize_population()

    def _initialize_population(self):
        """Creates an initial random population of chromosomes."""
        pop = []
        for _ in range(self.population_size):
            waypoints = np.random.rand(self.num_waypoints, 3)
            waypoints = waypoints * (self.bounds_max - self.bounds_min) + self.bounds_min
            pop.append(Chromosome(waypoints=waypoints))
        return pop

    def predictive_mutate(self, chromosome: Chromosome, lstm_predictions: dict) -> Chromosome:
        """Predictive mutation using LSTM forecasts (sources 34-57)"""
        mutated_waypoints = chromosome.waypoints.copy()
        predicted_obstacles = lstm_predictions.get('predicted_obstacles', [])
        confidence_scores = lstm_predictions.get('confidence_scores', [])
        
        for i, waypoint in enumerate(mutated_waypoints[1:-1], 1): # Don't mutate start/end
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
                avoidance_vector = threat_direction * max_threat * 100.0 # 100.0 = avoidance strength
                new_position = waypoint + avoidance_vector
                mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
                
        return Chromosome(waypoints=mutated_waypoints)

    def evaluate_fitness(self, chromosome: Chromosome, obstacles: List[dict], 
                         lstm_predictions: Optional[dict] = None) -> float:
        """Multi-objective fitness function (sources 66-96)"""
        waypoints = chromosome.waypoints
        
        # F1: Path distance cost (sources 68-72)
        total_distance = sum(np.linalg.norm(waypoints[i+1] - waypoints[i])
                             for i in range(len(waypoints) - 1))
        distance_cost = total_distance / 1000.0
        
        # F2: Energy cost from turning (sources 73-83)
        turning_cost = 0.0
        for i in range(1, len(waypoints) - 1):
            v1 = waypoints[i] - waypoints[i-1]
            v2 = waypoints[i+1] - waypoints[i]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) # Fixed from doc's '1, 1'
                turning_cost += angle ** 2
        
        # F3: Collision risk (sources 84-94)
        collision_cost = 0.0
        for obstacle in obstacles:
            obs_pos = np.array([obstacle['x'], obstacle['y'], obstacle['z']])
            for waypoint in waypoints:
                distance = np.linalg.norm(waypoint - obs_pos)
                safety_radius = 30.0 + obstacle.get('radius', 20.0)
                if distance < safety_radius:
                    collision_cost += (safety_radius - distance) ** 2
        
        # F4: Prediction confidence penalty (source 95)
        prediction_penalty = 0.0
        if lstm_predictions:
             prediction_penalty = lstm_predictions.get('confidence_penalty', 0.0)
        
        # Weighted sum (from source 96)
        fitness = (0.4 * distance_cost + 
                   0.2 * turning_cost + 
                   0.3 * collision_cost + 
                   0.1 * prediction_penalty)
        return fitness

    def evolve_generation(self, obstacles: List[dict], lstm_predictions: dict):
        """
        Placeholder method (implied by source 215) to run one GA generation.
        """
        # 1. Evaluate fitness for all in population
        for chrom in self.population:
            chrom.fitness = self.evaluate_fitness(chrom, obstacles, lstm_predictions)
        
        # 2. Selection (Elitism)
        self.population.sort(key=lambda x: x.fitness) # Lower fitness (cost) is better
        elite_size = int(self.population_size * self.elite_ratio)
        new_population = self.population[:elite_size]
        
        # 3. Crossover & Mutation
        while len(new_population) < self.population_size:
            # Simple crossover (placeholder)
            parent1 = np.random.choice(self.population[:elite_size])
            parent2 = np.random.choice(self.population[:elite_size])
            crossover_point = np.random.randint(1, self.num_waypoints - 1)
            child_waypoints = np.vstack((parent1.waypoints[:crossover_point],
                                         parent2.waypoints[crossover_point:]))
            child = Chromosome(child_waypoints)
            
            # 4. Mutation (using predictive_mutate)
            if np.random.rand() < self.mutation_rate:
                child = self.predictive_mutate(child, lstm_predictions)
                
            new_population.append(child)
            
        self.population = new_population