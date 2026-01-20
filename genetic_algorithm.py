# genetic_algorithm.py
# --- FINAL VERSION ---

import numpy as np
from typing import List, Dict, Optional

# Define global ranges
SWARM_RADIUS = 200.0         # A loose radius for the whole swarm
CRITICAL_LINK_RADIUS = 50.0  # A *very strict* 50m safety margin for the 0-1 link

class Chromosome:
    """Represents a single solution (a path) in the simulation."""
    def __init__(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self.fitness = 0.0
        self.costs = {'distance': 0, 'turning': 0, 'collision': 0, 
                      'penalty': 0, 'cohesion': 0, 'pdr_penalty': 0}

def evaluate_fitness(waypoints: np.ndarray, obstacles: List[dict], 
                     lstm_predictions: Optional[dict] = None,
                     all_uav_positions: Optional[List[np.ndarray]] = None,
                     current_uav_id: Optional[int] = None,
                     network_performance: Optional[Dict] = None
                     ) -> tuple[float, dict]:
    """
    Standalone fitness function.
    Uses a Hybrid Cohesion model with a strict 50m safety margin.
    """
    
    # F1: Distance Cost
    total_distance = sum(np.linalg.norm(waypoints[i+1] - waypoints[i]) for i in range(len(waypoints) - 1))
    distance_cost = total_distance / 1000.0
    
    # F2: Turning Cost
    turning_cost = 0.0
    for i in range(1, len(waypoints) - 1):
        v1, v2 = waypoints[i] - waypoints[i-1], waypoints[i+1] - waypoints[i]
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 > 1e-6 and norm_v2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            turning_cost += np.arccos(np.clip(cos_angle, -1.0, 1.0)) ** 2
            
    # F3: Collision Cost
    collision_cost = 0.0
    for obstacle in obstacles:
        obs_pos = np.array([obstacle['x'], obstacle['y'], obstacle['z']])
        for waypoint in waypoints:
            distance = np.linalg.norm(waypoint - obs_pos)
            safety_radius = 30.0 + obstacle.get('radius', 20.0)
            if distance < safety_radius:
                collision_cost += (safety_radius - distance) ** 2
                
    # F4: Prediction Penalty
    prediction_penalty = 0.0
    if lstm_predictions:
         prediction_penalty = lstm_predictions.get('confidence_penalty', 0.0)
         
    # F5: Hybrid Cohesion Cost
    cohesion_cost = 0.0
    swarm_cohesion_cost = 0.0
    link_cohesion_cost = 0.0

    if all_uav_positions is not None and current_uav_id is not None:
        
        # 1. Calculate Swarm Center (for all UAVs)
        swarm_center = np.mean(all_uav_positions, axis=0)
        
        # 2. Apply Swarm Cohesion (Loose 200m radius)
        for wp in waypoints:
            dist_to_center = np.linalg.norm(wp - swarm_center)
            if dist_to_center > SWARM_RADIUS: 
                swarm_cohesion_cost += (dist_to_center - SWARM_RADIUS) / 100.0 

        # 3. Apply specific LINK Cohesion to nodes 0 and 1
        teammate_id = -1
        if current_uav_id == 0: teammate_id = 1
        elif current_uav_id == 1: teammate_id = 0
            
        if teammate_id != -1:
            teammate_pos = all_uav_positions[teammate_id]
            for wp in waypoints:
                dist_to_teammate = np.linalg.norm(wp - teammate_pos)
                if dist_to_teammate > CRITICAL_LINK_RADIUS: 
                    link_cohesion_cost += (dist_to_teammate - CRITICAL_LINK_RADIUS) / 50.0 
        
        cohesion_cost = swarm_cohesion_cost + link_cohesion_cost
         
    # F6: PDR (Packet Delivery Ratio) Penalty
    pdr_penalty = 0.0
    if network_performance is not None:
        pdr = network_performance.get('packet_delivery_ratio', 0.0)
        pdr_penalty = 1.0 - pdr
    else:
        pdr_penalty = 0.0

    costs = {'distance': distance_cost, 'turning': turning_cost, 
             'collision': collision_cost, 'penalty': prediction_penalty,
             'cohesion': cohesion_cost, 'pdr_penalty': pdr_penalty}
    
    # Final Fitness Weights (PDR + Cohesion = 65%)
    total_fitness = (0.10 * costs['distance'] +
                     0.10 * costs['turning'] +
                     0.10 * costs['collision'] +
                     0.05 * costs['penalty'] +
                     0.25 * costs['cohesion'] +
                     0.40 * costs['pdr_penalty'])
                     
    return total_fitness, costs

class GeneticAlgorithm:
    def __init__(self, ga_params: dict, mode: str = 'lstm'):
        self.population_size = ga_params.get('population_size', 100)
        self.num_waypoints = ga_params.get('num_waypoints', 8)
        self.mutation_rate = ga_params.get('mutation_rate', 0.1)
        self.elite_ratio = ga_params.get('elite_ratio', 0.1)
        self.mode = mode
        self.bounds_min = np.array([0, 0, 50])
        self.bounds_max = np.array([1000, 1000, 500])
        self.population = self._initialize_population()
        # Track current position (for interface consistency with RRTAgent)
        self.current_pos = self.population[0].waypoints[0]
        # Track last fitness and costs (for interface consistency with RRTAgent)
        self.last_fitness = 0.0
        self.last_costs = {}
        print(f"[GA] Initialized in '{self.mode}' mode.")
    
    def advance_to_next_step(self, best_next_position: np.ndarray):
        """
        Implements a receding horizon. 
        Updates the entire population to a new starting position.
        """
        self.current_pos = best_next_position
        for chrom in self.population:
            chrom.waypoints[:-1] = chrom.waypoints[1:]
            chrom.waypoints[0] = best_next_position
            last_pos = chrom.waypoints[-2]
            random_vec = (np.random.rand(3) - 0.5) * 50.0
            new_last_pos = last_pos + random_vec
            chrom.waypoints[-1] = np.clip(new_last_pos, self.bounds_min, self.bounds_max)

    def _is_position_safe(self, pos: np.ndarray, obstacles: List[dict]) -> bool:
        for obstacle in obstacles:
            obs_pos = np.array([obstacle['x'], obstacle['y'], obstacle['z']])
            distance = np.linalg.norm(pos - obs_pos)
            safety_radius = 30.0 + obstacle.get('radius', 20.0)
            if distance < safety_radius:
                return False
        return True

    def _initialize_population(self):
        pop = []
        start_pos = np.array([50.0, 50.0, 100.0])
        goal_pos = np.array([950.0, 950.0, 100.0])
        for _ in range(self.population_size):
            base_path = np.linspace(start_pos, goal_pos, self.num_waypoints)
            noise = (np.random.rand(self.num_waypoints, 3) - 0.5) * 50.0
            noise[0] = 0
            noise[-1] = 0
            waypoints = base_path + noise
            waypoints = np.clip(waypoints, self.bounds_min, self.bounds_max)
            pop.append(Chromosome(waypoints=waypoints))
        return pop

    def standard_mutate(self, chromosome: Chromosome) -> Chromosome:
        mutated_waypoints = chromosome.waypoints.copy()
        for i in range(1, len(mutated_waypoints) - 1):
            if np.random.rand() < 0.1:
                mutation_vector = (np.random.rand(3) - 0.5) * 50.0
                new_position = mutated_waypoints[i] + mutation_vector
                mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
        return Chromosome(waypoints=mutated_waypoints)

    def predictive_mutate(self, chromosome: Chromosome, lstm_predictions: dict, obstacles: List[dict]) -> Chromosome:
        mutated_waypoints = chromosome.waypoints.copy()
        predicted_obstacles = lstm_predictions.get('predicted_obstacles', [])
        confidence_scores = lstm_predictions.get('confidence_scores', [])
        for i, waypoint in enumerate(mutated_waypoints[1:-1], 1):
            max_threat, threat_direction = 0.0, np.zeros(3)
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
                if self._is_position_safe(new_position, obstacles):
                    mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
        return Chromosome(waypoints=mutated_waypoints)

    def predictive_mutate_enhanced(self, chromosome: Chromosome, lstm_predictions: dict, obstacles: List[dict]) -> Chromosome:
        mutated_waypoints = chromosome.waypoints.copy()
        predicted_obstacles = lstm_predictions.get('predicted_obstacles', [])
        confidence_scores = lstm_predictions.get('confidence_scores', [])
        for i, waypoint in enumerate(mutated_waypoints[1:-1], 1):
            max_threat, threat_direction, confidence_at_max = 0.0, np.zeros(3), 0.0
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
                        confidence_at_max = confidence
            if max_threat > 0.3:
                threat_level_at_max = max_threat / (confidence_at_max + 1e-6)
                avoidance_strength = (50.0 + 150.0 * confidence_at_max) * threat_level_at_max
                maneuver_choice = np.random.rand()
                if maneuver_choice < 0.5:
                    avoidance_vector = threat_direction * avoidance_strength
                else:
                    side_step_dir = np.cross(threat_direction, np.array([0, 0, 1.0]))
                    side_step_dir /= (np.linalg.norm(side_step_dir) + 1e-6)
                    avoidance_vector = side_step_dir * avoidance_strength * (1 if maneuver_choice < 0.75 else -1)
                new_position = waypoint + avoidance_vector
                if self._is_position_safe(new_position, obstacles):
                    mutated_waypoints[i] = np.clip(new_position, self.bounds_min, self.bounds_max)
        return Chromosome(waypoints=mutated_waypoints)
    
    def get_next_position(self) -> np.ndarray:
        """Returns the next waypoint and updates internal state (receding horizon)."""
        if len(self.population[0].waypoints) > 1:
            best_next_pos = self.population[0].waypoints[1]
            self.advance_to_next_step(best_next_pos)
            return best_next_pos
        return self.current_pos
    
    def evolve_generation(self, uav_id: int, obstacles: List[dict], 
                          lstm_predictions: dict, all_uav_positions: List[np.ndarray],
                          network_performance: Dict):
        """
        Runs one generation of the GA.
        """
        for chrom in self.population:
            chrom.fitness, chrom.costs = evaluate_fitness(
                chrom.waypoints, 
                obstacles, 
                lstm_predictions,
                all_uav_positions, 
                uav_id,
                network_performance
            )
        
        self.population.sort(key=lambda x: x.fitness)
        elite_size = int(self.population_size * self.elite_ratio)
        new_population = self.population[:elite_size]
        
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(self.population[:elite_size], 2, replace=False)
            crossover_point = np.random.randint(1, self.num_waypoints - 1)
            child = Chromosome(np.vstack((parent1.waypoints[:crossover_point], parent2.waypoints[crossover_point:])))
            
            if np.random.rand() < self.mutation_rate:
                if self.mode == 'lstm_enhanced':
                    child = self.predictive_mutate_enhanced(child, lstm_predictions, obstacles)
                elif self.mode == 'lstm':
                    child = self.predictive_mutate(child, lstm_predictions, obstacles)
                else: 
                    child = self.standard_mutate(child)
            
            new_population.append(child)
        self.population = new_population
        
        # Update last fitness and costs (for interface consistency with RRTAgent)
        self.last_fitness = self.population[0].fitness
        self.last_costs = self.population[0].costs
