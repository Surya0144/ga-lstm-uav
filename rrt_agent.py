# rrt_agent.py
# --- FINAL VERSION ---

import numpy as np
from typing import List, Dict

from genetic_algorithm import Chromosome, evaluate_fitness 
from rrt_star_planner import RRTStar

class RRTAgent:
    """A 'wrapper' class that mimics the GeneticAlgorithm class interface."""
    def __init__(self, ga_params: dict, mode: str = 'rrt_star', uav_id: int = 0):
        self.mode = mode
        self.num_waypoints = ga_params.get('num_waypoints', 8)
        self.bounds_min = np.array([0, 0, 50])
        self.bounds_max = np.array([1000, 1000, 500])
        
        self.uav_id = uav_id
        offset_vector = np.array([float(self.uav_id), 0.0, 0.0])
        
        self.start_pos = np.array([50.0, 50.0, 100.0]) + offset_vector
        self.goal_pos = np.array([950.0, 950.0, 100.0]) + offset_vector
        
        # --- Attributes for state tracking ---
        self.current_pos = self.start_pos
        self.best_path = np.linspace(self.start_pos, self.goal_pos, self.num_waypoints)
        self.last_fitness = 0.0
        self.last_costs = {}
        self.population = [Chromosome(self.best_path)] # For interface compatibility
        
        print(f"[RRT*Agent] Initialized in '{self.mode}' mode for UAV {self.uav_id} (Pos: {self.start_pos}).")

    def get_next_position(self) -> np.ndarray:
        """Returns the next waypoint for the NS2 bridge and updates internal state."""
        if self.best_path is not None and len(self.best_path) > 1:
            self.current_pos = self.best_path[1]
            self.best_path = np.vstack([self.best_path[1:], self.best_path[-1]])
            return self.current_pos
        return self.current_pos

    def _interpolate_path(self, path: np.ndarray) -> np.ndarray:
        """Converts the RRT* path (variable length) to a fixed-length path."""
        distances = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        total_distance = distances[-1]
        
        if total_distance < 1e-6:
             return np.array([path[0]] * self.num_waypoints)

        interp_distances = np.linspace(0, total_distance, self.num_waypoints)
        interp_x = np.interp(interp_distances, distances, path[:, 0])
        interp_y = np.interp(interp_distances, distances, path[:, 1])
        interp_z = np.interp(interp_distances, distances, path[:, 2])
        return np.vstack((interp_x, interp_y, interp_z)).T

    def advance_to_next_step(self, best_next_position: np.ndarray):
        """
        Implements a receding horizon for RRT* agent.
        Updates the internal state to reflect movement to the next position.
        This method exists for interface compatibility with GeneticAlgorithm.
        """
        self.current_pos = best_next_position
        # The best_path will be recomputed in the next evolve_generation call
        # starting from the new current position
    
    def evolve_generation(self, uav_id: int, obstacles: List[dict], 
                          lstm_predictions: dict, all_uav_positions: List[np.ndarray],
                          network_performance: Dict):
        """
        Runs one iteration of RRT* planning.
        """
        rrt = RRTStar(
            start=self.current_pos, # Always plan from the current position
            goal=self.goal_pos,
            obstacles=obstacles,
            bounds=[0, 1000, 0, 1000, 50, 500], 
            step_size=50.0,
            search_radius=75.0,
            n_iterations=500 
        )
        new_path_points = rrt.run()
        
        if new_path_points is not None and len(new_path_points) > 1:
            waypoints = self._interpolate_path(new_path_points)
            self.best_path = waypoints
        else:
            waypoints = self.best_path
        
        # Evaluate the chosen path using the shared, network-aware fitness function
        fitness, costs = evaluate_fitness(
            waypoints, 
            obstacles, 
            lstm_predictions,
            all_uav_positions,
            uav_id,
            network_performance
        ) 
        
        # Store the results for logging by the main planner
        self.last_fitness = fitness
        self.last_costs = costs
