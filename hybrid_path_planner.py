# hybrid_path_planner.py
# Main integration module (sources 184, 208-220, 273-310)

import time
import numpy as np
from typing import Dict

# Import from other project files
from genetic_algorithm import GeneticAlgorithm
from lstm_predictor import LSTMPredictor
from ns2_python_bridge import NS2PythonBridge

class HybridGALSTMPathPlanner:
    def __init__(self, num_uavs: int, ga_params: Dict, lstm_params: Dict, simulation_params: Dict):
        print("[Planner] Initializing Hybrid GA-LSTM Path Planner...")
        
        self.num_uavs = num_uavs
        self.simulation_duration = simulation_params.get('simulation_duration', 100.0)
        self.position_update_interval = simulation_params.get('position_update_interval', 1.0)
        
        self.is_running = False
        self.current_time = 0.0
        
        # Initialize core components
        print(f"[Planner] Loading {num_uavs} GA components...")
        self.genetic_algorithms = [GeneticAlgorithm(ga_params) for _ in range(self.num_uavs)]
        
        print("[Planner] Loading LSTM predictor...")
        self.lstm_predictor = LSTMPredictor(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            prediction_horizon=lstm_params.get('prediction_horizon', 5)
        )
        
        print("[Planner] Initializing NS2 Bridge...")
        self.ns2_bridge = NS2PythonBridge(
            simulation_time=self.simulation_duration,
            num_uavs=self.num_uavs
        )
        
        self.obstacles = [] # List of dynamic obstacles
        self.network_performance = {}
        print("[Planner] Initialization complete.")

    def _update_obstacle_positions(self):
        """Placeholder: Update positions of dynamic obstacles (source 211)."""
        if not self.obstacles:
            # Create a dummy obstacle
            self.obstacles = [{'id': 1, 'x': 100, 'y': 100, 'z': 100, 'radius': 20, 
                               'velocity': np.array([20.0, 10.0, 0.0])}]
        else:
            for obs in self.obstacles:
                obs['x'] += obs['velocity'][0] * self.position_update_interval
                obs['y'] += obs['velocity'][1] * self.position_update_interval
                # Simple boundary bouncing
                if not (10 < obs['x'] < 990): obs['velocity'][0] *= -1
                if not (10 < obs['y'] < 990): obs['velocity'][1] *= -1

    def _update_uav_positions_ns2(self):
        """Send new UAV paths to NS2 via the bridge (source 216)."""
        for uav_id in range(self.num_uavs):
            # Get the best path from the GA
            best_chrom = self.genetic_algorithms[uav_id].population[0]
            # Get the *next* waypoint to move to
            next_position = best_chrom.waypoints[1] 
            self.ns2_bridge.update_uav_position(uav_id, next_position, self.current_time)

    def _collect_performance_metrics(self):
        """Get network metrics from the NS2 bridge (source 217)."""
        self.network_performance = self.ns2_bridge.get_network_metrics()

    def _check_replanning_triggers(self):
        """Placeholder: Check if replanning is needed (source 219)."""
        pdr = self.network_performance.get('packet_delivery_ratio', 1.0)
        if pdr < 0.8:
            print(f"[Planner] Warning: Low PDR ({pdr*100:.1f}%)! Replanning may be needed.")
        pass

    def _main_planning_loop(self):
        """Main planning loop with GA-LSTM integration (sources 208-220)"""
        print(f"[Planner] Starting main loop. Duration: {self.simulation_duration}s, Interval: {self.position_update_interval}s")
        while self.is_running and self.current_time < self.simulation_duration:
            loop_start_time = time.time()
            
            # Update dynamic obstacle positions (source 211)
            self._update_obstacle_positions()
            
            # Get LSTM predictions for all obstacles (source 213)
            lstm_predictions = self.lstm_predictor.get_all_predictions()
            
            # Evolve paths for each UAV (source 214-215)
            for uav_id in range(self.num_uavs):
                self.genetic_algorithms[uav_id].evolve_generation(
                    self.obstacles, lstm_predictions)
            
            # Update UAV positions in NS2 (source 216)
            self._update_uav_positions_ns2()
            
            # Collect network performance metrics (source 217)
            self._collect_performance_metrics()
            
            # Check for replanning triggers (source 219)
            self._check_replanning_triggers()
            
            if int(self.current_time) % 10 == 0:
                 pdr = self.network_performance.get('packet_delivery_ratio', 0)
                 best_fit = self.genetic_algorithms[0].population[0].fitness
                 print(f"[Planner] Time {self.current_time:3.0f}s | Best Fitness: {best_fit:5.2f} | Net PDR: {pdr*100:3.0f}%")

            self.current_time += self.position_update_interval
            
            # Ensure loop runs in (close to) real-time
            elapsed = time.time() - loop_start_time
            sleep_time = self.position_update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        print("[Planner] Main loop finished.")

    def run_simulation(self):
        """Starts the main planning loop and NS2."""
        try:
            self.ns2_bridge._start_ns2_simulation()
            self.is_running = True
            self._main_planning_loop()
        except Exception as e:
            print(f"[Planner] Error during simulation: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shuts down the planner and NS2 bridge."""
        print("[Planner] Shutting down...")
        self.is_running = False
        self.ns2_bridge.stop_ns2_simulation()
        print("[Planner] Simulation stopped.")

# Main execution block (sources 271, 273-291)
if __name__ == "__main__":
    print("--- Starting GA-LSTM UAV Path Planning Simulation (Research Config) ---")
    
    # Research Configuration (from sources 275-291)
    research_ga_params = {
        'population_size': 100,
        'num_waypoints': 8,
        'mutation_rate': 0.1,
        'elite_ratio': 0.1
    }
    
    research_lstm_params = {
        'sequence_length': 10,
        'prediction_horizon': 5,
        'hidden_size': 64,
        'num_layers': 2 # Added from doc default
    }
    
    research_sim_params = {
        'simulation_duration': 100.0,
        'position_update_interval': 1.0
    }
    
    planner = HybridGALSTMPathPlanner(
        num_uavs=5,
        ga_params=research_ga_params,
        lstm_params=research_lstm_params,
        simulation_params=research_sim_params
    )
    
    planner.run_simulation()
    
    print("--- Simulation Finished ---")