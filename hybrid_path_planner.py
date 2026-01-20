# hybrid_path_planner.py
# --- FINAL VERSION ---

import time
import numpy as np
import csv
from typing import Dict

from genetic_algorithm import GeneticAlgorithm, Chromosome
from lstm_predictor import LSTMPredictor
from ns2_python_bridge import NS2PythonBridge
from rrt_agent import RRTAgent

class HybridGALSTMPathPlanner:
    def __init__(self, num_uavs: int, ga_params: Dict, lstm_params: Dict, 
                 simulation_params: Dict, ga_mode: str, output_filename: str):
        print(f"[Planner] Initializing Hybrid Planner (Mode: {ga_mode})...")
        
        self.num_uavs = num_uavs
        self.simulation_duration = simulation_params.get('simulation_duration', 100.0)
        self.position_update_interval = simulation_params.get('position_update_interval', 1.0)
        
        self.is_running = False
        self.current_time = 0.0
        
        if ga_mode == 'rrt_star':
            self.planners = [RRTAgent(ga_params, mode=ga_mode, uav_id=i) for i in range(self.num_uavs)]
        else:
            self.planners = [GeneticAlgorithm(ga_params, mode=ga_mode) for _ in range(self.num_uavs)]
            print("[Planner] Synchronizing initial populations with offsets...")
            base_population = self.planners[0].population
            
            for i in range(1, self.num_uavs):
                new_population = []
                offset_vector = np.array([float(i), 0.0, 0.0]) 
                for c in base_population:
                    new_waypoints = c.waypoints.copy() + offset_vector
                    new_waypoints = np.clip(new_waypoints, 
                                            self.planners[i].bounds_min, 
                                            self.planners[i].bounds_max)
                    new_population.append(Chromosome(new_waypoints))
                self.planners[i].population = new_population

        self.lstm_predictor = LSTMPredictor(**lstm_params)
        self.ns2_bridge = NS2PythonBridge(self.simulation_duration, self.num_uavs)
        
        self.obstacles = []
        self.network_performance = {'packet_delivery_ratio': 1.0} 

        self.output_file = open(output_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.output_file)
        
        self.csv_writer.writerow(['time', 'best_fitness', 'distance_cost', 'turning_cost', 
                                  'collision_cost', 'prediction_penalty', 'cohesion_cost',
                                  'pdr', 'replanning_trigger'])
                                  
        print(f"[Planner] Logging results to {output_filename}")

    def _update_obstacle_positions(self):
        if not self.obstacles:
            self.obstacles = [{'id': 1, 'x': 100, 'y': 100, 'z': 100, 'radius': 20, 
                               'velocity': np.array([25.0, 15.0, 0.0])}]
        else:
            for obs in self.obstacles:
                obs['x'] += obs['velocity'][0] * self.position_update_interval
                obs['y'] += obs['velocity'][1] * self.position_update_interval
                if not (10 < obs['x'] < 990): obs['velocity'][0] *= -1
                if not (10 < obs['y'] < 990): obs['velocity'][1] *= -1

    def _update_uav_positions_ns2(self, new_positions: Dict):
        for uav_id, pos in new_positions.items():
            self.ns2_bridge.update_uav_position(uav_id, pos, self.current_time)

    def _collect_performance_metrics(self):
        self.network_performance = self.ns2_bridge.get_network_metrics()

    def _log_data(self):
        if self.planners[0].mode == 'rrt_star':
            best_fit = self.planners[0].last_fitness
            costs = self.planners[0].last_costs
        else:
            best_chrom = self.planners[0].population[0]
            best_fit = best_chrom.fitness
            costs = best_chrom.costs
        
        pdr = self.network_performance.get('packet_delivery_ratio', 0)
        replanning_trigger = 1 if pdr < 0.8 and self.current_time > 0 else 0

        self.csv_writer.writerow([
            self.current_time,
            best_fit,
            costs.get('distance', 0),
            costs.get('turning', 0), 
            costs.get('collision', 0),
            costs.get('penalty', 0),
            costs.get('cohesion', 0),
            pdr,
            replanning_trigger
        ])

    def _main_planning_loop(self):
        print(f"[Planner] Starting main loop...")
        
        current_positions = {}
        for i in range(self.num_uavs):
            if self.planners[i].mode == 'rrt_star':
                current_positions[i] = self.planners[i].current_pos
            else:
                current_positions[i] = self.planners[i].population[0].waypoints[0]

        while self.is_running and self.current_time < self.simulation_duration:
            loop_start_time = time.time()
            
            # 1. MEASURE
            self._collect_performance_metrics() 
            
            # 2. UPDATE
            self._update_obstacle_positions()
            lstm_predictions = self.lstm_predictor.get_all_predictions()
            all_uav_positions = list(current_positions.values())
            
            # 3. PLAN
            for uav_id in range(self.num_uavs):
                self.planners[uav_id].evolve_generation(
                    uav_id, 
                    self.obstacles, 
                    lstm_predictions,
                    all_uav_positions,
                    self.network_performance
                )
            
            # 4. LOG
            self._log_data() 
            
            # 5. ACT & ADVANCE STATE
            new_positions_to_send = {}
            for uav_id in range(self.num_uavs):
                next_pos = self.planners[uav_id].get_next_position()
                new_positions_to_send[uav_id] = next_pos
                current_positions[uav_id] = next_pos
            
            self._update_uav_positions_ns2(new_positions_to_send)
            
            if int(self.current_time) % 10 == 0:
                 pdr = self.network_performance.get('packet_delivery_ratio', 0)
                 if self.planners[0].mode == 'rrt_star':
                     best_fit = self.planners[0].last_fitness
                     cohesion = self.planners[0].last_costs.get('cohesion', 0)
                 else:
                     best_fit = self.planners[0].population[0].fitness
                     cohesion = self.planners[0].population[0].costs.get('cohesion', 0)
                 
                 print(f"[Planner] Time {self.current_time:3.0f}s | Best Fitness: {best_fit:5.2f} | Cohesion Cost: {cohesion:5.2f} | Net PDR: {pdr*100:3.0f}%")

            self.current_time += self.position_update_interval
            
            elapsed = time.time() - loop_start_time
            sleep_time = self.position_update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        print("[Planner] Main loop finished.")

    def run_simulation(self):
        try:
            self.ns2_bridge._start_ns2_simulation()
            self.is_running = True
            self._main_planning_loop() 
        except Exception as e:
            print(f"[Planner] Error during simulation: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("[Planner] Shutting down...")
        self.is_running = False
        self.ns2_bridge.stop_ns2_simulation()
        if self.output_file and not self.output_file.closed:
            self.output_file.close()
            print(f"[Planner] Simulation stopped. Log file '{self.output_file.name}' closed.")

if __name__ == "__main__":
    print("--- Starting Single-Run Test for Hybrid Planner ---")
    TEST_MODE = 'lstm_enhanced' 
    ga_params = {
        'population_size': 100, 'num_waypoints': 8, 'mutation_rate': 0.1, 'elite_ratio': 0.1
    }
    lstm_params = {
        'hidden_size': 64, 'num_layers': 2, 'prediction_horizon': 5
    }
    simulation_params = {
        'simulation_duration': 100.0, 'position_update_interval': 1.0
    }
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    planner = HybridGALSTMPathPlanner(
        num_uavs=5,
        ga_params=ga_params,
        lstm_params=lstm_params,
        simulation_params=simulation_params,
        ga_mode=TEST_MODE,
        output_filename=f'results/single_test_run_{TEST_MODE}.csv'
    )
    
    planner.run_simulation()
    print("--- Single-Run Test Finished ---")
