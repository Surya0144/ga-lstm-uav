# run_experiments.py
# --- UPDATED to include the RRT* baseline ---

import numpy as np
import os
from hybrid_path_planner import HybridGALSTMPathPlanner

def run_single_experiment(config: dict):
    """Runs one simulation with a given configuration."""
    print("\n" + "="*50)
    print(f"  Starting Experiment: {config['name']} (Mode: {config['mode']})")
    print("="*50)

    np.random.seed(config['seed'])
    ga_params = {'population_size': 100, 'num_waypoints': 8, 'mutation_rate': 0.1, 'elite_ratio': 0.1}
    lstm_params = {'hidden_size': 64, 'num_layers': 2, 'prediction_horizon': 5}
    simulation_params = {'simulation_duration': 100.0, 'position_update_interval': 1.0}

    planner = HybridGALSTMPathPlanner(
        num_uavs=5,
        ga_params=ga_params,
        lstm_params=lstm_params,
        simulation_params=simulation_params,
        ga_mode=config['mode'],
        output_filename=config['output_file']
    )
    planner.run_simulation()
    print(f"--- Experiment {config['name']} Finished ---")

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    NUMBER_OF_RUNS = 3 # For a real paper, use 30+
    experiments = []
    
    for i in range(NUMBER_OF_RUNS):
        seed = 42 + i
        
        # 1. Our superior method
        experiments.append({
            'name': f'GA-LSTM-Enhanced_run{i+1}',
            'mode': 'lstm_enhanced',
            'seed': seed,
            'output_file': f'results/ga_lstm_enhanced_run_{i+1}.csv'
        })
        
        # 2. The original novel method
        experiments.append({
            'name': f'GA-LSTM_run{i+1}',
            'mode': 'lstm',
            'seed': seed,
            'output_file': f'results/ga_lstm_run_{i+1}.csv'
        })

        # 3. The standard GA baseline
        experiments.append({
            'name': f'Standard-GA_run{i+1}',
            'mode': 'standard',
            'seed': seed,
            'output_file': f'results/standard_ga_run_{i+1}.csv'
        })
        
        # 4. The new published baseline (RRT*)
        experiments.append({
            'name': f'RRT-Star_run{i+1}',
            'mode': 'rrt_star',
            'seed': seed,
            'output_file': f'results/rrt_star_run_{i+1}.csv'
        })

    for exp_config in experiments:
        run_single_experiment(exp_config)

    print("\n" + "="*50)
    print("  All experiments are complete.  ")
    print("  Check the 'results' directory for the output CSV files.  ")
    print("="*50)
