import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

def analyze_corrected_results(results_dir='.', output_dir='results'):
    """
    Loads all CSVs from results_dir, aggregates data, and generates
    CORRECTED plots and tables into output_dir.
    """
    
    all_data = []
    
    # --- 1. Create Output Directory ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # --- 2. Load and Process Data ---
    print(f"Loading data from '{results_dir}' directory...")
    try:
        filenames = os.listdir(results_dir)
    except FileNotFoundError:
        print(f"Error: The directory '{results_dir}' was not found.")
        return

    csv_files = [f for f in filenames if f.endswith('.csv')]
    if not csv_files:
        print("Error: No CSV files found.")
        return

    # Filter for the specific run files
    run_files = [f for f in csv_files if re.match(r'(.+)_run_(\d+)\.csv', f)]
    if not run_files:
        print("Error: No valid simulation run CSV files found.")
        print("Please ensure files like 'ga_lstm_run_1.csv' are in the directory.")
        return

    print(f"Found {len(run_files)} data files to process.")

    for f in run_files:
        match = re.match(r'(.+)_run_(\d+)\.csv', f)
        if not match:
            continue
            
        algorithm, run_num = match.groups()
        
        # Remap names for clarity and better plot legends
        if algorithm == 'ga_lstm_enhanced':
            algo_name = 'GA-LSTM-ENHANCED'
        elif algorithm == 'ga_lstm':
            algo_name = 'GA-LSTM'
        elif algorithm == 'standard_ga':
            algo_name = 'STANDARD-GA'
        elif algorithm == 'rrt_star':
            algo_name = 'RRT-STAR'
        else:
            algo_name = algorithm.replace('_', '-').upper()
            
        filepath = os.path.join(results_dir, f)
        try:
            df = pd.read_csv(filepath)
            df['algorithm'] = algo_name
            df['run'] = int(run_num)
            all_data.append(df)
        except Exception as e:
            print(f"Could not load {f}: {e}")

    if not all_data:
        print("Error: No valid data could be loaded.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully loaded {len(all_data)} experiment runs.")
    
    # --- 3. Aggregate Data (Time Series) ---
    
    # *** KEY ADDITION ***
    # Calculate the 'pdr_penalty' as it is in the fitness function
    # (pdr_penalty = 1.0 - pdr)
    full_df['pdr_penalty'] = 1.0 - full_df['pdr']
    
    agg_df = full_df.groupby(['algorithm', 'time']).agg(
        # Primary fitness
        mean_fitness=('best_fitness', 'mean'),
        sem_fitness=('best_fitness', 'sem'),
        
        # Cost components
        mean_distance_cost=('distance_cost', 'mean'),
        mean_turning_cost=('turning_cost', 'mean'),
        sem_turning_cost=('turning_cost', 'sem'),
        mean_collision_cost=('collision_cost', 'mean'),
        sem_collision_cost=('collision_cost', 'sem'),
        mean_prediction_penalty=('prediction_penalty', 'mean'),
        mean_cohesion_cost=('cohesion_cost', 'mean'),
        sem_cohesion_cost=('cohesion_cost', 'sem'),
        
        # *** NEWLY TRACKED METRICS FOR PLOTS ***
        mean_pdr=('pdr', 'mean'),
        sem_pdr=('pdr', 'sem'),
        mean_pdr_penalty=('pdr_penalty', 'mean')
    ).reset_index()

    # Get sorted list of algorithms for consistent plot ordering
    algorithms = sorted(agg_df['algorithm'].unique())
    
    plt.style.use('seaborn-v0_8-paper') 

    # --- 4. Generate Line Plots ---

    # Plot 0: *** THE MOST IMPORTANT PLOT (NEW) *** (Network Performance)
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_df = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_df['time'], algo_df['mean_pdr'] * 100, label=algo, lw=2.5)
        plt.fill_between(
            algo_df['time'],
            (algo_df['mean_pdr'] - algo_df['sem_pdr']) * 100,
            (algo_df['mean_pdr'] + algo_df['sem_pdr']) * 100,
            alpha=0.15
        )
    plt.title('Network Performance (Packet Delivery Ratio)', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Packet Delivery Ratio (PDR) (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-5, 105) # Set Y-axis from 0 to 100
    plot_path = os.path.join(output_dir, '0_pdr_performance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{plot_path}'")

    # Plot 1: Convergence Speed (Overall Fitness)
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_df = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_df['time'], algo_df['mean_fitness'], label=algo, lw=2)
        plt.fill_between(
            algo_df['time'],
            algo_df['mean_fitness'] - algo_df['sem_fitness'],
            algo_df['mean_fitness'] + algo_df['sem_fitness'],
            alpha=0.15
        )
    plt.title('Algorithm Convergence (Total Weighted Cost)', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Best Fitness (Total Cost)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, '1_convergence_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{plot_path}'")

    # Plot 2: Cohesion Cost (The "Why" for the PDR plot)
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_df = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_df['time'], algo_df['mean_cohesion_cost'], label=algo, lw=2)
        plt.fill_between(
            algo_df['time'],
            algo_df['mean_cohesion_cost'] - algo_df['sem_cohesion_cost'],
            algo_df['mean_cohesion_cost'] + algo_df['sem_cohesion_cost'],
            alpha=0.15
        )
    plt.title('Network Cohesion Cost', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Cohesion Cost', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, '2_cohesion_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{plot_path}'")

    # Plot 3: Energy Efficiency
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_df = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_df['time'], algo_df['mean_turning_cost'], label=algo, lw=2)
        plt.fill_between(
            algo_df['time'],
            algo_df['mean_turning_cost'] - algo_df['sem_turning_cost'],
            algo_df['mean_turning_cost'] + algo_df['sem_turning_cost'],
            alpha=0.15
        )
    plt.title('Energy Efficiency (Turning Cost)', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Turning Cost', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, '3_energy_efficiency_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{plot_path}'")

    # Plot 4: Safety
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_df = agg_df[agg_df['algorithm'] == algo]
        plt.plot(algo_df['time'], algo_df['mean_collision_cost'], label=algo, lw=2)
        plt.fill_between(
            algo_df['time'],
            algo_df['mean_collision_cost'] - algo_df['sem_collision_cost'],
            algo_df['mean_collision_cost'] + algo_df['sem_collision_cost'],
            alpha=0.15
        )
    plt.title('Path Safety (Collision Cost)', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Collision Cost', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, '4_safety_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved '{plot_path}'")

    # --- 5. Generate Summary Table (for console and CSV) ---
    final_stats_df = full_df[full_df['time'] == full_df['time'].max()].groupby('algorithm').agg(
        Final_Fitness_Mean=('best_fitness', 'mean'),
        Final_Fitness_StdErr=('best_fitness', 'sem'),
        Final_Cohesion_Cost_Mean=('cohesion_cost', 'mean'),
        Final_PDR_Mean=('pdr', 'mean'),
        Final_Turning_Cost_Mean=('turning_cost', 'mean'),
        Final_Collision_Cost_Mean=('collision_cost', 'mean')
    ).round(3)
    
    print("\n" + "="*70)
    print("           --- FINAL PERFORMANCE SUMMARY TABLE ---")
    print("="*70)
    print(final_stats_df)
    print("="*70)
    
    table_path = os.path.join(output_dir, '5_final_summary_table.csv')
    final_stats_df.to_csv(table_path)
    print(f"Saved '{table_path}'")

    # --- 6. Generate *** CORRECTED *** Stacked Bar Chart ---
    print("\nGenerating CORRECTED final cost composition bar chart...")
    
    final_time = full_df['time'].max()
    final_df = full_df[full_df['time'] == final_time]
    
    # Get the mean of the *raw, unweighted* costs
    component_df = final_df.groupby('algorithm')[[
        'distance_cost', 
        'turning_cost', 
        'collision_cost', 
        'prediction_penalty',
        'cohesion_cost',
        'pdr_penalty' # *** MUST INCLUDE THIS ***
    ]].mean()

    # *** THESE ARE THE CORRECT WEIGHTS from genetic_algorithm.py ***
    weights = {
        'distance_cost': 0.10,
        'turning_cost': 0.10,
        'collision_cost': 0.10,
        'prediction_penalty': 0.05,
        'cohesion_cost': 0.25,
        'pdr_penalty': 0.40  # This is (1.0 - mean_pdr)
    }
    
    weighted_component_df = component_df.copy()
    for col in weighted_component_df.columns:
        if col in weights:
            # Apply the weight to the mean raw cost
            weighted_component_df[col] = weighted_component_df[col] * weights[col]

    # Re-order columns for a logical stack (PDR penalty at the top)
    weighted_component_df = weighted_component_df[[
        'distance_cost',
        'turning_cost',
        'collision_cost',
        'prediction_penalty',
        'cohesion_cost',
        'pdr_penalty'
    ]]

    # *** NEW LEGEND based on CORRECT weights ***
    weighted_component_df.columns = [
        'Distance (10%)', 
        'Turning (10%)', 
        'Collision (10%)', 
        'Prediction (5%)',
        'Cohesion (25%)',
        'PDR Penalty (40%)' # *** THE MOST IMPORTANT PART ***
    ]
    
    # Sort the dataframe by the total fitness (sum of weighted components)
    # This makes the bar chart easier to read
    weighted_component_df['Total_Fitness'] = weighted_component_df.sum(axis=1)
    weighted_component_df = weighted_component_df.sort_values('Total_Fitness')
    
    plt.figure(figsize=(10, 7))
    ax = weighted_component_df.drop('Total_Fitness', axis=1).plot(
        kind='bar', 
        stacked=True, 
        figsize=(11, 7), # Wider figure to accommodate legend
        colormap='viridis' 
    )
    
    plt.title('Final Fitness Score Composition (Corrected Weights)', fontsize=16, fontweight='bold')
    plt.ylabel('Weighted Fitness Cost (Total = Final Fitness)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=0) 
    plt.legend(title='Cost Component', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    plot_path = os.path.join(output_dir, '6_cost_composition_barchart_CORRECTED.png')
    plt.savefig(plot_path, dpi=300) # No bbox_inches='tight' to respect tight_layout
    print(f"Saved '{plot_path}'")
    
    table_path = os.path.join(output_dir, '6_cost_composition_table_CORRECTED.csv')
    weighted_component_df.to_csv(table_path)
    print(f"Saved '{table_path}'")
    
    print("\nAll corrected analysis is complete. Check the 'results' directory.")

if __name__ == "__main__":
    analyze_corrected_results()