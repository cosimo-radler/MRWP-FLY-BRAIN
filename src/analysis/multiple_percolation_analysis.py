#!/usr/bin/env python3
"""
Multiple Configuration Models Percolation Analysis

This script performs both edge (bond) and node percolation analysis on:
1. Multiple scaled configuration models (1500 nodes)
2. Multiple unscaled configuration models (original node count)

Results are aggregated across model instances to provide robust statistics.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from functools import partial
import glob

# Set the style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Constants
CONFIG_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
SCALED_MODELS_DIR = os.path.join(CONFIG_MODELS_DIR, "multiple_scaled")
UNSCALED_MODELS_DIR = os.path.join(CONFIG_MODELS_DIR, "multiple_unscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MULTIPLE_RESULTS_DIR = os.path.join(RESULTS_DIR, "multiple_percolation")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
MULTIPLE_FIGURES_DIR = os.path.join(FIGURES_DIR, "multiple_percolation")

# Ensure directories exist
os.makedirs(MULTIPLE_RESULTS_DIR, exist_ok=True)
os.makedirs(MULTIPLE_FIGURES_DIR, exist_ok=True)

# Network types
NETWORK_TYPES = ['eb', 'fb', 'mb_kc']
NETWORK_NAMES = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body KC'
}

# Model types
MODEL_TYPES = ['scaled', 'unscaled']
MODEL_DIRS = {
    'scaled': SCALED_MODELS_DIR,
    'unscaled': UNSCALED_MODELS_DIR
}
MODEL_LABELS = {
    'scaled': 'Scaled (1500 nodes)',
    'unscaled': 'Unscaled (original size)'
}

# Percolation parameters
NUM_SIMULATIONS = 20  # Simulations per probability value per model
NUM_PROB_STEPS = 50  # Number of probability values to test
PERCOLATION_TYPES = ['edge', 'node']  # Types of percolation to perform

# Colors for visualization
COLORS = {
    'scaled': 'red',
    'unscaled': 'green'
}

def load_network_models(network_type, model_type):
    """Load all instances of a specific network and model type.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'scaled' or 'unscaled'
        
    Returns:
        List of NetworkX graphs
    """
    model_dir = MODEL_DIRS[model_type]
    pattern = os.path.join(model_dir, f"{network_type}_{model_type}_config_model_*.gexf")
    file_paths = sorted(glob.glob(pattern))
    
    if not file_paths:
        print(f"No models found matching pattern: {pattern}")
        return []
    
    print(f"Loading {len(file_paths)} {model_type} {network_type} models...")
    
    graphs = []
    for file_path in file_paths:
        try:
            G = nx.read_gexf(file_path)
            graphs.append(G)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return graphs

def run_edge_percolation(G, removal_probability, num_simulations=NUM_SIMULATIONS):
    """Run edge percolation simulation for a specific removal probability.
    
    Args:
        G: NetworkX graph
        removal_probability: Probability of removing an edge
        num_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with simulation results
    """
    # Create undirected graph for percolation
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # Run multiple simulations
    lcc_sizes = []
    
    for _ in range(num_simulations):
        # Create a copy for this simulation
        G_sim = G_undirected.copy()
        
        # Randomly remove edges
        edges_to_remove = []
        for u, v in G_sim.edges():
            if np.random.random() < removal_probability:
                edges_to_remove.append((u, v))
        
        G_sim.remove_edges_from(edges_to_remove)
        
        # Calculate largest connected component size
        if G_sim.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc) / original_size
        else:
            lcc_size = 0
            
        lcc_sizes.append(lcc_size)
    
    # Calculate statistics
    mean_lcc = np.mean(lcc_sizes)
    std_lcc = np.std(lcc_sizes)
    
    return {
        'removal_probability': removal_probability,
        'mean_lcc_size': mean_lcc,
        'std_lcc_size': std_lcc
    }

def run_node_percolation(G, removal_probability, num_simulations=NUM_SIMULATIONS):
    """Run node percolation simulation for a specific removal probability.
    
    Args:
        G: NetworkX graph
        removal_probability: Probability of removing a node
        num_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary with simulation results
    """
    # Create undirected graph for percolation
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # Run multiple simulations
    lcc_sizes = []
    
    for _ in range(num_simulations):
        # Create a copy for this simulation
        G_sim = G_undirected.copy()
        
        # Randomly remove nodes
        nodes_to_remove = []
        for node in G_sim.nodes():
            if np.random.random() < removal_probability:
                nodes_to_remove.append(node)
        
        G_sim.remove_nodes_from(nodes_to_remove)
        
        # Calculate largest connected component size
        if G_sim.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc) / original_size
        else:
            lcc_size = 0
            
        lcc_sizes.append(lcc_size)
    
    # Calculate statistics
    mean_lcc = np.mean(lcc_sizes)
    std_lcc = np.std(lcc_sizes)
    
    return {
        'removal_probability': removal_probability,
        'mean_lcc_size': mean_lcc,
        'std_lcc_size': std_lcc
    }

def process_edge_probability(args):
    """Process a single edge removal probability (for parallel processing)."""
    G, removal_probability, num_simulations = args
    return run_edge_percolation(G, removal_probability, num_simulations)

def process_node_probability(args):
    """Process a single node removal probability (for parallel processing)."""
    G, removal_probability, num_simulations = args
    return run_node_percolation(G, removal_probability, num_simulations)

def run_percolation_analysis(G, network_type, model_type, model_index, percolation_type, 
                            num_prob_steps=NUM_PROB_STEPS, num_simulations=NUM_SIMULATIONS):
    """Run percolation analysis on a single model instance.
    
    Args:
        G: NetworkX graph
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'scaled' or 'unscaled'
        model_index: Index/identifier for this model instance
        percolation_type: 'edge' or 'node'
        num_prob_steps: Number of probability points to evaluate
        num_simulations: Number of simulations per probability
        
    Returns:
        DataFrame with results
    """
    print(f"\nRunning {percolation_type} percolation for {network_type} {model_type} model {model_index}...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal probabilities
    removal_probabilities = np.linspace(0, 1, num_prob_steps)
    
    # Use parallel processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Create argument list
    args_list = [(G, p, num_simulations) for p in removal_probabilities]
    
    # Run simulations in parallel
    if percolation_type == 'edge':
        results = list(tqdm(
            pool.imap(process_edge_probability, args_list),
            total=len(args_list),
            desc=f"Simulating edge removal ({model_type} model {model_index})"
        ))
    else:  # node percolation
        results = list(tqdm(
            pool.imap(process_node_probability, args_list),
            total=len(args_list),
            desc=f"Simulating node removal ({model_type} model {model_index})"
        ))
    
    pool.close()
    pool.join()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    filename = f"{network_type}_{model_type}_config_{percolation_type}_percolation_{model_index:02d}.csv"
    df.to_csv(os.path.join(MULTIPLE_RESULTS_DIR, filename), index=False)
    
    return df

def run_all_percolation_analyses():
    """Run percolation analyses for all network types, model types, and percolation types."""
    results = {}
    
    for network_type in NETWORK_TYPES:
        results[network_type] = {}
        
        for model_type in MODEL_TYPES:
            print(f"\nProcessing {network_type} {model_type} models...")
            
            # Load all models of this type
            models = load_network_models(network_type, model_type)
            
            if not models:
                print(f"No {model_type} models found for {network_type}")
                continue
                
            results[network_type][model_type] = {
                'edge': [],
                'node': []
            }
            
            # Process each model instance
            for i, G in enumerate(models):
                model_index = i + 1
                
                # Run edge percolation
                edge_df = run_percolation_analysis(
                    G, network_type, model_type, model_index, 'edge'
                )
                results[network_type][model_type]['edge'].append(edge_df)
                
                # Run node percolation
                node_df = run_percolation_analysis(
                    G, network_type, model_type, model_index, 'node'
                )
                results[network_type][model_type]['node'].append(node_df)
    
    return results

def aggregate_results(network_type, model_type, percolation_type):
    """Aggregate results across multiple model instances.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'scaled' or 'unscaled'
        percolation_type: 'edge' or 'node'
        
    Returns:
        DataFrame with aggregated results
    """
    pattern = os.path.join(MULTIPLE_RESULTS_DIR, 
                          f"{network_type}_{model_type}_config_{percolation_type}_percolation_*.csv")
    file_paths = sorted(glob.glob(pattern))
    
    if not file_paths:
        print(f"No results found matching pattern: {pattern}")
        return None
    
    # Load all result files
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dfs:
        return None
    
    # Aggregate results
    # For each removal probability, calculate mean and std across all model instances
    agg_data = []
    prob_values = dfs[0]['removal_probability'].values
    
    for prob in prob_values:
        # Collect LCC sizes across models for this probability
        all_lcc_sizes = []
        
        for df in dfs:
            prob_row = df[df['removal_probability'] == prob].iloc[0]
            all_lcc_sizes.append(prob_row['mean_lcc_size'])
        
        agg_data.append({
            'removal_probability': prob,
            'mean_lcc_size': np.mean(all_lcc_sizes),
            'std_lcc_size': np.std(all_lcc_sizes),
            'ci_lower': np.percentile(all_lcc_sizes, 5),  # 5th percentile
            'ci_upper': np.percentile(all_lcc_sizes, 95)  # 95th percentile
        })
    
    return pd.DataFrame(agg_data)

def extract_percolation_threshold(df):
    """Extract percolation threshold from results using 5% LCC criterion.
    
    Args:
        df: DataFrame with percolation results
        
    Returns:
        Threshold value
    """
    if df is None or 'removal_probability' not in df.columns or 'mean_lcc_size' not in df.columns:
        return np.nan
    
    # Find where LCC drops below 5%
    for i in range(len(df) - 1):
        if df['mean_lcc_size'].iloc[i] >= 0.05 and df['mean_lcc_size'].iloc[i+1] < 0.05:
            # Linear interpolation
            x1 = df['removal_probability'].iloc[i]
            x2 = df['removal_probability'].iloc[i+1]
            y1 = df['mean_lcc_size'].iloc[i]
            y2 = df['mean_lcc_size'].iloc[i+1]
            
            threshold = x1 + (x2 - x1) * (0.05 - y1) / (y2 - y1)
            return round(threshold, 3)
    
    return np.nan

def plot_percolation_comparison(network_type, save=True):
    """Create comparison plots for a specific network type.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        save: Whether to save the plot to file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set titles
    fig.suptitle(f"{NETWORK_NAMES[network_type]} Network: Multiple Configuration Models Percolation Analysis", 
                 fontsize=16, fontweight='bold')
    axes[0].set_title('Edge Percolation')
    axes[1].set_title('Node Percolation')
    
    # Set axis labels
    for ax in axes:
        ax.set_xlabel('Removal Probability')
        ax.set_ylabel('Largest Connected Component Size')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    # Add threshold line
    for ax in axes:
        ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.0)
        ax.text(0.05, 0.07, 'Threshold (5%)', fontsize=8, color='gray')
    
    # Plot edge percolation results
    for model_type in MODEL_TYPES:
        df = aggregate_results(network_type, model_type, 'edge')
        
        if df is not None:
            # Calculate threshold
            threshold = extract_percolation_threshold(df)
            label = f"{MODEL_LABELS[model_type]} (threshold = {threshold:.3f})"
            
            # Plot mean with confidence interval
            axes[0].plot(
                df['removal_probability'], 
                df['mean_lcc_size'],
                color=COLORS[model_type],
                linewidth=2,
                label=label
            )
            
            # Add confidence interval
            axes[0].fill_between(
                df['removal_probability'],
                df['ci_lower'],
                df['ci_upper'],
                color=COLORS[model_type],
                alpha=0.2
            )
    
    # Plot node percolation results
    for model_type in MODEL_TYPES:
        df = aggregate_results(network_type, model_type, 'node')
        
        if df is not None:
            # Calculate threshold
            threshold = extract_percolation_threshold(df)
            label = f"{MODEL_LABELS[model_type]} (threshold = {threshold:.3f})"
            
            # Plot mean with confidence interval
            axes[1].plot(
                df['removal_probability'], 
                df['mean_lcc_size'],
                color=COLORS[model_type],
                linewidth=2,
                label=label
            )
            
            # Add confidence interval
            axes[1].fill_between(
                df['removal_probability'],
                df['ci_lower'],
                df['ci_upper'],
                color=COLORS[model_type],
                alpha=0.2
            )
    
    # Add legends
    for ax in axes:
        ax.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save:
        output_path = os.path.join(MULTIPLE_FIGURES_DIR, f"{network_type}_multiple_percolation_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def create_comprehensive_visualization(save=True):
    """Create a comprehensive visualization with all networks and percolation types."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Set overall title
    fig.suptitle('Multiple Configuration Models: Edge and Node Percolation Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add row titles
    for i, network_type in enumerate(NETWORK_TYPES):
        fig.text(0.02, 0.78 - i*0.30, f"{NETWORK_NAMES[network_type]}", 
                fontsize=14, fontweight='bold', rotation=90, va='center')
    
    # Add column titles
    fig.text(0.25, 0.91, 'Edge Percolation', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.75, 0.91, 'Node Percolation', fontsize=14, fontweight='bold', ha='center')
    
    # Add informational text
    info_text = """
    Methodology:
    - 10 independent configuration model instances per network type
    - 20 Monte Carlo simulations per removal probability per model
    - 50 probability points from 0 to 1
    - Shaded areas show 90% confidence intervals across model instances
    - Threshold defined as removal probability where LCC drops below 5%
    """
    fig.text(0.5, 0.02, info_text, fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', alpha=0.5))
    
    # Track thresholds for summary table
    thresholds = {network_type: {percolation_type: {} for percolation_type in PERCOLATION_TYPES} 
                 for network_type in NETWORK_TYPES}
    
    # For each network type and percolation type
    for i, network_type in enumerate(NETWORK_TYPES):
        # Set axis labels
        for j, percolation_type in enumerate(PERCOLATION_TYPES):
            ax = axes[i, j]
            ax.set_xlabel('Removal Probability')
            ax.set_ylabel('Largest Connected Component Size')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            # Add threshold line
            ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.0)
            ax.text(0.05, 0.07, 'Threshold (5%)', fontsize=8, color='gray')
            
            # Plot results for each model type
            for model_type in MODEL_TYPES:
                df = aggregate_results(network_type, model_type, percolation_type)
                
                if df is not None:
                    # Calculate threshold
                    threshold = extract_percolation_threshold(df)
                    thresholds[network_type][percolation_type][model_type] = threshold
                    label = f"{MODEL_LABELS[model_type]} (threshold = {threshold:.3f})"
                    
                    # Plot mean with confidence interval
                    ax.plot(
                        df['removal_probability'], 
                        df['mean_lcc_size'],
                        color=COLORS[model_type],
                        linewidth=2,
                        label=label
                    )
                    
                    # Add confidence interval
                    ax.fill_between(
                        df['removal_probability'],
                        df['ci_lower'],
                        df['ci_upper'],
                        color=COLORS[model_type],
                        alpha=0.2
                    )
            
            # Add legend
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
    
    if save:
        output_path = os.path.join(MULTIPLE_FIGURES_DIR, "comprehensive_multiple_percolation.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to {output_path}")
    
    plt.show()
    
    # Return thresholds data for further analysis
    return thresholds

def create_threshold_summary_table(thresholds, save=True):
    """Create a summary table of percolation thresholds.
    
    Args:
        thresholds: Dictionary of threshold values from create_comprehensive_visualization
        save: Whether to save the table to file
    """
    # Create DataFrame for table
    data = []
    
    for network_type in NETWORK_TYPES:
        for model_type in MODEL_TYPES:
            edge_threshold = thresholds[network_type]['edge'].get(model_type, np.nan)
            node_threshold = thresholds[network_type]['node'].get(model_type, np.nan)
            
            data.append({
                'Network': NETWORK_NAMES[network_type],
                'Model Type': MODEL_LABELS[model_type],
                'Edge Percolation Threshold': edge_threshold,
                'Node Percolation Threshold': node_threshold
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    if save:
        output_path = os.path.join(MULTIPLE_RESULTS_DIR, "percolation_thresholds_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"Threshold summary saved to {output_path}")
    
    return df

def main():
    """Main function."""
    print("Starting multiple configuration models percolation analysis...")
    
    # Check if the models exist
    model_files_exist = False
    for network_type in NETWORK_TYPES:
        for model_type in MODEL_TYPES:
            pattern = os.path.join(MODEL_DIRS[model_type], f"{network_type}_{model_type}_config_model_*.gexf")
            if glob.glob(pattern):
                model_files_exist = True
                break
    
    if not model_files_exist:
        print("No model files found. Please run multiple_config_models.py first.")
        return
    
    # Check if results already exist
    results_exist = True
    for network_type in NETWORK_TYPES:
        for model_type in MODEL_TYPES:
            for percolation_type in PERCOLATION_TYPES:
                pattern = os.path.join(MULTIPLE_RESULTS_DIR, 
                                      f"{network_type}_{model_type}_config_{percolation_type}_percolation_*.csv")
                if not glob.glob(pattern):
                    results_exist = False
                    break
    
    if not results_exist:
        print("Running percolation analyses for all networks and models...")
        run_all_percolation_analyses()
    else:
        print("Using existing percolation results...")
    
    # Create visualizations
    print("\nCreating visualizations...")
    thresholds = create_comprehensive_visualization()
    
    # Create summary table
    create_threshold_summary_table(thresholds)
    
    print("Done!")

if __name__ == "__main__":
    main() 