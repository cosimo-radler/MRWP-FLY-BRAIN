#!/usr/bin/env python3
"""
Targeted Attack Analysis for Unscaled Configuration Models

This script performs targeted edge attacks (degree centrality and betweenness centrality) on 
unscaled configuration models of the ellipsoid-body (EB), fan-shaped-body (FB), 
and mushroom-body (MB) Kenyon-cell networks, using the same methodology as for the original networks.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing
from functools import partial

# Constants
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/unscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of Monte Carlo simulations per fraction value
NUM_SIMULATIONS = 20

# Number of fraction steps to evaluate
NUM_FRACTION_STEPS = 50

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{network_type}_network.gexf"))

def load_unscaled_config_model(network_type):
    """Load unscaled configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf"))

def run_targeted_attack_simulation(G, removal_fraction, attack_strategy='degree', num_simulations=NUM_SIMULATIONS):
    """Run targeted attack simulation for a specific edge removal fraction.
    
    Args:
        G: NetworkX graph
        removal_fraction: Fraction of edges to remove
        attack_strategy: Strategy for selecting edges to remove ('degree', 'betweenness', 'random')
        num_simulations: Number of Monte Carlo simulations to run
        
    Returns:
        Dictionary with simulation results
    """
    # Create an undirected graph for analysis
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    total_edges = G_undirected.number_of_edges()
    
    # Calculate number of edges to remove
    num_edges_to_remove = int(removal_fraction * total_edges)
    
    # List to store largest connected component sizes for each simulation
    lcc_sizes = []
    
    for _ in range(num_simulations):
        # Create a copy of the original graph for this simulation
        G_sim = G_undirected.copy()
        
        if attack_strategy == 'random':
            # Random attack (equivalent to bond percolation)
            edges = list(G_sim.edges())
            if edges:
                edges_to_remove = np.random.choice(
                    len(edges), 
                    size=min(num_edges_to_remove, len(edges)), 
                    replace=False
                )
                G_sim.remove_edges_from([edges[i] for i in edges_to_remove])
                
        elif attack_strategy == 'degree':
            # Targeted attack based on node degree
            edges_removed = 0
            
            # Calculate initial node degrees
            node_degrees = dict(G_sim.degree())
            
            # Sort nodes by degree (highest first)
            sorted_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)
            
            # Remove edges connected to highest degree nodes first
            for node in sorted_nodes:
                if edges_removed >= num_edges_to_remove or not G_sim.edges():
                    break
                    
                # Get edges connected to this node
                edges_to_remove = list(G_sim.edges(node))
                
                # Update how many edges we've removed
                if edges_to_remove:
                    # Remove some or all edges
                    num_to_remove = min(len(edges_to_remove), num_edges_to_remove - edges_removed)
                    edges_to_actually_remove = edges_to_remove[:num_to_remove]
                    G_sim.remove_edges_from(edges_to_actually_remove)
                    edges_removed += len(edges_to_actually_remove)
                    
                    # Recalculate node degrees if not done
                    if edges_removed < num_edges_to_remove:
                        node_degrees = dict(G_sim.degree())
                        sorted_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)
        
        elif attack_strategy == 'betweenness':
            # Targeted attack based on edge betweenness centrality
            try:
                edge_betweenness = nx.edge_betweenness_centrality(G_sim)
                edges_to_remove = sorted(edge_betweenness.keys(), 
                                         key=lambda x: edge_betweenness[x], 
                                         reverse=True)[:num_edges_to_remove]
                G_sim.remove_edges_from(edges_to_remove)
            except Exception as e:
                print(f"Warning: Edge betweenness calculation failed: {e}")
                # Fall back to degree-based attack
                return run_targeted_attack_simulation(G, removal_fraction, 'degree', num_simulations)
        
        # Calculate largest connected component size
        if len(G_sim) > 0:  # Check if graph is not empty
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc) / original_size  # Normalized size
        else:
            lcc_size = 0
        
        lcc_sizes.append(lcc_size)
    
    # Calculate statistics across simulations
    mean_lcc = np.mean(lcc_sizes)
    std_lcc = np.std(lcc_sizes)
    
    return {
        'removal_fraction': removal_fraction,
        'mean_lcc_size': mean_lcc,
        'std_lcc_size': std_lcc,
        'lcc_sizes': lcc_sizes,
        'attack_strategy': attack_strategy
    }

def process_fraction(args):
    """Process a single removal fraction (for parallel processing)."""
    G, removal_fraction, attack_strategy, num_simulations = args
    return run_targeted_attack_simulation(G, removal_fraction, attack_strategy, num_simulations)

def run_targeted_attack_analysis(G, network_name, attack_strategy='degree', 
                               num_fraction_steps=NUM_FRACTION_STEPS, 
                               num_simulations=NUM_SIMULATIONS):
    """Run targeted attack analysis across a range of removal fractions.
    
    Args:
        G: NetworkX graph
        network_name: Name for the results
        attack_strategy: Strategy for selecting edges to remove
        num_fraction_steps: Number of fraction steps to evaluate
        num_simulations: Number of simulations per fraction value
        
    Returns:
        DataFrame with results for each fraction
    """
    print(f"\nRunning targeted attack analysis for {network_name} unscaled configuration model...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Attack strategy: {attack_strategy}")
    
    # Generate removal fractions
    removal_fractions = np.linspace(0, 1, num_fraction_steps)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Prepare arguments for each worker
    args_list = [(G, f, attack_strategy, num_simulations) for f in removal_fractions]
    
    # Run simulations in parallel
    results = list(tqdm(
        pool.imap(process_fraction, args_list),
        total=len(args_list),
        desc=f"Simulating {attack_strategy} attack"
    ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, 
                          f"{network_name.lower().replace(' ', '_')}_unscaled_config_{attack_strategy}_attack_results.csv"), 
              index=False)
    
    return df

def estimate_critical_fraction(df):
    """Estimate the critical fraction from targeted attack results.
    
    Args:
        df: DataFrame with targeted attack results
        
    Returns:
        Dictionary with critical fraction and robustness index
    """
    x = df['removal_fraction'].values
    y = df['mean_lcc_size'].values
    
    # Simple interpolation to find where LCC size approaches zero
    threshold = None
    for i in range(len(y) - 1):
        if y[i] > 0.05 and y[i+1] < 0.05:  # Threshold at 5% connectivity
            # Linear interpolation
            threshold = x[i] + (x[i+1] - x[i]) * (0.05 - y[i]) / (y[i+1] - y[i])
            break
    
    if threshold is None and y[-1] > 0.05:
        threshold = 1.0  # If never drops below threshold
    elif threshold is None:
        threshold = 0.0  # If already below threshold at start
    
    # Calculate robustness index (area under LCC curve)
    robustness_index = np.trapz(y, x)
    
    return {
        'critical_fraction': threshold,
        'robustness_index': robustness_index
    }

def plot_attack_comparison(original_df, unscaled_df, network_type, attack_strategy, original_params, unscaled_params):
    """Plot targeted attack comparison between original and unscaled configuration model.
    
    Args:
        original_df: DataFrame with original network attack results
        unscaled_df: DataFrame with unscaled configuration model attack results
        network_type: Type of network for naming
        attack_strategy: Strategy used for the attack
        original_params: Dictionary with original network critical parameters
        unscaled_params: Dictionary with unscaled configuration model critical parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original network results
    plt.errorbar(
        original_df['removal_fraction'], 
        original_df['mean_lcc_size'],
        yerr=original_df['std_lcc_size'],
        fmt='o-', 
        color='blue',
        alpha=0.7,
        label=f"Original Network (qc ≈ {original_params['critical_fraction']:.3f})"
    )
    
    # Plot unscaled configuration model results
    plt.errorbar(
        unscaled_df['removal_fraction'], 
        unscaled_df['mean_lcc_size'],
        yerr=unscaled_df['std_lcc_size'],
        fmt='s-', 
        color='green',
        alpha=0.7,
        label=f"Unscaled Config Model (qc ≈ {unscaled_params['critical_fraction']:.3f})"
    )
    
    # Add vertical lines at critical thresholds
    plt.axvline(x=original_params['critical_fraction'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=unscaled_params['critical_fraction'], color='green', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title(f"{network_type.upper()} Network vs. Unscaled Config Model: {attack_strategy.title()} Attack")
    plt.xlabel("Edge Removal Fraction")
    plt.ylabel("Largest Connected Component Size (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with robustness index
    plt.text(
        0.05, 0.15,
        f"Robustness Index (Original): {original_params['robustness_index']:.3f}\n"
        f"Robustness Index (Unscaled Config): {unscaled_params['robustness_index']:.3f}",
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_unscaled_config_{attack_strategy}_attack_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def process_original_network_attack(network_type, attack_strategy):
    """Run targeted attack analysis on the original network.
    
    Args:
        network_type: Type of network ('eb', 'fb', 'mb_kc')
        attack_strategy: Strategy for selecting edges to remove
        
    Returns:
        Tuple of (DataFrame with results, parameters dictionary)
    """
    # Get full network name for file naming
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # Check if results already exist
    results_file = os.path.join(RESULTS_DIR, f"{full_name}_original_{attack_strategy}_attack_results.csv")
    
    if os.path.exists(results_file):
        print(f"Loading existing original network {attack_strategy} attack results...")
        df = pd.read_csv(results_file)
        params = estimate_critical_fraction(df)
        return df, params
    
    # Otherwise, run the analysis
    print(f"Running {attack_strategy} attack on original {network_type} network...")
    G = load_original_network(network_type)
    df = run_targeted_attack_analysis(G, f"{full_name}_original", attack_strategy)
    params = estimate_critical_fraction(df)
    return df, params

def process_network(network_type):
    """
    Process a single network with different attack strategies.
    
    Args:
        network_type: Type of network ('eb', 'fb', 'mb_kc')
    """
    print(f"\nProcessing {network_type} unscaled configuration model...")
    
    # Load unscaled configuration model
    G = load_unscaled_config_model(network_type)
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get full network name for file naming
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # Perform degree attack on unscaled model
    print("Performing degree centrality attack...")
    unscaled_degree_results = run_targeted_attack_analysis(G, full_name, 'degree')
    unscaled_degree_params = estimate_critical_fraction(unscaled_degree_results)
    
    # Perform degree attack on original network and compare
    original_degree_results, original_degree_params = process_original_network_attack(network_type, 'degree')
    plot_attack_comparison(original_degree_results, unscaled_degree_results, network_type, 'degree', 
                         original_degree_params, unscaled_degree_params)
    
    # Perform betweenness attack on unscaled model
    print("Performing betweenness centrality attack...")
    unscaled_betweenness_results = run_targeted_attack_analysis(G, full_name, 'betweenness')
    unscaled_betweenness_params = estimate_critical_fraction(unscaled_betweenness_results)
    
    # Perform betweenness attack on original network and compare
    original_betweenness_results, original_betweenness_params = process_original_network_attack(network_type, 'betweenness')
    plot_attack_comparison(original_betweenness_results, unscaled_betweenness_results, network_type, 'betweenness', 
                         original_betweenness_params, unscaled_betweenness_params)
    
    # Perform random attack (equivalent to bond percolation)
    print("Performing random edge removal...")
    unscaled_random_results = run_targeted_attack_analysis(G, full_name, 'random')
    unscaled_random_params = estimate_critical_fraction(unscaled_random_results)
    
    # Perform random attack on original network and compare
    original_random_results, original_random_params = process_original_network_attack(network_type, 'random')
    plot_attack_comparison(original_random_results, unscaled_random_results, network_type, 'random', 
                         original_random_params, unscaled_random_params)
    
    # Save parameters for unscaled model
    unscaled_attack_params = {
        'degree': unscaled_degree_params,
        'betweenness': unscaled_betweenness_params,
        'random': unscaled_random_params
    }
    
    # Save parameters to JSON
    with open(os.path.join(RESULTS_DIR, f"{full_name}_unscaled_config_attack_parameters.json"), 'w') as f:
        json.dump(unscaled_attack_params, f, indent=4)
        
    return {
        'original': {
            'degree': {
                'results': original_degree_results,
                'params': original_degree_params
            },
            'betweenness': {
                'results': original_betweenness_results,
                'params': original_betweenness_params
            },
            'random': {
                'results': original_random_results,
                'params': original_random_params
            }
        },
        'unscaled_config': {
            'degree': {
                'results': unscaled_degree_results,
                'params': unscaled_degree_params
            },
            'betweenness': {
                'results': unscaled_betweenness_results,
                'params': unscaled_betweenness_params
            },
            'random': {
                'results': unscaled_random_results,
                'params': unscaled_random_params
            }
        }
    }

def create_comparison_summary(results):
    """Create a summary DataFrame of all attack results.
    
    Args:
        results: Dictionary with all results
        
    Returns:
        DataFrame with comparison summary
    """
    summary_data = []
    
    for network_type in results:
        for attack_type in ['degree', 'betweenness', 'random']:
            original_params = results[network_type]['original'][attack_type]['params']
            unscaled_params = results[network_type]['unscaled_config'][attack_type]['params']
            
            summary_data.append({
                'network_type': network_type,
                'model_type': 'original',
                'attack_type': attack_type,
                'critical_fraction': original_params['critical_fraction'],
                'robustness_index': original_params['robustness_index']
            })
            
            summary_data.append({
                'network_type': network_type,
                'model_type': 'unscaled_config',
                'attack_type': attack_type,
                'critical_fraction': unscaled_params['critical_fraction'],
                'robustness_index': unscaled_params['robustness_index']
            })
    
    return pd.DataFrame(summary_data)

def main():
    """Main function to run the script."""
    print("Starting targeted attack analysis for unscaled configuration models...")
    
    # Process each network type
    results = {}
    for network_type in ['eb', 'fb', 'mb_kc']:
        results[network_type] = process_network(network_type)
    
    # Create summary DataFrame
    summary_df = create_comparison_summary(results)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "unscaled_config_attack_summary.csv"), index=False)
    
    print("\nTargeted attack analysis completed for all unscaled configuration models!")
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'unscaled_config_attack_summary.csv')}")

if __name__ == "__main__":
    main() 