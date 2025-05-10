#!/usr/bin/env python3
"""
Upscaled Configuration Model Targeted Attack Analysis for Drosophila Circuit Robustness

This script performs targeted attack analysis on the upscaled configuration models (3500 nodes)
of the ellipsoid-body (EB), fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks.
Targeted attacks remove edges connected to high-centrality nodes or based on edge betweenness.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import warnings
import json

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/upscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Attack parameters
FRACTION_STEPS = 21  # Number of fraction steps from 0 to 1
ATTACK_TYPES = ['degree', 'betweenness']  # Types of targeted attacks
NUM_SIMULATIONS = 1  # Number of simulations per attack (1 for deterministic attacks)

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    data_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    return nx.read_gexf(data_path)

def load_upscaled_config_model(network_type):
    """Load upscaled configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(UPSCALED_CONFIG_MODEL_DIR, f"{network_type}_upscaled_config_model.gexf"))

def run_targeted_attack_simulation(G, removal_fraction, attack_type, num_simulations=NUM_SIMULATIONS):
    """Run targeted attack simulation for a specific edge removal fraction.
    
    Args:
        G: NetworkX graph
        removal_fraction: Fraction of edges to remove
        attack_type: Type of targeted attack ('degree' or 'betweenness')
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
        
        if attack_type == 'degree':
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
        
        elif attack_type == 'betweenness':
            # Targeted attack based on edge betweenness centrality
            try:
                print("Calculating edge betweenness centrality (this may take a while)...")
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
        'attack_type': attack_type
    }

def process_fraction(args):
    """Process a single removal fraction (for parallel processing)."""
    G, removal_fraction, attack_type, num_simulations = args
    return run_targeted_attack_simulation(G, removal_fraction, attack_type, num_simulations)

def run_attack_analysis(G, model_type, network_name, attack_type):
    """Run a targeted attack analysis.
    
    Args:
        G: NetworkX graph
        model_type: 'original' or 'upscaled_config_model'
        network_name: Name for the results (e.g., 'eb', 'fb', 'mb_kc')
        attack_type: Type of targeted attack ('degree' or 'betweenness')
        
    Returns:
        DataFrame with attack results and critical parameters
    """
    full_name = f"{network_name}_{model_type}_{attack_type}_attack"
    print(f"\nRunning {attack_type}-based targeted attack for {network_name} {model_type} network...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal fractions
    removal_fractions = np.linspace(0, 1, FRACTION_STEPS)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Prepare arguments for each worker
    args_list = [(G, f, attack_type, NUM_SIMULATIONS) for f in removal_fractions]
    
    # Run simulations in parallel
    results = list(tqdm(
        pool.imap(process_fraction, args_list),
        total=len(args_list),
        desc=f"Simulating {attack_type} attack"
    ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, f"{full_name}_results.csv"), index=False)
    
    # Estimate critical threshold
    params = estimate_critical_threshold(df)
    
    # Save parameters
    with open(os.path.join(RESULTS_DIR, f"{full_name}_parameters.json"), 'w') as f:
        json.dump(params, f)
    
    return df, params

def estimate_critical_threshold(df):
    """Estimate critical threshold from attack data.
    
    Args:
        df: DataFrame with attack results
        
    Returns:
        Dictionary with estimated parameters
    """
    x = df['removal_fraction'].values
    y = df['mean_lcc_size'].values
    
    # Simple threshold estimate (where LCC size drops below 0.05)
    threshold_simple = 0
    for i, size in enumerate(y):
        if size < 0.05:
            if i > 0:
                threshold_simple = x[i-1] + (x[i] - x[i-1]) * (0.05 - y[i-1]) / (y[i] - y[i-1])
            else:
                threshold_simple = x[i]
            break
    
    # If we didn't cross 0.05, use the last fraction
    if threshold_simple == 0 and len(y) > 0:
        threshold_simple = x[-1]
    
    return {
        'threshold_simple': threshold_simple
    }

def plot_attack_comparison(original_df, upscaled_df, network_type, attack_type, original_params, upscaled_params):
    """Plot attack comparison between original and upscaled configuration model.
    
    Args:
        original_df: DataFrame with original network attack results
        upscaled_df: DataFrame with upscaled configuration model attack results
        network_type: Type of network for naming
        attack_type: Type of targeted attack ('degree' or 'betweenness')
        original_params: Dictionary with original network critical parameters
        upscaled_params: Dictionary with upscaled configuration model critical parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original network results
    plt.plot(
        original_df['removal_fraction'], 
        original_df['mean_lcc_size'],
        'o-', 
        color='blue',
        alpha=0.7,
        label=f"Original Network (qc ≈ {original_params['threshold_simple']:.3f})"
    )
    
    # Plot upscaled configuration model results
    plt.plot(
        upscaled_df['removal_fraction'], 
        upscaled_df['mean_lcc_size'],
        's-', 
        color='red',
        alpha=0.7,
        label=f"Upscaled Config Model (3500 nodes) (qc ≈ {upscaled_params['threshold_simple']:.3f})"
    )
    
    # Add vertical lines at critical thresholds
    plt.axvline(x=original_params['threshold_simple'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=upscaled_params['threshold_simple'], color='red', linestyle='--', alpha=0.5)
    
    # Add title and labels
    attack_name = "Degree-Based" if attack_type == "degree" else "Betweenness-Based"
    plt.title(f"{network_type.upper()} Network vs. Upscaled Configuration Model: {attack_name} Attack")
    plt.xlabel("Fraction of Edges Removed")
    plt.ylabel("Largest Connected Component Size (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(FIGURES_DIR, f"{network_type}_vs_upscaled_{attack_type}_attack.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved comparison plot to {output_path}")

def create_multi_comparison_plot(original_df, config_df, upscaled_df, network_type, attack_type,
                               original_params, config_params, upscaled_params):
    """Create a plot comparing original, config, and upscaled models.
    
    Args:
        original_df: DataFrame with original network results
        config_df: DataFrame with config model results (1500 nodes)
        upscaled_df: DataFrame with upscaled config model results (3500 nodes)
        network_type: Type of network for naming
        attack_type: Type of targeted attack ('degree' or 'betweenness')
        original_params: Dictionary with original network parameters
        config_params: Dictionary with config model parameters
        upscaled_params: Dictionary with upscaled model parameters
    """
    plt.figure(figsize=(12, 7))
    
    # Plot original network results
    if original_df is not None:
        plt.plot(
            original_df['removal_fraction'], 
            original_df['mean_lcc_size'],
            'o-', 
            color='blue',
            alpha=0.7,
            label=f"Original Network (qc ≈ {original_params['threshold_simple']:.3f})"
        )
        plt.axvline(x=original_params['threshold_simple'], color='blue', linestyle='--', alpha=0.3)
    
    # Plot configuration model results
    if config_df is not None:
        plt.plot(
            config_df['removal_fraction'], 
            config_df['mean_lcc_size'],
            's-', 
            color='green',
            alpha=0.7,
            label=f"Config Model (1500 nodes) (qc ≈ {config_params['threshold_simple']:.3f})"
        )
        plt.axvline(x=config_params['threshold_simple'], color='green', linestyle='--', alpha=0.3)
    
    # Plot upscaled configuration model results
    if upscaled_df is not None:
        plt.plot(
            upscaled_df['removal_fraction'], 
            upscaled_df['mean_lcc_size'],
            'd-', 
            color='red',
            alpha=0.7,
            label=f"Upscaled Config Model (3500 nodes) (qc ≈ {upscaled_params['threshold_simple']:.3f})"
        )
        plt.axvline(x=upscaled_params['threshold_simple'], color='red', linestyle='--', alpha=0.3)
    
    # Add title and labels
    attack_name = "Degree-Based" if attack_type == "degree" else "Betweenness-Based"
    plt.title(f"{network_type.upper()}: {attack_name} Targeted Attack Comparison")
    plt.xlabel("Fraction of Edges Removed")
    plt.ylabel("Largest Connected Component Size (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    output_path = os.path.join(FIGURES_DIR, f"{network_type}_{attack_type}_attack_model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved multi-comparison plot to {output_path}")

def create_comparison_summary(results):
    """Create a summary DataFrame of the attack results for all networks and models.
    
    Args:
        results: Dictionary with results for each network and model
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for network_type, network_results in results.items():
        for attack_type, attack_results in network_results.items():
            for model_type, params in attack_results.items():
                if params is not None:
                    summary_data.append({
                        'network_type': network_type,
                        'attack_type': attack_type,
                        'model_type': model_type,
                        'critical_threshold': params['threshold_simple']
                    })
    
    return pd.DataFrame(summary_data)

def plot_summary_comparison(summary_df):
    """Create a bar plot comparing critical thresholds across models and networks.
    
    Args:
        summary_df: DataFrame with summary statistics
    """
    plt.figure(figsize=(15, 8))
    
    # Create a grouped bar plot
    bar_width = 0.25
    index = np.arange(3)  # Three network types
    
    # Get model types and network types
    model_types = sorted(summary_df['model_type'].unique())
    attack_types = sorted(summary_df['attack_type'].unique())
    network_types = ['eb', 'fb', 'mb_kc']
    network_labels = {'eb': 'Ellipsoid Body', 'fb': 'Fan-shaped Body', 'mb_kc': 'Mushroom Body KC'}
    
    # Colors for different model types
    colors = {
        'original': 'blue',
        'config_model': 'green',
        'upscaled_config_model': 'red'
    }
    
    # Create a figure for each attack type
    for attack_type in attack_types:
        plt.figure(figsize=(15, 8))
        
        for i, model_type in enumerate(model_types):
            # Get data for this model and attack type
            data = []
            for network_type in network_types:
                subset = summary_df[(summary_df['network_type'] == network_type) & 
                                    (summary_df['model_type'] == model_type) & 
                                    (summary_df['attack_type'] == attack_type)]
                if not subset.empty:
                    data.append(subset['critical_threshold'].values[0])
                else:
                    data.append(0)  # No data
            
            # Plot bars
            plt.bar(index + i*bar_width, data, bar_width, 
                    color=colors.get(model_type, 'gray'), 
                    label=model_type.replace('_', ' ').title())
        
        # Add labels and legend
        plt.xlabel('Network Type')
        plt.ylabel('Critical Threshold')
        plt.title(f"{attack_type.title()} Attack: Critical Threshold Comparison")
        plt.xticks(index + bar_width, [network_labels[nt] for nt in network_types])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)
        
        # Save the figure
        output_path = os.path.join(FIGURES_DIR, f"upscaled_{attack_type.lower()}_attack_threshold_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved {attack_type} attack summary to {output_path}")

def process_network_attack(network_type, attack_type):
    """Process a single network for a specific attack type.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        attack_type: 'degree' or 'betweenness'
        
    Returns:
        Tuple of DataFrames (original, config, upscaled) and parameter dictionaries
    """
    network_names = {
        'eb': 'Ellipsoid Body',
        'fb': 'Fan-shaped Body',
        'mb_kc': 'Mushroom Body Kenyon Cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    print(f"\nProcessing {full_name} network with {attack_type} attack...")
    
    # Load networks
    G_original = load_original_network(network_type)
    G_upscaled = load_upscaled_config_model(network_type)
    
    # Run analysis for original network
    orig_df, orig_params = run_attack_analysis(G_original, 'original', network_type, attack_type)
    
    # Run analysis for upscaled configuration model
    upscaled_df, upscaled_params = run_attack_analysis(G_upscaled, 'upscaled_config_model', network_type, attack_type)
    
    # Plot comparison
    plot_attack_comparison(orig_df, upscaled_df, network_type, attack_type, orig_params, upscaled_params)
    
    # Try to load the standard configuration model results
    config_path = os.path.join(RESULTS_DIR, f"{full_name.lower().replace(' ', '_')}_config_{attack_type}_attack_results.csv")
    config_params_path = os.path.join(RESULTS_DIR, f"{full_name.lower().replace(' ', '_')}_config_{attack_type}_attack_parameters.json")
    
    if os.path.exists(config_path) and os.path.exists(config_params_path):
        print(f"Loading existing configuration model results from {config_path}")
        config_df = pd.read_csv(config_path)
        with open(config_params_path, 'r') as f:
            config_params = json.load(f)
        
        # Create multi-comparison plot
        create_multi_comparison_plot(orig_df, config_df, upscaled_df, network_type, attack_type,
                                   orig_params, config_params, upscaled_params)
    else:
        print(f"No configuration model results found at {config_path}")
        create_multi_comparison_plot(orig_df, None, upscaled_df, network_type, attack_type,
                                   orig_params, {'threshold_simple': 0}, upscaled_params)
    
    return orig_df, upscaled_df, orig_params, upscaled_params

def main():
    """Main function to run the script."""
    results = {}
    
    for network_type in ['eb', 'fb', 'mb_kc']:
        results[network_type] = {}
        
        for attack_type in ATTACK_TYPES:
            print(f"\n{'='*80}\nProcessing {network_type} with {attack_type} attack\n{'='*80}")
            
            orig_df, upscaled_df, orig_params, upscaled_params = process_network_attack(network_type, attack_type)
            
            results[network_type][attack_type] = {
                'original': orig_params,
                'upscaled_config_model': upscaled_params
            }
    
    # Create and save summary
    summary_df = create_comparison_summary(results)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "upscaled_attack_comparison_summary.csv"), index=False)
    plot_summary_comparison(summary_df)
    
    print(f"\nSaved summary to {os.path.join(RESULTS_DIR, 'upscaled_attack_comparison_summary.csv')}")
    print("\nUpscaled configuration model targeted attack analysis complete!")

if __name__ == "__main__":
    main() 