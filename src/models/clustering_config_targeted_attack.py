#!/usr/bin/env python3
"""
Clustering-Preserved Configuration Models Targeted Attack Analysis

This script performs targeted attack analysis on the configuration models that
preserve both degree distribution and clustering coefficient of the original
brain region networks. The analysis involves removing edges based on centrality
measures (degree and betweenness) and measuring the size of the largest connected component.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import pickle
from tqdm import tqdm
import multiprocessing
from functools import partial

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLUSTERING_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "clustering")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Attack parameters
REMOVAL_FRACTIONS = np.arange(0, 1.01, 0.01)

# Number of CPU cores to use (use all available cores)
NUM_CORES = multiprocessing.cpu_count()

def load_clustering_config_model(network_type, scale_type="scaled"):
    """Load clustering-preserved configuration model.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        scale_type: "scaled" (1500 nodes) or "unscaled" (original size)
        
    Returns:
        NetworkX DiGraph
    """
    # Try loading from pickle first (faster)
    pickle_path = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, scale_type, f"{network_type}_{scale_type}_clustering_config_model.pkl")
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle: {e}")
    
    # Fall back to GEXF
    gexf_path = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, scale_type, f"{network_type}_{scale_type}_clustering_config_model.gexf")
    
    if os.path.exists(gexf_path):
        return nx.read_gexf(gexf_path)
    
    print(f"Error: Could not find model file for {network_type} ({scale_type})")
    return None

def targeted_attack(G, attack_strategy, removal_fraction):
    """Perform targeted attack on a graph.
    
    Args:
        G: NetworkX graph
        attack_strategy: 'degree' or 'betweenness'
        removal_fraction: Fraction of edges to remove
        
    Returns:
        Size of the largest connected component as a fraction of the original size
    """
    # Make a copy to avoid modifying the original
    G_copy = G.copy()
    
    # Convert to undirected for attack analysis
    G_undirected = G_copy.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # Calculate centrality based on attack strategy
    if attack_strategy == 'degree':
        centrality = dict(G_undirected.degree())
    elif attack_strategy == 'betweenness':
        centrality = nx.betweenness_centrality(G_undirected)
    else:
        raise ValueError(f"Unknown attack strategy: {attack_strategy}")
    
    # Sort edges by combined centrality of their endpoints (descending)
    all_edges = list(G_undirected.edges())
    edge_centrality = [(u, v, centrality[u] + centrality[v]) for u, v in all_edges]
    edge_centrality.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate number of edges to remove
    num_edges_to_remove = int(len(all_edges) * removal_fraction)
    
    # Remove edges with highest centrality
    edges_to_remove = [(u, v) for u, v, _ in edge_centrality[:num_edges_to_remove]]
    G_undirected.remove_edges_from(edges_to_remove)
    
    # Find the largest connected component
    if nx.is_connected(G_undirected):
        largest_cc_size = original_size
    else:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        largest_cc_size = len(largest_cc)
    
    # Return the relative size
    return largest_cc_size / original_size

def process_removal_fraction(params):
    """Process a single removal fraction (for parallel processing).
    
    Args:
        params: Tuple of (G, attack_strategy, fraction)
        
    Returns:
        Dictionary with results for this fraction
    """
    G, attack_strategy, fraction = params
    lcc_size = targeted_attack(G, attack_strategy, fraction)
    
    return {
        'removal_probability': fraction,
        'lcc_size': lcc_size,
        'mean_lcc_size': lcc_size  # For compatibility with other results
    }

def run_targeted_attack_analysis(network_type, attack_strategy, scale_type="scaled"):
    """Run targeted attack analysis on a clustering-preserved configuration model.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        attack_strategy: 'degree' or 'betweenness'
        scale_type: "scaled" (1500 nodes) or "unscaled" (original size)
        
    Returns:
        DataFrame with attack results
    """
    print(f"\nRunning {attack_strategy} attack analysis on {network_type.upper()} {scale_type} clustering-preserved configuration model")
    
    # Load the network
    G = load_clustering_config_model(network_type, scale_type)
    
    if G is None:
        print(f"Error: Failed to load network {network_type} ({scale_type})")
        return None
    
    # Print network info
    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Using {NUM_CORES} CPU cores for parallel processing")
    
    # Prepare parameters for parallel processing
    params = [(G, attack_strategy, fraction) for fraction in REMOVAL_FRACTIONS]
    
    # Run attacks in parallel
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        with tqdm(total=len(REMOVAL_FRACTIONS), desc=f"{attack_strategy.capitalize()} Attack") as pbar:
            results = []
            for result in pool.imap_unordered(process_removal_fraction, params):
                results.append(result)
                pbar.update(1)
    
    # Sort results by removal probability
    results.sort(key=lambda x: x['removal_probability'])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def save_attack_results(results_df, network_type, attack_strategy, scale_type="scaled"):
    """Save attack results to CSV file.
    
    Args:
        results_df: DataFrame with attack results
        network_type: Network type for naming
        attack_strategy: Attack strategy for naming
        scale_type: Scale type for naming
    """
    if results_df is None:
        return
        
    if scale_type == "scaled":
        output_path = os.path.join(RESULTS_DIR, f"{network_type}_clustering_config_model_{attack_strategy}_attack_results.csv")
    else:
        output_path = os.path.join(RESULTS_DIR, f"{network_type}_{scale_type}_clustering_config_model_{attack_strategy}_attack_results.csv")
    
    results_df.to_csv(output_path, index=False)
    print(f"Attack results saved to: {output_path}")

def plot_attack_results(results_df, network_type, attack_strategy, scale_type="scaled"):
    """Plot attack results.
    
    Args:
        results_df: DataFrame with attack results
        network_type: Network type for naming
        attack_strategy: Attack strategy for naming
        scale_type: Scale type for naming
    """
    if results_df is None:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Get network display name
    network_names = {
        'eb': 'Ellipsoid Body',
        'fb': 'Fan-shaped Body',
        'mb_kc': 'Mushroom Body Kenyon Cells'
    }
    network_display = network_names.get(network_type, network_type.upper())
    
    # Plot the results
    plt.plot(results_df['removal_probability'], results_df['mean_lcc_size'], 
             marker='o', markersize=4, linewidth=2, label='LCC Size')
    
    # Add threshold line
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='5% Threshold')
    
    # Find critical threshold (approximation)
    threshold_idx = np.where(results_df['mean_lcc_size'] < 0.05)[0]
    if len(threshold_idx) > 0:
        threshold = results_df['removal_probability'].iloc[threshold_idx[0]]
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=1, 
                   label=f'Critical Threshold ({threshold:.2f})')
    
    # Format plot
    plt.title(f"{network_display} {scale_type.capitalize()} Clustering Config Model {attack_strategy.capitalize()} Attack")
    plt.xlabel("Edge Removal Fraction")
    plt.ylabel("Relative Size of Largest Connected Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    if scale_type == "scaled":
        output_path = os.path.join(FIGURES_DIR, f"{network_type}_clustering_config_model_{attack_strategy}_attack.png")
    else:
        output_path = os.path.join(FIGURES_DIR, f"{network_type}_{scale_type}_clustering_config_model_{attack_strategy}_attack.png")
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attack plot saved to: {output_path}")

def main():
    """Main function to run the script."""
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process each network type, attack strategy, and scale
    network_types = ['eb', 'fb', 'mb_kc']
    attack_strategies = ["degree", "betweenness"]
    scale_types = ["scaled", "unscaled"]  # Now includes both scaled and unscaled models
    
    for network_type in network_types:
        for attack_strategy in attack_strategies:
            for scale_type in scale_types:
                # Run attack analysis
                results_df = run_targeted_attack_analysis(network_type, attack_strategy, scale_type)
                
                # Save and plot results
                save_attack_results(results_df, network_type, attack_strategy, scale_type)
                plot_attack_results(results_df, network_type, attack_strategy, scale_type)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Use 'spawn' start method for better multiprocessing compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main() 