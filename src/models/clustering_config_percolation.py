#!/usr/bin/env python3
"""
Clustering-Preserved Configuration Models Percolation Analysis

This script performs percolation analysis on the configuration models that
preserve both degree distribution and clustering coefficient of the original
brain region networks. The analysis involves randomly removing edges at
different probabilities and measuring the size of the largest connected component.
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

# Percolation parameters
REMOVAL_PROBABILITIES = np.arange(0, 1.01, 0.01)
NUM_ITERATIONS = 10  # Number of iterations for each removal probability

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

def random_edge_percolation(G, removal_probability):
    """Perform random edge percolation on a graph.
    
    Args:
        G: NetworkX graph
        removal_probability: Probability of removing an edge
        
    Returns:
        Size of the largest connected component as a fraction of the original size
    """
    # Make a copy to avoid modifying the original
    G_copy = G.copy()
    
    # Convert to undirected for percolation analysis
    G_undirected = G_copy.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # Get all edges
    all_edges = list(G_undirected.edges())
    
    # Remove edges with the given probability
    edges_to_remove = []
    for edge in all_edges:
        if random.random() < removal_probability:
            edges_to_remove.append(edge)
    
    G_undirected.remove_edges_from(edges_to_remove)
    
    # Find the largest connected component
    if nx.is_connected(G_undirected):
        largest_cc_size = original_size
    else:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        largest_cc_size = len(largest_cc)
    
    # Return the relative size
    return largest_cc_size / original_size

def process_removal_probability(params):
    """Process a single removal probability with multiple iterations (for parallel processing).
    
    Args:
        params: Tuple of (G, p, num_iterations)
        
    Returns:
        Dictionary with results for this probability
    """
    G, p, num_iterations = params
    
    # Run multiple iterations
    lcc_sizes = []
    for i in range(num_iterations):
        lcc_size = random_edge_percolation(G, p)
        lcc_sizes.append(lcc_size)
    
    # Calculate mean and standard deviation
    mean_lcc_size = np.mean(lcc_sizes)
    std_lcc_size = np.std(lcc_sizes)
    
    return {
        'removal_probability': p,
        'mean_lcc_size': mean_lcc_size,
        'std_lcc_size': std_lcc_size
    }

def run_percolation_analysis(network_type, scale_type="scaled"):
    """Run percolation analysis on a clustering-preserved configuration model.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        scale_type: "scaled" (1500 nodes) or "unscaled" (original size)
        
    Returns:
        DataFrame with percolation results
    """
    print(f"\nRunning percolation analysis on {network_type.upper()} {scale_type} clustering-preserved configuration model")
    
    # Load the network
    G = load_clustering_config_model(network_type, scale_type)
    
    if G is None:
        print(f"Error: Failed to load network {network_type} ({scale_type})")
        return None
    
    # Print network info
    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Using {NUM_CORES} CPU cores for parallel processing")
    
    # Prepare parameters for parallel processing
    params = [(G, p, NUM_ITERATIONS) for p in REMOVAL_PROBABILITIES]
    
    # Run percolation for each removal probability in parallel
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        with tqdm(total=len(REMOVAL_PROBABILITIES), desc="Percolation Analysis") as pbar:
            results = []
            for result in pool.imap_unordered(process_removal_probability, params):
                results.append(result)
                pbar.update(1)
    
    # Sort results by removal probability
    results.sort(key=lambda x: x['removal_probability'])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def save_percolation_results(results_df, network_type, scale_type="scaled"):
    """Save percolation results to CSV file.
    
    Args:
        results_df: DataFrame with percolation results
        network_type: Network type for naming
        scale_type: Scale type for naming
    """
    if results_df is None:
        return
        
    if scale_type == "scaled":
        output_path = os.path.join(RESULTS_DIR, f"{network_type}_clustering_config_model_percolation_results.csv")
    else:
        output_path = os.path.join(RESULTS_DIR, f"{network_type}_{scale_type}_clustering_config_model_percolation_results.csv")
    
    results_df.to_csv(output_path, index=False)
    print(f"Percolation results saved to: {output_path}")

def plot_percolation_results(results_df, network_type, scale_type="scaled"):
    """Plot percolation results.
    
    Args:
        results_df: DataFrame with percolation results
        network_type: Network type for naming
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
             marker='o', markersize=4, linewidth=2, label='Mean LCC Size')
    
    # Add standard deviation as shaded area
    plt.fill_between(
        results_df['removal_probability'],
        results_df['mean_lcc_size'] - results_df['std_lcc_size'],
        results_df['mean_lcc_size'] + results_df['std_lcc_size'],
        alpha=0.3
    )
    
    # Add threshold line
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='5% Threshold')
    
    # Find critical threshold (approximation)
    threshold_idx = np.where(results_df['mean_lcc_size'] < 0.05)[0]
    if len(threshold_idx) > 0:
        threshold = results_df['removal_probability'].iloc[threshold_idx[0]]
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=1, 
                   label=f'Critical Threshold ({threshold:.2f})')
    
    # Format plot
    plt.title(f"{network_display} {scale_type.capitalize()} Clustering-Preserved Configuration Model Percolation")
    plt.xlabel("Edge Removal Probability")
    plt.ylabel("Relative Size of Largest Connected Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    if scale_type == "scaled":
        output_path = os.path.join(FIGURES_DIR, f"{network_type}_clustering_config_model_percolation.png")
    else:
        output_path = os.path.join(FIGURES_DIR, f"{network_type}_{scale_type}_clustering_config_model_percolation.png")
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Percolation plot saved to: {output_path}")

def main():
    """Main function to run the script."""
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process each network type and scale
    network_types = ['eb', 'fb', 'mb_kc']
    scale_types = ["scaled", "unscaled"]
    
    for network_type in network_types:
        for scale_type in scale_types:
            # Run percolation analysis
            results_df = run_percolation_analysis(network_type, scale_type)
            
            # Save and plot results
            save_percolation_results(results_df, network_type, scale_type)
            plot_percolation_results(results_df, network_type, scale_type)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Use 'spawn' start method for better multiprocessing compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main() 