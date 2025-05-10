#!/usr/bin/env python3
"""
Betweenness Attack Analysis Script for Drosophila Circuit Robustness Analysis

This script performs targeted attack analysis on the ellipsoid-body (EB),
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks by removing 
edges based on betweenness centrality, comparing to random and degree-based attacks.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from functools import partial

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure results and figures directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of Monte Carlo simulations per fraction value
NUM_SIMULATIONS = 10

# Number of fraction steps to evaluate
NUM_FRACTION_STEPS = 25

def load_network(network_type):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))

def run_betweenness_attack_simulation(G, removal_fraction):
    """Run betweenness-based targeted attack simulation for a specific edge removal fraction.
    
    Args:
        G: NetworkX graph
        removal_fraction: Fraction of edges to remove
        
    Returns:
        Dictionary with simulation results
    """
    # Create an undirected graph for analysis
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    total_edges = G_undirected.number_of_edges()
    
    # Calculate number of edges to remove
    num_edges_to_remove = int(removal_fraction * total_edges)
    
    # Create a copy of the original graph for this simulation
    G_sim = G_undirected.copy()
    
    # Calculate edge betweenness centrality
    try:
        edge_betweenness = nx.edge_betweenness_centrality(G_sim)
        
        # Sort edges by betweenness (highest first)
        sorted_edges = sorted(edge_betweenness.keys(), 
                             key=lambda x: edge_betweenness[x], 
                             reverse=True)
        
        # Remove edges with highest betweenness first
        edges_to_remove = sorted_edges[:num_edges_to_remove]
        G_sim.remove_edges_from(edges_to_remove)
    except Exception as e:
        print(f"Warning: Edge betweenness calculation failed: {e}")
        # If betweenness calculation fails, remove random edges
        edges = list(G_sim.edges())
        if edges and num_edges_to_remove > 0:
            edges_to_remove = np.random.choice(
                len(edges), 
                size=min(num_edges_to_remove, len(edges)), 
                replace=False
            )
            G_sim.remove_edges_from([edges[i] for i in edges_to_remove])
    
    # Calculate largest connected component size
    if len(G_sim) > 0:  # Check if graph is not empty
        largest_cc = max(nx.connected_components(G_sim), key=len)
        lcc_size = len(largest_cc) / original_size  # Normalized size
    else:
        lcc_size = 0
    
    return {
        'removal_fraction': removal_fraction,
        'lcc_size': lcc_size
    }

def process_fraction(args):
    """Process a single removal fraction (for parallel processing)."""
    G, removal_fraction = args
    return run_betweenness_attack_simulation(G, removal_fraction)

def run_betweenness_attack_analysis(G, network_name, num_fraction_steps=NUM_FRACTION_STEPS):
    """Run betweenness-targeted attack analysis across a range of removal fractions.
    
    Args:
        G: NetworkX graph
        network_name: Name for the results
        num_fraction_steps: Number of fraction steps to evaluate
        
    Returns:
        DataFrame with results for each fraction
    """
    print(f"\nRunning betweenness-targeted attack analysis for {network_name} network...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal fractions
    removal_fractions = np.linspace(0, 1, num_fraction_steps)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Prepare arguments for each worker
    args_list = [(G, f) for f in removal_fractions]
    
    # Run simulations in parallel
    results = list(tqdm(
        pool.imap(process_fraction, args_list),
        total=len(args_list),
        desc=f"Simulating betweenness attack"
    ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, 
                          f"{network_name.lower().replace(' ', '_')}_betweenness_attack_results.csv"), 
              index=False)
    
    return df

def estimate_critical_fraction(df):
    """Estimate the critical fraction from attack results.
    
    Args:
        df: DataFrame with attack results
        
    Returns:
        Dictionary with critical fraction and robustness index
    """
    x = df['removal_fraction'].values
    y = df['lcc_size'].values
    
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

def compare_all_attack_strategies(network_type, network_name):
    """Compare betweenness attack with degree and random attacks.
    
    Args:
        network_type: Type identifier ('eb', 'fb', or 'mb_kc')
        network_name: Full name for plots and results
    """
    # Load network
    G = load_network(network_type)
    
    # Run betweenness attack analysis
    betweenness_df = run_betweenness_attack_analysis(G, network_name)
    betweenness_params = estimate_critical_fraction(betweenness_df)
    
    # Load other attack results if they exist
    try:
        degree_df = pd.read_csv(os.path.join(RESULTS_DIR, f"{network_name.lower().replace(' ', '_')}_degree_attack_results.csv"))
        degree_params = estimate_critical_fraction(degree_df[['removal_fraction', 'mean_lcc_size']].rename(columns={'mean_lcc_size': 'lcc_size'}))
        
        random_df = pd.read_csv(os.path.join(RESULTS_DIR, f"{network_name.lower().replace(' ', '_')}_random_attack_results.csv"))
        random_params = estimate_critical_fraction(random_df[['removal_fraction', 'mean_lcc_size']].rename(columns={'mean_lcc_size': 'lcc_size'}))
        
        has_other_results = True
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Could not find previous attack results for {network_name}")
        has_other_results = False
    
    # Print betweenness results
    print(f"\nResults for {network_name}:")
    print(f"  Betweenness Attack Critical Fraction: {betweenness_params['critical_fraction']:.4f}")
    print(f"  Betweenness Attack Robustness Index: {betweenness_params['robustness_index']:.4f}")
    
    if has_other_results:
        print(f"  Degree Attack Critical Fraction: {degree_params['critical_fraction']:.4f}")
        print(f"  Degree Attack Robustness Index: {degree_params['robustness_index']:.4f}")
        print(f"  Random Attack Critical Fraction: {random_params['critical_fraction']:.4f}")
        print(f"  Random Attack Robustness Index: {random_params['robustness_index']:.4f}")
    
    # Plot comparison if other results exist
    if has_other_results:
        plt.figure(figsize=(10, 6))
        
        # Plot betweenness attack results
        plt.plot(
            betweenness_df['removal_fraction'], 
            betweenness_df['lcc_size'],
            'go-', 
            alpha=0.7,
            label=f"Betweenness Attack (fc ≈ {betweenness_params['critical_fraction']:.3f})"
        )
        
        # Plot degree-based attack results
        plt.plot(
            degree_df['removal_fraction'], 
            degree_df['mean_lcc_size'],
            'ro-', 
            alpha=0.7,
            label=f"Degree Attack (fc ≈ {degree_params['critical_fraction']:.3f})"
        )
        
        # Plot random attack results
        plt.plot(
            random_df['removal_fraction'], 
            random_df['mean_lcc_size'],
            'bo-', 
            alpha=0.7,
            label=f"Random Attack (fc ≈ {random_params['critical_fraction']:.3f})"
        )
        
        # Add vertical lines for critical fractions
        plt.axvline(x=betweenness_params['critical_fraction'], color='green', linestyle='--', alpha=0.5)
        plt.axvline(x=degree_params['critical_fraction'], color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=random_params['critical_fraction'], color='blue', linestyle='--', alpha=0.5)
        
        # Add robustness indices to the legend
        btwn_ri = betweenness_params['robustness_index']
        deg_ri = degree_params['robustness_index']
        rand_ri = random_params['robustness_index']
        plt.plot([], [], ' ', label=f"Betweenness Robustness: {btwn_ri:.3f}")
        plt.plot([], [], ' ', label=f"Degree Robustness: {deg_ri:.3f}")
        plt.plot([], [], ' ', label=f"Random Robustness: {rand_ri:.3f}")
        
        plt.xlabel('Edge Removal Fraction')
        plt.ylabel('Normalized Largest Connected Component Size')
        plt.title(f'{network_name} Network: Attack Strategy Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(FIGURES_DIR, f"{network_name.lower().replace(' ', '_')}_attack_comparison.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    return betweenness_params

def main():
    """Main execution function."""
    print("Running Betweenness Attack Analysis...")
    
    # Run analysis for EB network
    eb_params = compare_all_attack_strategies('eb', 'Ellipsoid Body')
    
    # Run analysis for FB network
    fb_params = compare_all_attack_strategies('fb', 'Fan-shaped Body')
    
    # Run analysis for MB-KC network
    mb_params = compare_all_attack_strategies('mb_kc', 'Mushroom Body Kenyon Cell')
    
    # Save parameters
    with open(os.path.join(RESULTS_DIR, "betweenness_attack_parameters.json"), 'w') as f:
        json.dump({
            'Ellipsoid_Body': eb_params,
            'Fan_Shaped_Body': fb_params,
            'Mushroom_Body_Kenyon_Cell': mb_params
        }, f, indent=2)
    
    print("\nBetweenness attack analysis complete!")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 