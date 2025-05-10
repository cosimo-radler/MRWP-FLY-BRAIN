#!/usr/bin/env python3
"""
Targeted Attack Analysis Script for Drosophila Circuit Robustness Analysis

This script performs targeted attack analysis on the ellipsoid-body (EB),
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks by removing 
edges based on node degree, and compares to random edge removal.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.interpolate import interp1d
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
NUM_SIMULATIONS = 20

# Number of fraction steps to evaluate
NUM_FRACTION_STEPS = 50

def load_network(network_type):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))

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
    print(f"\nRunning targeted attack analysis for {network_name} network...")
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
                          f"{network_name.lower().replace(' ', '_')}_{attack_strategy}_attack_results.csv"), 
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

def plot_targeted_vs_random(network_name, targeted_df, random_df, targeted_params, random_params):
    """Plot targeted attack results compared to random edge removal.
    
    Args:
        network_name: Name of the network for the plot title
        targeted_df: DataFrame with targeted attack results
        random_df: DataFrame with random attack results
        targeted_params: Dictionary with targeted attack parameters
        random_params: Dictionary with random attack parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot targeted attack results
    plt.errorbar(
        targeted_df['removal_fraction'], 
        targeted_df['mean_lcc_size'],
        yerr=targeted_df['std_lcc_size'],
        fmt='o-', 
        color='red',
        alpha=0.7,
        label=f"Targeted Attack (fc ≈ {targeted_params['critical_fraction']:.3f})"
    )
    
    # Plot random attack results
    plt.errorbar(
        random_df['removal_fraction'], 
        random_df['mean_lcc_size'],
        yerr=random_df['std_lcc_size'],
        fmt='o-', 
        color='blue',
        alpha=0.7,
        label=f"Random Attack (fc ≈ {random_params['critical_fraction']:.3f})"
    )
    
    # Add vertical lines for critical fractions
    plt.axvline(x=targeted_params['critical_fraction'], color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=random_params['critical_fraction'], color='blue', linestyle='--', alpha=0.5)
    
    # Add robustness indices to the legend
    targeted_ri = targeted_params['robustness_index']
    random_ri = random_params['robustness_index']
    plt.plot([], [], ' ', label=f"Targeted Robustness: {targeted_ri:.3f}")
    plt.plot([], [], ' ', label=f"Random Robustness: {random_ri:.3f}")
    
    plt.xlabel('Edge Removal Fraction')
    plt.ylabel('Normalized Largest Connected Component Size')
    plt.title(f'{network_name} Network: Targeted vs Random Attack')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_name.lower().replace(' ', '_')}_targeted_vs_random.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_and_compare_attacks(network_type, network_name):
    """Run both targeted and random attacks and compare results.
    
    Args:
        network_type: Type identifier ('eb', 'fb', or 'mb_kc')
        network_name: Full name for plots and results
        
    Returns:
        Tuple of (targeted_params, random_params)
    """
    # Load network
    G = load_network(network_type)
    
    # Run targeted attack analysis
    targeted_df = run_targeted_attack_analysis(G, network_name, 'degree')
    
    # Run random attack analysis (for comparison)
    random_df = run_targeted_attack_analysis(G, network_name, 'random')
    
    # Estimate critical fractions
    targeted_params = estimate_critical_fraction(targeted_df)
    random_params = estimate_critical_fraction(random_df)
    
    # Plot comparison
    plot_targeted_vs_random(network_name, targeted_df, random_df, targeted_params, random_params)
    
    # Print results
    print(f"\nResults for {network_name}:")
    print(f"  Targeted Attack Critical Fraction: {targeted_params['critical_fraction']:.4f}")
    print(f"  Targeted Attack Robustness Index: {targeted_params['robustness_index']:.4f}")
    print(f"  Random Attack Critical Fraction: {random_params['critical_fraction']:.4f}")
    print(f"  Random Attack Robustness Index: {random_params['robustness_index']:.4f}")
    
    return targeted_params, random_params

def main():
    """Main execution function."""
    print("Running Targeted Attack Analysis...")
    
    # Run analysis for EB network
    eb_targeted_params, eb_random_params = run_and_compare_attacks('eb', 'Ellipsoid Body')
    
    # Run analysis for FB network
    fb_targeted_params, fb_random_params = run_and_compare_attacks('fb', 'Fan-shaped Body')
    
    # Run analysis for MB-KC network
    mb_targeted_params, mb_random_params = run_and_compare_attacks('mb_kc', 'Mushroom Body Kenyon Cell')
    
    # Save parameters
    params = {
        'Ellipsoid_Body': {
            'targeted': eb_targeted_params,
            'random': eb_random_params
        },
        'Fan_Shaped_Body': {
            'targeted': fb_targeted_params,
            'random': fb_random_params
        },
        'Mushroom_Body_Kenyon_Cell': {
            'targeted': mb_targeted_params,
            'random': mb_random_params
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "targeted_attack_parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Compare all networks with both attack types
    plt.figure(figsize=(12, 8))
    
    # Load results
    eb_targeted_df = pd.read_csv(os.path.join(RESULTS_DIR, "ellipsoid_body_degree_attack_results.csv"))
    eb_random_df = pd.read_csv(os.path.join(RESULTS_DIR, "ellipsoid_body_random_attack_results.csv"))
    fb_targeted_df = pd.read_csv(os.path.join(RESULTS_DIR, "fan-shaped_body_degree_attack_results.csv"))
    fb_random_df = pd.read_csv(os.path.join(RESULTS_DIR, "fan-shaped_body_random_attack_results.csv"))
    mb_targeted_df = pd.read_csv(os.path.join(RESULTS_DIR, "mushroom_body_kenyon_cell_degree_attack_results.csv"))
    mb_random_df = pd.read_csv(os.path.join(RESULTS_DIR, "mushroom_body_kenyon_cell_random_attack_results.csv"))
    
    # Plot all curves
    plt.plot(eb_targeted_df['removal_fraction'], eb_targeted_df['mean_lcc_size'], 'b-', 
             label=f"EB Targeted (fc={eb_targeted_params['critical_fraction']:.3f})")
    plt.plot(eb_random_df['removal_fraction'], eb_random_df['mean_lcc_size'], 'b--', 
             label=f"EB Random (fc={eb_random_params['critical_fraction']:.3f})")
    
    plt.plot(fb_targeted_df['removal_fraction'], fb_targeted_df['mean_lcc_size'], 'g-', 
             label=f"FB Targeted (fc={fb_targeted_params['critical_fraction']:.3f})")
    plt.plot(fb_random_df['removal_fraction'], fb_random_df['mean_lcc_size'], 'g--', 
             label=f"FB Random (fc={fb_random_params['critical_fraction']:.3f})")
    
    plt.plot(mb_targeted_df['removal_fraction'], mb_targeted_df['mean_lcc_size'], 'r-', 
             label=f"MB-KC Targeted (fc={mb_targeted_params['critical_fraction']:.3f})")
    plt.plot(mb_random_df['removal_fraction'], mb_random_df['mean_lcc_size'], 'r--', 
             label=f"MB-KC Random (fc={mb_random_params['critical_fraction']:.3f})")
    
    plt.xlabel('Edge Removal Fraction (f)')
    plt.ylabel('Normalized Largest Connected Component Size')
    plt.title('Targeted vs Random Attack: Network Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "targeted_attack_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTargeted attack analysis complete!")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 