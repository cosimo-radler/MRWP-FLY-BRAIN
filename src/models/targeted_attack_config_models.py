#!/usr/bin/env python3
"""
Targeted Attack Analysis for Configuration Models

This script performs targeted edge attacks (degree centrality and betweenness centrality) on 
configuration models of the ellipsoid-body (EB), fan-shaped-body (FB), 
and mushroom-body (MB) Kenyon-cell networks, using the same methodology as for the original networks.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import json
import multiprocessing
from functools import partial

# Constants
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Number of Monte Carlo simulations per fraction value
NUM_SIMULATIONS = 20

# Number of fraction steps to evaluate
NUM_FRACTION_STEPS = 50

def load_config_model(network_type):
    """Load configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf"))

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
    print(f"\nRunning targeted attack analysis for {network_name} configuration model...")
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
                          f"{network_name.lower().replace(' ', '_')}_config_{attack_strategy}_attack_results.csv"), 
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

def process_network(network_type):
    """
    Process a single network with different attack strategies.
    
    Args:
        network_type: Type of network ('eb', 'fb', 'mb_kc')
    """
    print(f"\nProcessing {network_type} configuration model...")
    
    # Load configuration model
    G = load_config_model(network_type)
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get full network name for file naming
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # Perform degree attack
    print("Performing degree centrality attack...")
    degree_results = run_targeted_attack_analysis(G, full_name, 'degree')
    
    # Perform betweenness attack
    print("Performing betweenness centrality attack...")
    betweenness_results = run_targeted_attack_analysis(G, full_name, 'betweenness')
    
    # Perform random attack (equivalent to bond percolation)
    print("Performing random edge removal...")
    random_results = run_targeted_attack_analysis(G, full_name, 'random')
    
    # Save parameters
    attack_params = {
        'degree': estimate_critical_fraction(degree_results),
        'betweenness': estimate_critical_fraction(betweenness_results),
        'random': estimate_critical_fraction(random_results)
    }
    
    # Save parameters to JSON
    with open(os.path.join(RESULTS_DIR, f"{full_name}_config_attack_parameters.json"), 'w') as f:
        json.dump(attack_params, f, indent=4)
        
    return {
        'degree': degree_results,
        'betweenness': betweenness_results,
        'random': random_results,
        'params': attack_params
    }

def main():
    """Main function to run the script."""
    print("Starting targeted attack analysis for configuration models...")
    
    # Process each network type
    results = {}
    for network_type in ['eb', 'fb', 'mb_kc']:
        results[network_type] = process_network(network_type)
    
    print("\nTargeted attack analysis completed for all configuration models!")

if __name__ == "__main__":
    main() 