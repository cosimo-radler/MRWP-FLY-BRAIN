#!/usr/bin/env python3
"""
Targeted Attack Analysis for Configuration Models

This script performs targeted attacks (betweenness centrality and degree) on 
configuration models of the ellipsoid-body (EB), fan-shaped-body (FB), 
and mushroom-body (MB) Kenyon-cell networks.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import json

# Constants
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_config_model(network_type):
    """Load configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf"))

def perform_targeted_attack(G, removal_strategy):
    """
    Performs targeted attack on the network based on the specified removal strategy.
    
    Args:
        G: NetworkX graph
        removal_strategy: 'degree' or 'betweenness'
        
    Returns:
        DataFrame with attack results
    """
    # Create an undirected copy for attack analysis
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # Get the nodes sorted by the specified centrality metric
    if removal_strategy == 'degree':
        # Get degree centrality
        centrality = dict(G_undirected.degree())
    elif removal_strategy == 'betweenness':
        # Get betweenness centrality
        centrality = nx.betweenness_centrality(G_undirected)
    elif removal_strategy == 'eigenvector':
        # Get eigenvector centrality
        centrality = nx.eigenvector_centrality_numpy(G_undirected)
    else:
        raise ValueError(f"Unknown removal strategy: {removal_strategy}")
    
    # Sort nodes by centrality (highest first)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    sorted_node_ids = [node for node, _ in sorted_nodes]
    
    # Initialize results
    results = []
    
    # Perform attack by removing nodes one by one
    G_attack = G_undirected.copy()
    
    # Number of steps to take (ensure we get 50 points for 0 to 100% removal)
    n_nodes = len(sorted_node_ids)
    step_size = max(1, n_nodes // 50)
    
    # Calculate removal fractions
    removal_fractions = [i/n_nodes for i in range(0, n_nodes, step_size)]
    removal_fractions.append(1.0)  # Ensure we include complete removal
    
    for frac in tqdm(removal_fractions, desc=f"{removal_strategy} attack"):
        # Number of nodes to remove at this step
        nodes_to_remove_count = int(frac * n_nodes)
        
        if nodes_to_remove_count == 0:
            # No nodes removed yet
            lcc_size = 1.0
        else:
            # Remove nodes up to this point if not already removed
            nodes_to_remove = sorted_node_ids[:nodes_to_remove_count]
            G_attack.remove_nodes_from(nodes_to_remove)
            
            # Calculate largest connected component size
            if len(G_attack) > 0:
                largest_cc = max(nx.connected_components(G_attack), key=len)
                lcc_size = len(largest_cc) / original_size
            else:
                lcc_size = 0
            
            # Create new graph for next iteration to avoid re-removing nodes
            G_attack = G_undirected.copy()
            G_attack.remove_nodes_from(sorted_node_ids[:nodes_to_remove_count])
        
        # Store the result
        results.append({
            'removal_fraction': frac,
            'mean_lcc_size': lcc_size,
            'lcc_size': lcc_size,  # For compatibility with different formats
            'attack_strategy': removal_strategy
        })
    
    return pd.DataFrame(results)

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
    degree_results = perform_targeted_attack(G, 'degree')
    degree_results.to_csv(os.path.join(RESULTS_DIR, f"{full_name}_config_degree_attack_results.csv"), index=False)
    
    # Perform betweenness attack
    print("Performing betweenness centrality attack...")
    betweenness_results = perform_targeted_attack(G, 'betweenness')
    betweenness_results.to_csv(os.path.join(RESULTS_DIR, f"{full_name}_config_betweenness_attack_results.csv"), index=False)
    
    # Perform eigenvector attack
    try:
        print("Performing eigenvector centrality attack...")
        eigenvector_results = perform_targeted_attack(G, 'eigenvector')
        eigenvector_results.to_csv(os.path.join(RESULTS_DIR, f"{full_name}_config_eigenvector_attack_results.csv"), index=False)
    except Exception as e:
        print(f"Warning: Eigenvector centrality attack failed: {e}")

def main():
    """Main function to run the script."""
    print("Starting targeted attack analysis for configuration models...")
    
    # Process each network type
    for network_type in ['eb', 'fb', 'mb_kc']:
        process_network(network_type)
    
    print("\nTargeted attack analysis completed for all configuration models!")

if __name__ == "__main__":
    main() 