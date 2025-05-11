#!/usr/bin/env python3
"""
Clustering-Preserving Configuration Model Generation Script

This script creates configuration models of the ellipsoid-body (EB), 
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks
while preserving both the degree distribution and clustering coefficient
of the original networks. Both unscaled (original size) and scaled (1500 nodes)
versions are generated.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random
import time
import sys
import pickle
from tqdm import tqdm

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GEPHI_REAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gephi Graphs/real_models")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
CLUSTERING_CONFIG_MODEL_DIR = os.path.join(CONFIG_MODEL_DIR, "clustering")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(CLUSTERING_CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(CLUSTERING_CONFIG_MODEL_DIR, "unscaled"), exist_ok=True)
os.makedirs(os.path.join(CLUSTERING_CONFIG_MODEL_DIR, "scaled"), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Target number of nodes for scaling
TARGET_NODES = 1500

def load_network(network_type):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    # Try to load from the Gephi real models directory first, then fall back to the data directory
    gephi_path = os.path.join(GEPHI_REAL_MODELS_DIR, f"{network_type}_network.gexf")
    data_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    
    if os.path.exists(gephi_path):
        return nx.read_gexf(gephi_path)
    else:
        return nx.read_gexf(data_path)

def get_scaled_degree_sequence(G, target_nodes):
    """Extract and scale the degree sequences from a directed graph.
    
    Args:
        G: NetworkX directed graph
        target_nodes: Target number of nodes for scaling
    
    Returns:
        Tuple of (scaled in-degree sequence, scaled out-degree sequence)
    """
    # Get original degree sequences
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    # Calculate scaling factor
    scaling_factor = target_nodes / G.number_of_nodes()
    print(f"Scaling factor: {scaling_factor:.2f} (original nodes: {G.number_of_nodes()}, target: {target_nodes})")
    
    # Scale the degree sequences while preserving the distribution shape
    # We'll replicate each degree value proportionally to the scaling factor
    scaled_in_degrees = []
    scaled_out_degrees = []
    
    # Count occurrences of each degree value
    in_degree_counts = Counter(in_degrees)
    out_degree_counts = Counter(out_degrees)
    
    # Scale each degree value
    for degree, count in in_degree_counts.items():
        scaled_count = int(count * scaling_factor)
        scaled_in_degrees.extend([degree] * scaled_count)
    
    for degree, count in out_degree_counts.items():
        scaled_count = int(count * scaling_factor)
        scaled_out_degrees.extend([degree] * scaled_count)
    
    # Adjust lengths to exactly match target_nodes
    while len(scaled_in_degrees) < target_nodes:
        scaled_in_degrees.append(random.choice(in_degrees))
    while len(scaled_in_degrees) > target_nodes:
        scaled_in_degrees.pop()
        
    while len(scaled_out_degrees) < target_nodes:
        scaled_out_degrees.append(random.choice(out_degrees))
    while len(scaled_out_degrees) > target_nodes:
        scaled_out_degrees.pop()
    
    # Ensure sum of in-degrees equals sum of out-degrees
    total_in = sum(scaled_in_degrees)
    total_out = sum(scaled_out_degrees)
    
    if total_in != total_out:
        print(f"Adjusting degree sequences to balance (in: {total_in}, out: {total_out})")
        # Find the difference
        diff = total_in - total_out
        
        # Keep trying to adjust until balanced
        max_attempts = 100
        attempts = 0
        
        while total_in != total_out and attempts < max_attempts:
            attempts += 1
            
            # Adjust the larger sum
            if diff > 0:
                # In-degrees sum is larger, reduce some in-degrees
                for _ in range(min(abs(diff), 10)):  # Adjust at most 10 degrees at a time
                    # Find a non-zero degree to decrement
                    indices = [i for i, d in enumerate(scaled_in_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        scaled_in_degrees[idx] -= 1
                        total_in -= 1
            else:
                # Out-degrees sum is larger, reduce some out-degrees
                for _ in range(min(abs(diff), 10)):  # Adjust at most 10 degrees at a time
                    # Find a non-zero degree to decrement
                    indices = [i for i, d in enumerate(scaled_out_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        scaled_out_degrees[idx] -= 1
                        total_out -= 1
            
            # Recalculate the difference
            diff = total_in - total_out
            
            if attempts % 10 == 0:
                print(f"  Attempt {attempts}: diff = {diff} (in: {total_in}, out: {total_out})")
        
        if total_in != total_out:
            # If we still can't balance, force it by creating a new sequence
            print(f"Could not balance sequences after {max_attempts} attempts. Forcing balance...")
            total = min(total_in, total_out)
            
            # Force both sequences to have the same total by scaling
            scale_in = total / total_in
            scale_out = total / total_out
            
            # Create new sequences with the same total
            scaled_in_degrees = [max(1, int(d * scale_in)) for d in scaled_in_degrees]
            scaled_out_degrees = [max(1, int(d * scale_out)) for d in scaled_out_degrees]
            
            # Adjust lengths to exactly match target_nodes
            while len(scaled_in_degrees) < target_nodes:
                scaled_in_degrees.append(1)
            while len(scaled_in_degrees) > target_nodes:
                scaled_in_degrees.pop()
                
            while len(scaled_out_degrees) < target_nodes:
                scaled_out_degrees.append(1)
            while len(scaled_out_degrees) > target_nodes:
                scaled_out_degrees.pop()
            
            # Make final adjustment to ensure equal sums
            total_in = sum(scaled_in_degrees)
            total_out = sum(scaled_out_degrees)
            
            if total_in > total_out:
                # Remove the difference from the last non-zero element
                diff = total_in - total_out
                for i in range(len(scaled_in_degrees)):
                    if scaled_in_degrees[i] > diff:
                        scaled_in_degrees[i] -= diff
                        break
                    elif scaled_in_degrees[i] > 0:
                        diff -= scaled_in_degrees[i]
                        scaled_in_degrees[i] = 0
            elif total_out > total_in:
                # Remove the difference from the last non-zero element
                diff = total_out - total_in
                for i in range(len(scaled_out_degrees)):
                    if scaled_out_degrees[i] > diff:
                        scaled_out_degrees[i] -= diff
                        break
                    elif scaled_out_degrees[i] > 0:
                        diff -= scaled_out_degrees[i]
                        scaled_out_degrees[i] = 0
    
    # Verify equal sums
    assert sum(scaled_in_degrees) == sum(scaled_out_degrees), "Degree sequences must have equal sums"
    print(f"Balanced degree sequences: sum = {sum(scaled_in_degrees)}")
    
    return scaled_in_degrees, scaled_out_degrees

def create_configuration_model(in_sequence, out_sequence, network_type, scale_type="unscaled"):
    """Create a directed configuration model with the given degree sequences.
    
    Args:
        in_sequence: In-degree sequence
        out_sequence: Out-degree sequence
        network_type: Type of network for naming
        scale_type: "unscaled" or "scaled"
    
    Returns:
        NetworkX DiGraph of the configuration model
    """
    print(f"Creating {scale_type} configuration model for {network_type} network")
    
    # Create a directed configuration model
    try:
        G_config = nx.directed_configuration_model(
            in_sequence, 
            out_sequence,
            seed=42  # For reproducibility
        )
        
        # Remove parallel edges
        G_config = nx.DiGraph(G_config)
        
        # Remove self-loops
        G_config.remove_edges_from(nx.selfloop_edges(G_config))
        
        print(f"Configuration model created: {G_config.number_of_nodes()} nodes, {G_config.number_of_edges()} edges")
        
        # Relabel nodes to integers starting from 0
        G_config = nx.convert_node_labels_to_integers(G_config)
        
        return G_config
    
    except Exception as e:
        print(f"Error creating configuration model: {e}")
        return None

def edge_swap_to_adjust_clustering(G, target_clustering, max_iterations=10000, tolerance=0.001):
    """Adjust the graph's clustering coefficient to match the target value.
    
    Args:
        G: NetworkX graph
        target_clustering: Target clustering coefficient to achieve
        max_iterations: Maximum number of edge swap iterations
        tolerance: Acceptable difference between current and target clustering
        
    Returns:
        NetworkX graph with adjusted clustering coefficient
    """
    G_undirected = G.to_undirected()
    current_clustering = nx.average_clustering(G_undirected)
    
    print(f"Initial clustering coefficient: {current_clustering:.6f}")
    print(f"Target clustering coefficient: {target_clustering:.6f}")
    
    if abs(current_clustering - target_clustering) <= tolerance:
        print("Current clustering is already within tolerance of target")
        return G
    
    iterations = 0
    G_best = G.copy()
    best_diff = abs(current_clustering - target_clustering)
    
    pbar = tqdm(total=max_iterations, desc="Adjusting clustering coefficient")
    
    while iterations < max_iterations:
        # Make a copy of the graph to try edge swaps
        G_trial = G.copy()
        
        # Decide whether to increase or decrease clustering
        if current_clustering < target_clustering:
            # Need to increase clustering - look for potential triangles
            success = try_to_increase_clustering(G_trial)
        else:
            # Need to decrease clustering - break existing triangles
            success = try_to_decrease_clustering(G_trial)
        
        if success:
            # Check if the edge swap improved the clustering coefficient
            trial_undirected = G_trial.to_undirected()
            trial_clustering = nx.average_clustering(trial_undirected)
            diff = abs(trial_clustering - target_clustering)
            
            # If this is better than our previous best, keep it
            if diff < best_diff:
                G = G_trial
                G_best = G_trial.copy()
                best_diff = diff
                current_clustering = trial_clustering
                
                if best_diff <= tolerance:
                    print(f"\nTarget clustering achieved within tolerance after {iterations} iterations")
                    print(f"Final clustering coefficient: {current_clustering:.6f}")
                    break
        
        iterations += 1
        pbar.update(1)
        
        # Print progress periodically
        if iterations % 500 == 0:
            print(f"\nIteration {iterations}: Current clustering = {current_clustering:.6f}, "
                  f"diff = {best_diff:.6f}")
    
    pbar.close()
    
    if iterations == max_iterations:
        print(f"\nMaximum iterations reached. Best clustering achieved: {current_clustering:.6f}")
        print(f"Difference from target: {best_diff:.6f}")
        return G_best
    
    return G_best

def try_to_increase_clustering(G):
    """Try to increase clustering by adding triangles.
    
    Args:
        G: NetworkX graph to modify
        
    Returns:
        Boolean indicating whether a valid edge swap was performed
    """
    # Get all nodes with at least two connections
    nodes_with_multiple_edges = [n for n in G.nodes() if G.degree(n) >= 2]
    
    if not nodes_with_multiple_edges:
        return False
    
    # Try to find a potential triangle-forming edge swap
    for _ in range(50):  # Try a limited number of times
        # Select a random node with multiple connections
        node = random.choice(nodes_with_multiple_edges)
        
        # Get neighbors of the node (fix: ensure these are actual outgoing edges)
        neighbors = list(G.successors(node))
        
        if len(neighbors) < 2:
            continue
            
        # Choose two random neighbors that are not connected to each other
        if len(neighbors) >= 2:
            n1, n2 = random.sample(neighbors, 2)
        else:
            continue
            
        # Verify that the edges exist before trying to manipulate them
        if not G.has_edge(node, n1) or not G.has_edge(node, n2):
            continue
        
        if not G.has_edge(n1, n2) and not G.has_edge(n2, n1):
            # Find a random edge to swap with
            all_edges = list(G.edges())
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Skip edges connected to our target nodes
                if u in [node, n1, n2] or v in [node, n1, n2]:
                    continue
                
                # Verify the swap won't create multiple edges or self-loops
                if not (G.has_edge(n1, v) or G.has_edge(v, n1) or 
                        G.has_edge(n2, u) or G.has_edge(u, n2) or
                        n1 == v or n2 == u):
                    
                    # Double-check that the edges we're about to remove actually exist
                    if G.has_edge(node, n1) and G.has_edge(u, v):
                        # Remove existing edges
                        G.remove_edge(node, n1)
                        G.remove_edge(u, v)
                        
                        # Add new edges to form triangle
                        G.add_edge(n1, n2)
                        G.add_edge(u, node)
                        
                        return True
    
    return False

def try_to_decrease_clustering(G):
    """Try to decrease clustering by breaking triangles.
    
    Args:
        G: NetworkX graph to modify
        
    Returns:
        Boolean indicating whether a valid edge swap was performed
    """
    # Find triangles in the graph (this can be expensive for large graphs)
    triangles = []
    
    # Simplified approach: just look for connected triads
    for node in G.nodes():
        neighbors = list(G.successors(node))
        
        # Need at least 2 neighbors to form a triangle
        if len(neighbors) < 2:
            continue
        
        # Check pairs of neighbors to see if they're connected
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if G.has_edge(n1, n2) or G.has_edge(n2, n1):
                    # Only add if all edges exist
                    if G.has_edge(node, n1) and G.has_edge(node, n2):
                        triangles.append((node, n1, n2))
    
    if not triangles:
        return False
    
    # Try to break a random triangle
    for _ in range(50):  # Try a limited number of times
        if not triangles:
            return False
            
        # Choose a random triangle
        n1, n2, n3 = random.choice(triangles)
        
        # Select a random edge from the triangle to break
        potential_edges = []
        if G.has_edge(n1, n2):
            potential_edges.append((n1, n2))
        if G.has_edge(n2, n3):
            potential_edges.append((n2, n3))
        if G.has_edge(n3, n1):
            potential_edges.append((n3, n1))
        if G.has_edge(n2, n1):
            potential_edges.append((n2, n1))
        if G.has_edge(n3, n2):
            potential_edges.append((n3, n2))
        if G.has_edge(n1, n3):
            potential_edges.append((n1, n3))
        
        if not potential_edges:
            continue
            
        edge_to_break = random.choice(potential_edges)
        
        # Find a random edge to swap with
        all_edges = list(G.edges())
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Skip edges connected to our triangle
            if u in [n1, n2, n3] or v in [n1, n2, n3]:
                continue
            
            # Ensure the swap won't create multiple edges or self-loops
            source, target = edge_to_break
            if not (G.has_edge(source, v) or G.has_edge(v, source) or 
                    G.has_edge(u, target) or G.has_edge(target, u) or
                    source == v or target == u):
                
                # Double-check that the edges we're about to remove actually exist
                if G.has_edge(source, target) and G.has_edge(u, v):
                    # Remove existing edges
                    G.remove_edge(source, target)
                    G.remove_edge(u, v)
                    
                    # Add new edges that don't form triangles
                    G.add_edge(source, v)
                    G.add_edge(u, target)
                    
                    return True
    
    return False

def save_network(G, network_type, scale_type="unscaled"):
    """Save network to GEXF file.
    
    Args:
        G: NetworkX graph
        network_type: Type of network for naming
        scale_type: "unscaled" or "scaled"
    """
    # Save to appropriate directory
    output_dir = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, scale_type)
    output_path = os.path.join(output_dir, f"{network_type}_{scale_type}_clustering_config_model.gexf")
    nx.write_gexf(G, output_path)
    print(f"Saved {scale_type} clustering-preserved configuration model to: {output_path}")
    
    # Also save as pickle for faster loading
    pickle_path = os.path.join(output_dir, f"{network_type}_{scale_type}_clustering_config_model.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved pickle version to: {pickle_path}")

def plot_clustering_comparison(G_original, G_config, network_type, scale_type="unscaled"):
    """Plot degree distribution comparison between original and configuration model.
    
    Args:
        G_original: Original network
        G_config: Configuration model network
        network_type: Type of network for naming
        scale_type: "unscaled" or "scaled"
    """
    # Convert to undirected for clustering calculations
    G_original_undirected = G_original.to_undirected()
    G_config_undirected = G_config.to_undirected()
    
    # Calculate clustering coefficients
    original_gcc = nx.average_clustering(G_original_undirected)
    config_gcc = nx.average_clustering(G_config_undirected)
    
    # Calculate node-level clustering
    original_clustering = nx.clustering(G_original_undirected)
    config_clustering = nx.clustering(G_config_undirected)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Distribution of clustering coefficients
    ax1.hist(list(original_clustering.values()), bins=20, alpha=0.7, 
             label=f"Original (GCC={original_gcc:.4f})")
    ax1.hist(list(config_clustering.values()), bins=20, alpha=0.7, 
             label=f"Config Model (GCC={config_gcc:.4f})")
    ax1.set_title(f"{network_type.upper()} Network: Clustering Coefficient Distribution")
    ax1.set_xlabel("Node Clustering Coefficient")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Degree vs Clustering
    original_degree = dict(G_original_undirected.degree())
    config_degree = dict(G_config_undirected.degree())
    
    original_data = [(d, original_clustering[n]) for n, d in original_degree.items()]
    config_data = [(d, config_clustering[n]) for n, d in config_degree.items()]
    
    original_x, original_y = zip(*original_data) if original_data else ([], [])
    config_x, config_y = zip(*config_data) if config_data else ([], [])
    
    ax2.scatter(original_x, original_y, alpha=0.7, label="Original")
    ax2.scatter(config_x, config_y, alpha=0.7, label="Config Model")
    ax2.set_title(f"{network_type.upper()} Network: Degree vs Clustering")
    ax2.set_xlabel("Node Degree")
    ax2.set_ylabel("Clustering Coefficient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_{scale_type}_clustering_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_network_metrics(G):
    """Calculate key network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic properties
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Degrees
    degrees = [d for _, d in G.degree()]
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    metrics['avg_degree'] = np.mean(degrees)
    metrics['max_degree'] = max(degrees)
    metrics['avg_in_degree'] = np.mean(in_degrees)
    metrics['avg_out_degree'] = np.mean(out_degrees)
    
    # Connectivity
    G_undirected = G.to_undirected()
    try:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        metrics['largest_cc_size'] = len(largest_cc) / G.number_of_nodes()
    except:
        metrics['largest_cc_size'] = 0
    
    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(G_undirected)
    
    return metrics

def process_network(network_type):
    """Process a single network, creating both unscaled and scaled configurations models
    with preserved clustering coefficients.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
    """
    print(f"\nProcessing {network_type.upper()} network...")
    
    # Load original network
    G_original = load_network(network_type)
    
    # Calculate original clustering coefficient
    G_original_undirected = G_original.to_undirected()
    original_clustering = nx.average_clustering(G_original_undirected)
    
    print(f"Original network: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
    print(f"Original clustering coefficient: {original_clustering:.6f}")
    
    # Calculate original metrics
    original_metrics = calculate_network_metrics(G_original)
    
    # 1. Create unscaled configuration model
    print("\nCreating unscaled configuration model...")
    in_degrees = [d for _, d in G_original.in_degree()]
    out_degrees = [d for _, d in G_original.out_degree()]
    
    G_unscaled = create_configuration_model(in_degrees, out_degrees, network_type, "unscaled")
    
    if G_unscaled:
        # Adjust clustering coefficient to match original
        print("\nAdjusting clustering coefficient for unscaled model...")
        G_unscaled = edge_swap_to_adjust_clustering(G_unscaled, original_clustering)
        
        # Calculate metrics after adjustment
        unscaled_metrics = calculate_network_metrics(G_unscaled)
        
        # Save metrics comparison
        metrics_df = pd.DataFrame([original_metrics, unscaled_metrics], 
                                 index=['original', 'unscaled_clustering_config'])
        metrics_df.to_csv(os.path.join(CLUSTERING_CONFIG_MODEL_DIR, 'unscaled', 
                                      f"{network_type}_unscaled_metrics_comparison.csv"))
        
        # Plot degree and clustering comparisons
        plot_clustering_comparison(G_original, G_unscaled, network_type, "unscaled")
        
        # Save unscaled model
        save_network(G_unscaled, network_type, "unscaled")
    
    # 2. Create scaled configuration model (1500 nodes)
    print("\nCreating scaled configuration model (1500 nodes)...")
    scaled_in_degrees, scaled_out_degrees = get_scaled_degree_sequence(G_original, TARGET_NODES)
    
    G_scaled = create_configuration_model(scaled_in_degrees, scaled_out_degrees, network_type, "scaled")
    
    if G_scaled:
        # Adjust clustering coefficient to match original
        print("\nAdjusting clustering coefficient for scaled model...")
        G_scaled = edge_swap_to_adjust_clustering(G_scaled, original_clustering)
        
        # Calculate metrics after adjustment
        scaled_metrics = calculate_network_metrics(G_scaled)
        
        # Save metrics comparison
        metrics_df = pd.DataFrame([original_metrics, scaled_metrics], 
                                 index=['original', 'scaled_clustering_config'])
        metrics_df.to_csv(os.path.join(CLUSTERING_CONFIG_MODEL_DIR, 'scaled', 
                                      f"{network_type}_scaled_metrics_comparison.csv"))
        
        # Plot degree and clustering comparisons
        plot_clustering_comparison(G_original, G_scaled, network_type, "scaled")
        
        # Save scaled model
        save_network(G_scaled, network_type, "scaled")

def main():
    """Main function to process all network types."""
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process each network type
    for network_type in ['eb', 'fb', 'mb_kc']:
        process_network(network_type)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 