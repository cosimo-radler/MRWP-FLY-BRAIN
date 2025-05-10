#!/usr/bin/env python3
"""
Configuration Model Generation Script for Drosophila Circuit Robustness Analysis

This script creates configuration models of the ellipsoid-body (EB), 
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks,
all scaled to networks of 1500 nodes while preserving degree distributions.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(CONFIG_MODEL_DIR, exist_ok=True)
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
    return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))

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

def create_configuration_model(in_sequence, out_sequence, network_type):
    """Create a directed configuration model with the given degree sequences.
    
    Args:
        in_sequence: In-degree sequence
        out_sequence: Out-degree sequence
        network_type: Type of network for naming
    
    Returns:
        NetworkX DiGraph of the configuration model
    """
    print(f"Creating configuration model for {network_type} network")
    
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

def plot_degree_distribution_comparison(G_original, G_config, network_type):
    """Plot degree distribution comparison between original and configuration model.
    
    Args:
        G_original: Original network
        G_config: Configuration model network
        network_type: Type of network for naming
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original network in-degree
    in_degrees_orig = sorted([d for _, d in G_original.in_degree()], reverse=True)
    in_degree_count_orig = Counter(in_degrees_orig)
    in_deg_orig, in_cnt_orig = zip(*sorted(in_degree_count_orig.items()))
    ax1.loglog(in_deg_orig, in_cnt_orig, 'bo-', alpha=0.7, label=f"Original ({G_original.number_of_nodes()} nodes)")
    
    # Configuration model in-degree
    in_degrees_config = sorted([d for _, d in G_config.in_degree()], reverse=True)
    in_degree_count_config = Counter(in_degrees_config)
    in_deg_config, in_cnt_config = zip(*sorted(in_degree_count_config.items()))
    ax1.loglog(in_deg_config, in_cnt_config, 'go-', alpha=0.7, label=f"Config Model ({G_config.number_of_nodes()} nodes)")
    
    ax1.set_title(f"{network_type} In-Degree Distribution")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Original network out-degree
    out_degrees_orig = sorted([d for _, d in G_original.out_degree()], reverse=True)
    out_degree_count_orig = Counter(out_degrees_orig)
    out_deg_orig, out_cnt_orig = zip(*sorted(out_degree_count_orig.items()))
    ax2.loglog(out_deg_orig, out_cnt_orig, 'ro-', alpha=0.7, label=f"Original ({G_original.number_of_nodes()} nodes)")
    
    # Configuration model out-degree
    out_degrees_config = sorted([d for _, d in G_config.out_degree()], reverse=True)
    out_degree_count_config = Counter(out_degrees_config)
    out_deg_config, out_cnt_config = zip(*sorted(out_degree_count_config.items()))
    ax2.loglog(out_deg_config, out_cnt_config, 'mo-', alpha=0.7, label=f"Config Model ({G_config.number_of_nodes()} nodes)")
    
    ax2.set_title(f"{network_type} Out-Degree Distribution")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Degree rank plot - in-degree
    ax3.loglog(range(1, len(in_degrees_orig) + 1), in_degrees_orig, 'b-', alpha=0.7, label="Original In-Degree")
    ax3.loglog(range(1, len(in_degrees_config) + 1), in_degrees_config, 'g-', alpha=0.7, label="Config Model In-Degree")
    ax3.set_title(f"{network_type} In-Degree Rank Plot")
    ax3.set_xlabel("Rank")
    ax3.set_ylabel("In-Degree")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Degree rank plot - out-degree
    ax4.loglog(range(1, len(out_degrees_orig) + 1), out_degrees_orig, 'r-', alpha=0.7, label="Original Out-Degree")
    ax4.loglog(range(1, len(out_degrees_config) + 1), out_degrees_config, 'm-', alpha=0.7, label="Config Model Out-Degree")
    ax4.set_title(f"{network_type} Out-Degree Rank Plot")
    ax4.set_xlabel("Rank")
    ax4.set_ylabel("Out-Degree")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_config_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_network(G, network_type):
    """Save network to GEXF file.
    
    Args:
        G: NetworkX graph
        network_type: Type of network for naming
    """
    output_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    nx.write_gexf(G, output_path)
    print(f"Saved configuration model to {output_path}")

def calculate_network_metrics(G):
    """Calculate basic network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    # Convert to undirected for some metrics
    G_undirected = G.to_undirected()
    
    metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G_undirected),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
    }
    
    # Try to calculate metrics that might fail on disconnected graphs
    try:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        largest_subgraph = G.subgraph(largest_cc)
        
        # Average shortest path and diameter on largest connected component
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(largest_subgraph)
        metrics["diameter"] = nx.diameter(largest_subgraph)
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        # If the graph is disconnected or has other issues
        metrics["avg_shortest_path"] = float('nan')
        metrics["diameter"] = float('nan')
    
    return metrics

def process_network(network_type):
    """Process a single network to create a configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
    
    Returns:
        NetworkX DiGraph of the configuration model
    """
    print(f"\nProcessing {network_type} network")
    
    # Load original network
    G_original = load_network(network_type)
    
    # Get scaled degree sequences
    in_sequence, out_sequence = get_scaled_degree_sequence(G_original, TARGET_NODES)
    
    # Create configuration model
    G_config = create_configuration_model(in_sequence, out_sequence, network_type)
    
    if G_config:
        # Calculate metrics for original and configuration model
        orig_metrics = calculate_network_metrics(G_original)
        config_metrics = calculate_network_metrics(G_config)
        
        print("\nNetwork metrics comparison:")
        print(f"{'Metric':<20} {'Original':<15} {'Config Model':<15}")
        print("-" * 50)
        for metric in orig_metrics:
            print(f"{metric:<20} {orig_metrics[metric]:<15.4f} {config_metrics[metric]:<15.4f}")
        
        # Plot degree distribution comparison
        plot_degree_distribution_comparison(G_original, G_config, network_type)
        
        # Save configuration model
        save_network(G_config, network_type)
        
        return G_config
    
    return None

def main():
    """Main function to process all networks."""
    # Process EB network
    eb_config = process_network('eb')
    
    # Process FB network
    fb_config = process_network('fb')
    
    # Process MB-KC network
    mb_kc_config = process_network('mb_kc')
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame()
    
    for network_type, G_config in zip(['eb', 'fb', 'mb_kc'], [eb_config, fb_config, mb_kc_config]):
        if G_config:
            metrics = calculate_network_metrics(G_config)
            metrics['network_type'] = f"{network_type}_config_model"
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    
    metrics_df.to_csv(os.path.join(CONFIG_MODEL_DIR, "config_model_metrics.csv"), index=False)
    print(f"\nSaved metrics to {os.path.join(CONFIG_MODEL_DIR, 'config_model_metrics.csv')}")

if __name__ == "__main__":
    main() 