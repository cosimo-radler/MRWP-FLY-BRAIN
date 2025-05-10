#!/usr/bin/env python3
"""
Upscaled Configuration Model Generation Script for Drosophila Circuit Robustness Analysis

This script creates upscaled configuration models of the ellipsoid-body (EB), 
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks,
all scaled to networks of 3500 nodes while preserving degree distributions.
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
GEPHI_REAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gephi Graphs/real_models")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/upscaled")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(UPSCALED_CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Target number of nodes for upscaled model
TARGET_NODES = 3500

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

def get_upscaled_degree_sequence(G, target_nodes):
    """Extract and upscale the degree sequences from a directed graph.
    
    Args:
        G: NetworkX directed graph
        target_nodes: Target number of nodes for scaling
    
    Returns:
        Tuple of (upscaled in-degree sequence, upscaled out-degree sequence)
    """
    # Get original degree sequences
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    # Calculate scaling factor
    scaling_factor = target_nodes / G.number_of_nodes()
    print(f"Scaling factor: {scaling_factor:.2f} (original nodes: {G.number_of_nodes()}, target: {target_nodes})")
    
    # Scale the degree sequences while preserving the distribution shape
    # We'll replicate each degree value proportionally to the scaling factor
    upscaled_in_degrees = []
    upscaled_out_degrees = []
    
    # Count occurrences of each degree value
    in_degree_counts = Counter(in_degrees)
    out_degree_counts = Counter(out_degrees)
    
    # Scale each degree value
    for degree, count in in_degree_counts.items():
        scaled_count = int(count * scaling_factor)
        upscaled_in_degrees.extend([degree] * scaled_count)
    
    for degree, count in out_degree_counts.items():
        scaled_count = int(count * scaling_factor)
        upscaled_out_degrees.extend([degree] * scaled_count)
    
    # Adjust lengths to exactly match target_nodes
    while len(upscaled_in_degrees) < target_nodes:
        upscaled_in_degrees.append(random.choice(in_degrees))
    while len(upscaled_in_degrees) > target_nodes:
        upscaled_in_degrees.pop()
        
    while len(upscaled_out_degrees) < target_nodes:
        upscaled_out_degrees.append(random.choice(out_degrees))
    while len(upscaled_out_degrees) > target_nodes:
        upscaled_out_degrees.pop()
    
    # Ensure sum of in-degrees equals sum of out-degrees
    total_in = sum(upscaled_in_degrees)
    total_out = sum(upscaled_out_degrees)
    
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
                    indices = [i for i, d in enumerate(upscaled_in_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        upscaled_in_degrees[idx] -= 1
                        total_in -= 1
            else:
                # Out-degrees sum is larger, reduce some out-degrees
                for _ in range(min(abs(diff), 10)):  # Adjust at most 10 degrees at a time
                    # Find a non-zero degree to decrement
                    indices = [i for i, d in enumerate(upscaled_out_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        upscaled_out_degrees[idx] -= 1
                        total_out -= 1
            
            # Recalculate the difference
            diff = total_in - total_out
            
            if attempts % 10 == 0:
                print(f"  Attempt {attempts}: diff = {diff} (in: {total_in}, out: {total_out})")

    # Final check
    total_in = sum(upscaled_in_degrees)
    total_out = sum(upscaled_out_degrees)
    print(f"Final degree sums: in-degrees = {total_in}, out-degrees = {total_out}")
    
    return upscaled_in_degrees, upscaled_out_degrees

def create_upscaled_configuration_model(in_sequence, out_sequence, network_type):
    """Create a directed configuration model with the given degree sequences.
    
    Args:
        in_sequence: In-degree sequence
        out_sequence: Out-degree sequence
        network_type: Type of network for naming
    
    Returns:
        NetworkX DiGraph of the configuration model
    """
    print(f"Creating upscaled configuration model for {network_type} network")
    
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
        
        print(f"Upscaled configuration model created: {G_config.number_of_nodes()} nodes, {G_config.number_of_edges()} edges")
        
        # Relabel nodes to integers starting from 0
        G_config = nx.convert_node_labels_to_integers(G_config)
        
        return G_config
    
    except Exception as e:
        print(f"Error creating upscaled configuration model: {e}")
        return None

def save_network(G, network_type):
    """Save network to GEXF file for analysis.
    
    Args:
        G: NetworkX graph
        network_type: Type of network for naming
    """
    # Save to the configuration models directory
    output_path = os.path.join(UPSCALED_CONFIG_MODEL_DIR, f"{network_type}_upscaled_config_model.gexf")
    nx.write_gexf(G, output_path)
    print(f"Saved upscaled configuration model to {output_path}")

def plot_degree_distribution(G, title, color, ax=None, normalize=True):
    """Plot degree distribution for a directed graph.
    
    Args:
        G: NetworkX directed graph
        title: Plot title
        color: Line color
        ax: Matplotlib axes (optional)
        normalize: Whether to normalize the distribution (default: True)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get degree distributions
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # Count occurrences of each degree
    degree_counts = Counter(degree_values)
    
    # Sort by degree
    max_degree = max(degree_counts.keys())
    x = list(range(max_degree + 1))
    y = [degree_counts.get(d, 0) for d in x]
    
    # Normalize if requested
    if normalize:
        total = sum(y)
        y = [count / total for count in y]
        ylabel = 'P(k)'
    else:
        ylabel = 'Count'
    
    # Plot as line
    ax.plot(x, y, '-', color=color, linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_degree_distribution_comparison(G_original, G_config, network_type):
    """Plot comparison of degree distributions between original and configuration model.
    
    Args:
        G_original: Original NetworkX graph
        G_config: Configuration model NetworkX graph
        network_type: Type of network for naming
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original network degree distribution
    plot_degree_distribution(G_original, f"Original {network_type.upper()} Network", 'blue', ax)
    
    # Plot configuration model degree distribution
    plot_degree_distribution(G_config, f"Upscaled Configuration Model (3500 nodes)", 'green', ax)
    
    # Add legend
    ax.legend([f"Original Network ({G_original.number_of_nodes()} nodes)", 
              f"Upscaled Config Model ({G_config.number_of_nodes()} nodes)"])
    
    # Set title
    ax.set_title(f"{network_type.upper()} Network vs. Upscaled Configuration Model: Degree Distribution")
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, f"{network_type}_upscaled_degree_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved degree distribution comparison to {output_path}")
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
    """Process a single network to create its upscaled configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
    
    Returns:
        NetworkX DiGraph of the upscaled configuration model
    """
    print(f"\nProcessing {network_type} network for upscaling to {TARGET_NODES} nodes")
    
    # Load original network
    G_original = load_network(network_type)
    print(f"Original network has {G_original.number_of_nodes()} nodes and {G_original.number_of_edges()} edges")
    
    # Get upscaled degree sequences
    in_sequence, out_sequence = get_upscaled_degree_sequence(G_original, TARGET_NODES)
    
    # Create upscaled configuration model
    G_config = create_upscaled_configuration_model(in_sequence, out_sequence, network_type)
    
    if G_config:
        # Calculate metrics for original and upscaled configuration model
        orig_metrics = calculate_network_metrics(G_original)
        config_metrics = calculate_network_metrics(G_config)
        
        print("\nNetwork metrics comparison:")
        print(f"{'Metric':<20} {'Original':<15} {'Upscaled Config':<15}")
        print("-" * 50)
        for metric in orig_metrics:
            print(f"{metric:<20} {orig_metrics[metric]:<15.4f} {config_metrics[metric]:<15.4f}")
        
        # Plot degree distribution comparison
        plot_degree_distribution_comparison(G_original, G_config, network_type)
        
        # Save upscaled configuration model
        save_network(G_config, network_type)
        
        return G_config
    
    return None

def main():
    """Main function to process all networks."""
    print("Generating upscaled configuration models (3500 nodes) for brain networks")
    
    # Process EB network
    process_network('eb')
    
    # Process FB network
    process_network('fb')
    
    # Process MB-KC network
    process_network('mb_kc')
    
    print("\nUpscaled configuration model generation complete")

if __name__ == "__main__":
    main() 