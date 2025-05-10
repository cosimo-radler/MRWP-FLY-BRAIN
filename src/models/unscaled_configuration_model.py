#!/usr/bin/env python3
"""
Unscaled Configuration Model Generation Script for Drosophila Circuit Robustness Analysis

This script creates unscaled configuration models of the ellipsoid-body (EB), 
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks,
preserving both the original node count and degree distributions.
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
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/unscaled")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(UNSCALED_CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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

def get_degree_sequences(G):
    """Extract the degree sequences from a directed graph.
    
    Args:
        G: NetworkX directed graph
    
    Returns:
        Tuple of (in-degree sequence, out-degree sequence)
    """
    # Get original degree sequences
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    # Ensure sum of in-degrees equals sum of out-degrees
    if sum(in_degrees) != sum(out_degrees):
        print(f"Warning: Sum of in-degrees ({sum(in_degrees)}) does not equal sum of out-degrees ({sum(out_degrees)})")
        print("Adjusting degree sequences to balance them")
        
        # Find the difference
        diff = sum(in_degrees) - sum(out_degrees)
        
        # Keep trying to adjust until balanced
        max_attempts = 100
        attempts = 0
        
        while sum(in_degrees) != sum(out_degrees) and attempts < max_attempts:
            attempts += 1
            
            # Adjust the larger sum
            if diff > 0:
                # In-degrees sum is larger, reduce some in-degrees
                for _ in range(min(abs(diff), 10)):  # Adjust at most 10 degrees at a time
                    # Find a non-zero degree to decrement
                    indices = [i for i, d in enumerate(in_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        in_degrees[idx] -= 1
            else:
                # Out-degrees sum is larger, reduce some out-degrees
                for _ in range(min(abs(diff), 10)):  # Adjust at most 10 degrees at a time
                    # Find a non-zero degree to decrement
                    indices = [i for i, d in enumerate(out_degrees) if d > 0]
                    if indices:
                        idx = random.choice(indices)
                        out_degrees[idx] -= 1
            
            # Recalculate the difference
            diff = sum(in_degrees) - sum(out_degrees)
            
            if attempts % 10 == 0:
                print(f"  Attempt {attempts}: diff = {diff} (in: {sum(in_degrees)}, out: {sum(out_degrees)})")
        
        if sum(in_degrees) != sum(out_degrees):
            # If we still can't balance, force it by scaling
            print(f"Could not balance sequences after {max_attempts} attempts. Forcing balance...")
            
            # Force both sequences to have the same total by scaling
            total = min(sum(in_degrees), sum(out_degrees))
            
            # Create new sequences with the same total
            in_degrees = [max(1, int(d * total / sum(in_degrees))) for d in in_degrees]
            out_degrees = [max(1, int(d * total / sum(out_degrees))) for d in out_degrees]
    
    # Verify equal sums
    assert sum(in_degrees) == sum(out_degrees), "Degree sequences must have equal sums"
    print(f"Balanced degree sequences: sum = {sum(in_degrees)}")
    
    return in_degrees, out_degrees

def create_configuration_model(in_sequence, out_sequence, network_type):
    """Create a directed configuration model with the given degree sequences.
    
    Args:
        in_sequence: In-degree sequence
        out_sequence: Out-degree sequence
        network_type: Type of network for naming
    
    Returns:
        NetworkX DiGraph of the configuration model
    """
    print(f"Creating unscaled configuration model for {network_type} network")
    
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
        
        print(f"Unscaled configuration model created: {G_config.number_of_nodes()} nodes, {G_config.number_of_edges()} edges")
        
        # Relabel nodes to integers starting from 0
        G_config = nx.convert_node_labels_to_integers(G_config)
        
        return G_config
    
    except Exception as e:
        print(f"Error creating configuration model: {e}")
        return None

def plot_degree_distribution_comparison(G_original, G_config, network_type):
    """Plot degree distribution comparison between original and unscaled configuration model.
    
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
    ax1.loglog(in_deg_config, in_cnt_config, 'go-', alpha=0.7, label=f"Unscaled Config Model ({G_config.number_of_nodes()} nodes)")
    ax1.set_title(f"{network_type.upper()} Network: In-Degree Distribution")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Original network out-degree
    out_degrees_orig = sorted([d for _, d in G_original.out_degree()], reverse=True)
    out_degree_count_orig = Counter(out_degrees_orig)
    out_deg_orig, out_cnt_orig = zip(*sorted(out_degree_count_orig.items()))
    ax2.loglog(out_deg_orig, out_cnt_orig, 'bo-', alpha=0.7, label=f"Original ({G_original.number_of_nodes()} nodes)")
    
    # Configuration model out-degree
    out_degrees_config = sorted([d for _, d in G_config.out_degree()], reverse=True)
    out_degree_count_config = Counter(out_degrees_config)
    out_deg_config, out_cnt_config = zip(*sorted(out_degree_count_config.items()))
    ax2.loglog(out_deg_config, out_cnt_config, 'go-', alpha=0.7, label=f"Unscaled Config Model ({G_config.number_of_nodes()} nodes)")
    ax2.set_title(f"{network_type.upper()} Network: Out-Degree Distribution")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Original network total degree
    total_degrees_orig = sorted([d for _, d in G_original.degree()], reverse=True)
    ax3.plot(range(len(total_degrees_orig)), total_degrees_orig, 'b-', alpha=0.7, label=f"Original ({G_original.number_of_nodes()} nodes)")
    
    # Configuration model total degree
    total_degrees_config = sorted([d for _, d in G_config.degree()], reverse=True)
    ax3.plot(range(len(total_degrees_config)), total_degrees_config, 'g-', alpha=0.7, label=f"Unscaled Config Model ({G_config.number_of_nodes()} nodes)")
    ax3.set_title(f"{network_type.upper()} Network: Ranked Total Degree")
    ax3.set_xlabel("Rank")
    ax3.set_ylabel("Total Degree")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Normalized degree distributions for better comparison
    total_degree_count_orig = Counter(total_degrees_orig)
    total_degree_count_config = Counter(total_degrees_config)
    
    # Normalize by total count to get relative frequencies
    max_degree = max(max(total_degrees_orig), max(total_degrees_config))
    degrees = list(range(1, max_degree + 1))
    
    norm_orig = [total_degree_count_orig.get(d, 0) / len(total_degrees_orig) for d in degrees]
    norm_config = [total_degree_count_config.get(d, 0) / len(total_degrees_config) for d in degrees]
    
    ax4.plot(degrees, norm_orig, 'b-', alpha=0.7, label=f"Original ({G_original.number_of_nodes()} nodes)")
    ax4.plot(degrees, norm_config, 'g-', alpha=0.7, label=f"Unscaled Config Model ({G_config.number_of_nodes()} nodes)")
    ax4.set_title(f"{network_type.upper()} Network: Normalized Degree Distribution")
    ax4.set_xlabel("Degree")
    ax4.set_ylabel("Relative Frequency")
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_unscaled_config_degree_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_network(G, network_type):
    """Save network to GEXF file.
    
    Args:
        G: NetworkX graph
        network_type: Type of network for naming
    """
    # Save to the unscaled config models directory
    output_path = os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf")
    nx.write_gexf(G, output_path)
    print(f"Saved unscaled configuration model to: {output_path}")

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
    """Process a single network to create its unscaled configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Tuple of (original network, configuration model, metrics)
    """
    print(f"\nProcessing {network_type} network...")
    
    # Load original network
    G_original = load_network(network_type)
    print(f"Original network has {G_original.number_of_nodes()} nodes and {G_original.number_of_edges()} edges")
    
    # Get degree sequences
    in_sequence, out_sequence = get_degree_sequences(G_original)
    
    # Create unscaled configuration model
    G_config = create_configuration_model(in_sequence, out_sequence, network_type)
    
    if G_config is not None:
        # Plot degree distribution comparison
        plot_degree_distribution_comparison(G_original, G_config, network_type)
        
        # Save configuration model
        save_network(G_config, network_type)
        
        # Calculate metrics
        original_metrics = calculate_network_metrics(G_original)
        config_metrics = calculate_network_metrics(G_config)
        
        print(f"\nMetrics comparison for {network_type.upper()} network:")
        print(f"Original: {original_metrics}")
        print(f"Unscaled Configuration Model: {config_metrics}")
        
        return G_original, G_config, {
            'original': original_metrics,
            'config': config_metrics
        }
    else:
        return G_original, None, None

def main():
    """Main function to process selected networks."""
    # Create metrics DataFrame
    metrics_data = []
    
    # Process EB network
    eb_original, eb_config, eb_metrics = process_network('eb')
    if eb_metrics:
        metrics_data.append({'network': 'eb', 'model': 'original', **eb_metrics['original']})
        metrics_data.append({'network': 'eb', 'model': 'unscaled_config', **eb_metrics['config']})
    
    # Process FB network
    fb_original, fb_config, fb_metrics = process_network('fb')
    if fb_metrics:
        metrics_data.append({'network': 'fb', 'model': 'original', **fb_metrics['original']})
        metrics_data.append({'network': 'fb', 'model': 'unscaled_config', **fb_metrics['config']})
    
    # Process MB-KC network
    mb_kc_original, mb_kc_config, mb_kc_metrics = process_network('mb_kc')
    if mb_kc_metrics:
        metrics_data.append({'network': 'mb_kc', 'model': 'original', **mb_kc_metrics['original']})
        metrics_data.append({'network': 'mb_kc', 'model': 'unscaled_config', **mb_kc_metrics['config']})
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(CONFIG_MODEL_DIR, "unscaled_config_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics comparison to: {metrics_path}")
    
    print("\nUnscaled configuration models created successfully!")

if __name__ == "__main__":
    main() 