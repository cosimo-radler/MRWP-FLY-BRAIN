#!/usr/bin/env python3
"""
Network Visualization Script for Drosophila Circuit Robustness Analysis

This script creates visualizations for the ellipsoid-body (EB), fan-shaped-body (FB), and 
mushroom-body (MB) Kenyon-cell subnetworks to help understand their structure.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_network(network_type):
    """Load network from files.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    try:
        # Try to load from GEXF (if the full processing pipeline was run)
        return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))
    except FileNotFoundError:
        # If GEXF doesn't exist, try to build from connectivity CSV
        try:
            connectivity_df = pd.read_csv(os.path.join(DATA_DIR, f"{network_type}_connectivity.csv"))
            G = nx.DiGraph()
            for _, row in connectivity_df.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            return G
        except FileNotFoundError:
            # If both methods fail, create a small example network for testing
            print(f"Could not load {network_type} network from files, creating example network.")
            G = nx.scale_free_graph(n=100, seed=42)
            return G

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
        "avg_shortest_path": -1,  # Will be calculated if possible
        "diameter": -1,  # Will be calculated if possible
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "in_degree_centralization": -1,
        "out_degree_centralization": -1,
    }
    
    # Try to calculate metrics that might fail on disconnected graphs
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest_cc)
    
    try:
        # Average shortest path and diameter on largest connected component
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(largest_subgraph)
        metrics["diameter"] = nx.diameter(largest_subgraph)
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        # If the graph is disconnected or has other issues
        pass
    
    # Calculate degree centralization
    in_degrees = [deg for node, deg in G.in_degree()]
    out_degrees = [deg for node, deg in G.out_degree()]
    
    if in_degrees:
        max_in = max(in_degrees)
        sum_in_diff = sum(max_in - d for d in in_degrees)
        n = G.number_of_nodes()
        metrics["in_degree_centralization"] = sum_in_diff / ((n-1)*(n-2))
    
    if out_degrees:
        max_out = max(out_degrees)
        sum_out_diff = sum(max_out - d for d in out_degrees)
        n = G.number_of_nodes()
        metrics["out_degree_centralization"] = sum_out_diff / ((n-1)*(n-2))
    
    return metrics

def plot_degree_distribution(G, network_name, fig_path):
    """Plot degree distribution.
    
    Args:
        G: NetworkX graph
        network_name: Name for the plot title
        fig_path: Where to save the figure
    """
    in_degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)
    out_degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # In-degree
    in_degree_count = Counter(in_degree_sequence)
    in_deg, in_cnt = zip(*sorted(in_degree_count.items()))
    ax1.loglog(in_deg, in_cnt, 'bo-', alpha=0.7)
    ax1.set_title(f"{network_name} In-Degree Distribution")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # Out-degree
    out_degree_count = Counter(out_degree_sequence)
    out_deg, out_cnt = zip(*sorted(out_degree_count.items()))
    ax2.loglog(out_deg, out_cnt, 'ro-', alpha=0.7)
    ax2.set_title(f"{network_name} Out-Degree Distribution")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_visualization(G, network_name, fig_path, max_nodes=None):
    """Create a network visualization.
    
    Args:
        G: NetworkX graph
        network_name: Name for the plot title
        fig_path: Where to save the figure
        max_nodes: Maximum number of nodes to show (for readability), None for all nodes
    """
    # If network is too large and max_nodes is specified, take a subgraph of the most connected nodes
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        # Get nodes with highest total degree
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)
        subgraph_note = f"(showing top {max_nodes} most connected nodes out of {G.number_of_nodes()} total)"
    else:
        subgraph_note = f"(showing all {G.number_of_nodes()} nodes)"
    
    # For very large networks, adjust the figure size
    if G.number_of_nodes() > 1000:
        plt.figure(figsize=(20, 20))
    else:
        plt.figure(figsize=(12, 12))
    
    # Use different layout algorithms based on network size
    if G.number_of_nodes() < 100:
        pos = nx.spring_layout(G, seed=42)
    elif G.number_of_nodes() < 500:
        pos = nx.kamada_kawai_layout(G)
    else:
        # For very large networks, use a faster layout algorithm
        pos = nx.random_layout(G, seed=42)
    
    # Scale node sizes based on network size
    if G.number_of_nodes() > 1000:
        base_size = 1
        scale = 0.5
    elif G.number_of_nodes() > 500:
        base_size = 3
        scale = 0.8
    else:
        base_size = 10
        scale = 2
    
    # Node sizes based on degree
    node_size = [base_size + scale * G.degree(n) for n in G.nodes()]
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.2, arrows=False)
    
    plt.title(f"{network_name} Network Visualization\n{subgraph_note}, {G.number_of_edges()} edges")
    plt.axis('off')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_comparison(eb_metrics, fb_metrics, mb_kc_metrics, fig_path):
    """Plot side-by-side metrics comparison for all three networks.
    
    Args:
        eb_metrics: Dictionary of metrics for EB network
        fb_metrics: Dictionary of metrics for FB network
        mb_kc_metrics: Dictionary of metrics for MB-KC network
        fig_path: Where to save the figure
    """
    metrics = ['n_nodes', 'n_edges', 'density', 'avg_clustering', 
               'avg_shortest_path', 'diameter', 'avg_degree',
               'in_degree_centralization', 'out_degree_centralization']
    
    # Filter valid metrics (value > 0)
    valid_metrics = []
    eb_values = []
    fb_values = []
    mb_kc_values = []
    
    for metric in metrics:
        if eb_metrics[metric] > 0 and fb_metrics[metric] > 0 and mb_kc_metrics[metric] > 0:
            valid_metrics.append(metric)
            eb_values.append(eb_metrics[metric])
            fb_values.append(fb_metrics[metric])
            mb_kc_values.append(mb_kc_metrics[metric])
    
    # Create a figure with a logarithmic scale for certain metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(valid_metrics))
    width = 0.25
    
    # Plot bars
    rects1 = ax.bar(x - width, eb_values, width, label='EB Network', color='blue', alpha=0.7)
    rects2 = ax.bar(x, fb_values, width, label='FB Network', color='green', alpha=0.7)
    rects3 = ax.bar(x + width, mb_kc_values, width, label='MB-KC Network', color='red', alpha=0.7)
    
    # Add labels
    ax.set_ylabel('Value')
    ax.set_title('Network Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics], rotation=45, ha='right')
    ax.legend()
    
    # Label bars with values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 1000:
                text = f'{height:.0f}'
            elif height > 100:
                text = f'{height:.1f}'
            elif height > 1:
                text = f'{height:.2f}'
            else:
                text = f'{height:.3f}'
            ax.annotate(text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot for large-value metrics on a log scale
    large_metrics = ['n_nodes', 'n_edges']
    large_metric_indices = [valid_metrics.index(m) for m in large_metrics if m in valid_metrics]
    
    if large_metric_indices:
        large_eb_values = [eb_values[i] for i in large_metric_indices]
        large_fb_values = [fb_values[i] for i in large_metric_indices]
        large_mb_kc_values = [mb_kc_values[i] for i in large_metric_indices]
        large_metric_names = [valid_metrics[i] for i in large_metric_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(large_metric_names))
        
        rects1 = ax.bar(x - width, large_eb_values, width, label='EB Network', color='blue', alpha=0.7)
        rects2 = ax.bar(x, large_fb_values, width, label='FB Network', color='green', alpha=0.7)
        rects3 = ax.bar(x + width, large_mb_kc_values, width, label='MB-KC Network', color='red', alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_ylabel('Value (log scale)')
        ax.set_title('Network Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in large_metric_names])
        ax.legend()
        
        # Label bars with values
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(fig_path), 'network_size_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    print("Visualizing neural networks...")
    
    # Load networks
    print("\nLoading Ellipsoid Body (EB) network...")
    eb_network = load_network('eb')
    print(f"EB Network: {eb_network.number_of_nodes()} nodes, {eb_network.number_of_edges()} edges")
    
    print("\nLoading Fan-shaped Body (FB) network...")
    fb_network = load_network('fb')
    print(f"FB Network: {fb_network.number_of_nodes()} nodes, {fb_network.number_of_edges()} edges")
    
    print("\nLoading Mushroom Body Kenyon Cell (MB-KC) network...")
    mb_kc_network = load_network('mb_kc')
    print(f"MB-KC Network: {mb_kc_network.number_of_nodes()} nodes, {mb_kc_network.number_of_edges()} edges")
    
    # Calculate network metrics
    print("\nCalculating network metrics...")
    eb_metrics = calculate_network_metrics(eb_network)
    fb_metrics = calculate_network_metrics(fb_network)
    mb_kc_metrics = calculate_network_metrics(mb_kc_network)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Degree distributions
    plot_degree_distribution(
        eb_network, 
        "Ellipsoid Body", 
        os.path.join(FIGURES_DIR, "eb_degree_distribution.png")
    )
    
    plot_degree_distribution(
        fb_network, 
        "Fan-shaped Body", 
        os.path.join(FIGURES_DIR, "fb_degree_distribution.png")
    )
    
    plot_degree_distribution(
        mb_kc_network, 
        "Mushroom Body Kenyon Cells", 
        os.path.join(FIGURES_DIR, "mb_kc_degree_distribution.png")
    )
    
    # Network visualizations
    plot_network_visualization(
        eb_network, 
        "Ellipsoid Body", 
        os.path.join(FIGURES_DIR, "eb_network_visualization.png")
    )
    
    plot_network_visualization(
        fb_network, 
        "Fan-shaped Body", 
        os.path.join(FIGURES_DIR, "fb_network_visualization.png")
    )
    
    plot_network_visualization(
        mb_kc_network, 
        "Mushroom Body Kenyon Cells", 
        os.path.join(FIGURES_DIR, "mb_kc_network_visualization.png")
    )
    
    # Metrics comparison
    plot_network_comparison(
        eb_metrics,
        fb_metrics,
        mb_kc_metrics, 
        os.path.join(FIGURES_DIR, "network_metrics_comparison.png")
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': list(eb_metrics.keys()),
        'EB Network': list(eb_metrics.values()),
        'FB Network': list(fb_metrics.values()),
        'MB-KC Network': list(mb_kc_metrics.values())
    })
    metrics_df.to_csv(os.path.join(FIGURES_DIR, "network_metrics.csv"), index=False)
    
    print("\nNetwork visualization complete!")
    print(f"Visualizations saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 