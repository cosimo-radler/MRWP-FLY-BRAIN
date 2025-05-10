#!/usr/bin/env python3
"""
Network Metrics Comparison Script

This script calculates and compares network metrics including clustering coefficient, 
average path length, and characteristic path length between the original and 
configuration model networks for Drosophila brain regions.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_network(network_type, model_type="original"):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, 'mb_kc' for Mushroom Body Kenyon Cells
        model_type: 'original' or 'config' for configuration model
        
    Returns:
        NetworkX DiGraph
    """
    if model_type == "original":
        file_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    else:
        file_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    
    return nx.read_gexf(file_path)

def calculate_network_metrics(G, network_type, model_type):
    """Calculate comprehensive network metrics.
    
    Args:
        G: NetworkX graph
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        model_type: 'original' or 'config'
        
    Returns:
        Dictionary of metrics
    """
    print(f"Calculating metrics for {network_type} {model_type} network...")
    
    # Convert to undirected for some metrics
    G_undirected = G.to_undirected()
    
    metrics = {
        "network_type": network_type,
        "model_type": model_type,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G_undirected),
        "transitivity": nx.transitivity(G_undirected),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
    }
    
    # Try to calculate metrics that might fail on disconnected graphs
    try:
        # Find largest weakly connected component
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        largest_subgraph_directed = G.subgraph(largest_wcc)
        
        # Find largest strongly connected component
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        largest_strongly_connected = G.subgraph(largest_scc)
        
        # Undirected version of largest component
        largest_subgraph = G_undirected.subgraph(largest_wcc)
        
        # Metrics on largest weakly connected component
        metrics["largest_wcc_size"] = len(largest_wcc)
        metrics["largest_wcc_fraction"] = len(largest_wcc) / G.number_of_nodes()
        
        # Metrics on largest strongly connected component
        metrics["largest_scc_size"] = len(largest_scc)
        metrics["largest_scc_fraction"] = len(largest_scc) / G.number_of_nodes()
        
        # Path length metrics (on largest weakly connected component)
        avg_shortest_path = nx.average_shortest_path_length(largest_subgraph)
        metrics["avg_shortest_path"] = avg_shortest_path
        
        # Diameter (on largest weakly connected component)
        diameter = nx.diameter(largest_subgraph)
        metrics["diameter"] = diameter
        
        # Characteristic path length (equal to average shortest path length in connected graphs)
        metrics["characteristic_path_length"] = avg_shortest_path
        
    except (nx.NetworkXError, nx.NetworkXNotImplemented) as e:
        print(f"Error calculating some metrics: {e}")
        # If the graph is disconnected or has other issues
        metrics["avg_shortest_path"] = float('nan')
        metrics["diameter"] = float('nan')
        metrics["characteristic_path_length"] = float('nan')
    
    # Small-world coefficient (if possible)
    try:
        # Calculate small-world coefficient sigma
        # Sigma > 1 indicates small-world properties
        random_G = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=42)
        random_G_clustering = nx.average_clustering(random_G)
        
        if random_G_clustering > 0 and "avg_shortest_path" in metrics and not np.isnan(metrics["avg_shortest_path"]):
            random_path_length = nx.average_shortest_path_length(random_G)
            sigma = (metrics["avg_clustering"] / random_G_clustering) / (metrics["avg_shortest_path"] / random_path_length)
            metrics["small_world_coefficient"] = sigma
    except Exception as e:
        print(f"Could not calculate small-world coefficient: {e}")
        metrics["small_world_coefficient"] = float('nan')
    
    return metrics

def compare_metrics_for_network(network_type):
    """Compare metrics between original and configuration model for a specific network.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
    
    Returns:
        Tuple of (original_metrics, config_metrics)
    """
    # Load networks
    G_original = load_network(network_type, "original")
    G_config = load_network(network_type, "config")
    
    # Calculate metrics
    original_metrics = calculate_network_metrics(G_original, network_type, "original")
    config_metrics = calculate_network_metrics(G_config, network_type, "config")
    
    # Print comparison
    print(f"\n{network_type.upper()} Network Metrics Comparison:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Original':<15} {'Config Model':<15} {'Ratio (Orig/Config)':<20}")
    print("-" * 80)
    
    for metric in original_metrics.keys():
        if metric not in ["network_type", "model_type"]:
            orig_val = original_metrics[metric]
            conf_val = config_metrics[metric]
            
            if isinstance(orig_val, (int, float)) and isinstance(conf_val, (int, float)):
                if conf_val != 0:
                    ratio = orig_val / conf_val
                else:
                    ratio = "N/A"
                
                if isinstance(ratio, (int, float)):
                    print(f"{metric:<30} {orig_val:<15.4f} {conf_val:<15.4f} {ratio:<20.4f}")
                else:
                    print(f"{metric:<30} {orig_val:<15.4f} {conf_val:<15.4f} {ratio:<20}")
            else:
                print(f"{metric:<30} {orig_val:<15} {conf_val:<15}")
    
    return original_metrics, config_metrics

def create_comparison_plots(metrics_df):
    """Create comparison plots for network metrics.
    
    Args:
        metrics_df: DataFrame with metrics for all networks
    """
    # Set up the figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Common plotting parameters
    bar_width = 0.35
    opacity = 0.8
    
    # Define metrics to plot
    metrics_to_plot = [
        ("avg_clustering", "Clustering Coefficient"),
        ("avg_shortest_path", "Average Path Length"),
        ("characteristic_path_length", "Characteristic Path Length")
    ]
    
    # Get unique network types
    network_types = metrics_df["network_type"].unique()
    
    # Set positions for bars
    index = np.arange(len(network_types))
    
    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Extract data for original and config models
        orig_values = []
        config_values = []
        
        for network in network_types:
            orig_val = metrics_df[(metrics_df["network_type"] == network) & 
                                  (metrics_df["model_type"] == "original")][metric].values[0]
            config_val = metrics_df[(metrics_df["network_type"] == network) & 
                                    (metrics_df["model_type"] == "config")][metric].values[0]
            
            orig_values.append(orig_val)
            config_values.append(config_val)
        
        # Create bars
        rects1 = ax.bar(index - bar_width/2, orig_values, bar_width,
                        alpha=opacity, color='b', label='Original')
        rects2 = ax.bar(index + bar_width/2, config_values, bar_width,
                        alpha=opacity, color='g', label='Configuration Model')
        
        # Labels and formatting
        ax.set_xlabel('Network')
        ax.set_ylabel(title)
        ax.set_title(f'Comparison of {title}')
        ax.set_xticks(index)
        ax.set_xticklabels([nt.upper() for nt in network_types])
        ax.legend()
        
        # Add value labels on bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "network_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot to {os.path.join(FIGURES_DIR, 'network_metrics_comparison.png')}")
    
    # Also create a heatmap of ratios
    create_ratio_heatmap(metrics_df)

def create_ratio_heatmap(metrics_df):
    """Create a heatmap showing the ratios of metrics between original and configuration models.
    
    Args:
        metrics_df: DataFrame with metrics for all networks
    """
    # Define metrics to include in heatmap
    metrics_to_include = [
        "avg_clustering", 
        "avg_shortest_path", 
        "characteristic_path_length", 
        "transitivity",
        "density",
        "diameter"
    ]
    
    # Initialize data for heatmap
    network_types = metrics_df["network_type"].unique()
    ratio_data = {}
    
    # Calculate ratios
    for network in network_types:
        ratio_data[network] = {}
        
        for metric in metrics_to_include:
            orig_val = metrics_df[(metrics_df["network_type"] == network) & 
                                 (metrics_df["model_type"] == "original")][metric].values[0]
            config_val = metrics_df[(metrics_df["network_type"] == network) & 
                                   (metrics_df["model_type"] == "config")][metric].values[0]
            
            if not np.isnan(orig_val) and not np.isnan(config_val) and config_val != 0:
                ratio = orig_val / config_val
                ratio_data[network][metric] = ratio
            else:
                ratio_data[network][metric] = np.nan
    
    # Convert to DataFrame for heatmap
    ratio_df = pd.DataFrame(ratio_data).T
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ratio_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title("Ratio of Original/Config Model Metrics")
    plt.ylabel("Network")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "metrics_ratio_heatmap.png"), dpi=300, bbox_inches='tight')
    print(f"Saved metrics ratio heatmap to {os.path.join(FIGURES_DIR, 'metrics_ratio_heatmap.png')}")

def main():
    """Main function to run network metrics comparison."""
    network_types = ["eb", "fb", "mb_kc"]
    all_metrics = []
    
    for network_type in network_types:
        orig_metrics, config_metrics = compare_metrics_for_network(network_type)
        all_metrics.append(orig_metrics)
        all_metrics.append(config_metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    csv_path = os.path.join(RESULTS_DIR, "network_metrics_comparison.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved all metrics to {csv_path}")
    
    # Create comparison plots
    create_comparison_plots(metrics_df)

if __name__ == "__main__":
    main() 