#!/usr/bin/env python3
"""
Multi-panel Network Comparison Visualization

This script creates a 6-panel visualization comparing:
1. Degree distributions of the original networks and their configuration models (top row)
2. Percolation results for the original networks and their configuration models (bottom row)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.gridspec import GridSpec
from collections import Counter

# Set the style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
MULTIPANEL_DIR = os.path.join(FIGURES_DIR, "multipanel")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MULTIPANEL_DIR, exist_ok=True)

# Network types and their display names
NETWORKS = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body KC'
}

# Attack strategies
ATTACK_STRATEGIES = ['random', 'betweenness', 'degree']
STRATEGY_NAMES = {
    'random': 'Random Percolation',
    'betweenness': 'Betweenness Centrality',
    'degree': 'Degree Centrality'
}

def load_network(network_type, is_config=False):
    """Load original or configuration model network from GEXF file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        is_config: Whether to load the configuration model
        
    Returns:
        NetworkX Graph
    """
    if is_config:
        file_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    else:
        file_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    
    try:
        G = nx.read_gexf(file_path)
        return G
    except Exception as e:
        print(f"Error loading network {file_path}: {e}")
        return None

def get_normalized_degree_distribution(G):
    """Get normalized degree distribution of a graph.
    
    Args:
        G: NetworkX Graph
        
    Returns:
        degrees: Sorted unique degrees
        frequencies: Normalized frequencies
    """
    if G is None:
        return [], []
        
    # Convert to undirected for consistent degree calculation
    G_undirected = G.to_undirected()
    
    # Get degrees and their frequencies
    degrees = [d for n, d in G_undirected.degree()]
    degree_count = Counter(degrees)
    
    # Sort by degree
    sorted_degrees = sorted(degree_count.keys())
    frequencies = [degree_count[d] / G_undirected.number_of_nodes() for d in sorted_degrees]
    
    return sorted_degrees, frequencies

def load_percolation_results(network_type, attack_strategy, is_config=False):
    """Load percolation or targeted attack results.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        attack_strategy: 'random', 'betweenness', or 'degree'
        is_config: Whether to load results for config model
        
    Returns:
        DataFrame with results
    """
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    if attack_strategy == 'random':
        # For random percolation, use percolation results
        config_path = f"{network_type}_config_model_percolation_results.csv" if is_config else f"{full_name}_percolation_results.csv"
    else:
        # For targeted attacks, use attack results
        config_prefix = "config_" if is_config else ""
        config_path = f"{full_name}_{config_prefix}{attack_strategy}_attack_results.csv"
    
    file_path = os.path.join(RESULTS_DIR, config_path)
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'removal_fraction' in df.columns:
            df['removal_probability'] = df['removal_fraction']
        if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
            df['mean_lcc_size'] = df['lcc_size']
            
        return df
    except Exception as e:
        print(f"Error loading results {file_path}: {e}")
        return None

def plot_degree_distribution(ax, network_type, title):
    """Plot degree distribution comparison for original and config model.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        title: Title for the plot
    """
    # Load networks
    G_original = load_network(network_type, is_config=False)
    G_config = load_network(network_type, is_config=True)
    
    # Get degree distributions
    orig_degrees, orig_freq = get_normalized_degree_distribution(G_original)
    config_degrees, config_freq = get_normalized_degree_distribution(G_config)
    
    # Print information about the degree distributions
    print(f"\n{title} Degree Distribution Analysis:")
    print(f"Original network: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
    print(f"Config model: {G_config.number_of_nodes()} nodes, {G_config.number_of_edges()} edges")
    print(f"Number of unique degrees - Original: {len(orig_degrees)}, Config: {len(config_degrees)}")
    
    # Use scatter plot with connecting lines instead of stem plots
    ax.plot(orig_degrees, orig_freq, 'bo-', linewidth=1.5, markersize=5, alpha=0.8, label='Original')
    ax.plot(config_degrees, config_freq, 'ro-', linewidth=1.5, markersize=5, alpha=0.8, label='Config Model')
    
    # Set log scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and formatting
    ax.set_title(title)
    ax.set_xlabel('Degree (log scale)')
    ax.set_ylabel('Normalized Frequency (log scale)\n[P(k) = fraction of nodes with degree k]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Return degree sequences for analysis
    return {
        'original_degrees': orig_degrees,
        'original_freq': orig_freq,
        'config_degrees': config_degrees,
        'config_freq': config_freq
    }

def plot_percolation_comparison(ax, network_type, title):
    """Plot percolation comparison for original and config models with all attack strategies.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        title: Title for the plot
    """
    # Colors for different attack strategies
    colors = {
        'random': 'green',
        'betweenness': 'blue',
        'degree': 'red'
    }
    
    line_styles = {
        'original': '-',
        'config': '--'
    }
    
    markers = {
        'original': 'o',
        'config': 's'
    }
    
    marker_size = 6
    marker_every = 7
    
    # Plot each attack type
    for attack in ATTACK_STRATEGIES:
        # Load original results
        orig_df = load_percolation_results(network_type, attack, is_config=False)
        if orig_df is not None and 'removal_probability' in orig_df.columns and 'mean_lcc_size' in orig_df.columns:
            ax.plot(
                orig_df['removal_probability'],
                orig_df['mean_lcc_size'],
                color=colors[attack],
                linestyle=line_styles['original'],
                linewidth=2,
                marker=markers['original'],
                markersize=marker_size,
                markevery=marker_every,
                label=f"{STRATEGY_NAMES[attack]} (Original)"
            )
        
        # Load config model results
        config_df = load_percolation_results(network_type, attack, is_config=True)
        if config_df is not None and 'removal_probability' in config_df.columns and 'mean_lcc_size' in config_df.columns:
            ax.plot(
                config_df['removal_probability'],
                config_df['mean_lcc_size'],
                color=colors[attack],
                linestyle=line_styles['config'],
                linewidth=2,
                marker=markers['config'],
                markersize=marker_size,
                markevery=marker_every,
                label=f"{STRATEGY_NAMES[attack]} (Config)"
            )
    
    # Add threshold line
    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.0)
    ax.text(0.05, 0.07, 'Threshold (5%)', fontsize=8, color='gray')
    
    # Format plot
    ax.set_title(title)
    ax.set_xlabel('Edge Removal Probability')
    ax.set_ylabel('Largest Connected Component')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def create_multipanel_visualization():
    """Create the 6-panel visualization."""
    # Create figure with GridSpec for more control
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)
    
    # Create the six panels
    ax_deg_eb = fig.add_subplot(gs[0, 0])  # EB degree distribution
    ax_deg_fb = fig.add_subplot(gs[0, 1])  # FB degree distribution
    ax_deg_mb = fig.add_subplot(gs[0, 2])  # MB degree distribution
    
    ax_perc_eb = fig.add_subplot(gs[1, 0])  # EB percolation results
    ax_perc_fb = fig.add_subplot(gs[1, 1])  # FB percolation results
    ax_perc_mb = fig.add_subplot(gs[1, 2])  # MB percolation results
    
    # Plot degree distributions in top row and collect results
    degree_data = {}
    degree_data['eb'] = plot_degree_distribution(ax_deg_eb, 'eb', f"{NETWORKS['eb']} Degree Distribution")
    degree_data['fb'] = plot_degree_distribution(ax_deg_fb, 'fb', f"{NETWORKS['fb']} Degree Distribution")
    degree_data['mb_kc'] = plot_degree_distribution(ax_deg_mb, 'mb_kc', f"{NETWORKS['mb_kc']} Degree Distribution")
    
    # Plot percolation results in bottom row
    plot_percolation_comparison(ax_perc_eb, 'eb', f"{NETWORKS['eb']} Percolation Analysis")
    plot_percolation_comparison(ax_perc_fb, 'fb', f"{NETWORKS['fb']} Percolation Analysis")
    plot_percolation_comparison(ax_perc_mb, 'mb_kc', f"{NETWORKS['mb_kc']} Percolation Analysis")
    
    # Add main title
    plt.suptitle('Network Structure and Percolation Analysis Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # Add descriptive subtitles for rows
    fig.text(0.5, 0.51, 'Structural Analysis: Normalized Degree Distributions [P(k)]', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.01, 'Functional Analysis: Network Robustness under Different Attack Strategies', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Analyze differences in degree distribution
    print("\n===== Comparison of Degree Distributions =====")
    for network, data in degree_data.items():
        # Get set of degrees in both distributions
        all_degrees = set(data['original_degrees']).union(set(data['config_degrees']))
        
        # Check for degrees found in only one distribution
        only_in_original = set(data['original_degrees']) - set(data['config_degrees'])
        only_in_config = set(data['config_degrees']) - set(data['original_degrees'])
        
        print(f"\n{NETWORKS[network]} unique degree analysis:")
        if only_in_original:
            print(f"Degrees found only in original network: {sorted(only_in_original)}")
        if only_in_config:
            print(f"Degrees found only in config model: {sorted(only_in_config)}")
        
        # Check for differences in frequency for common degrees
        common_degrees = set(data['original_degrees']).intersection(set(data['config_degrees']))
        orig_freq_dict = dict(zip(data['original_degrees'], data['original_freq']))
        config_freq_dict = dict(zip(data['config_degrees'], data['config_freq']))
        
        max_diff = 0
        max_diff_degree = None
        for degree in common_degrees:
            diff = abs(orig_freq_dict[degree] - config_freq_dict[degree])
            if diff > max_diff:
                max_diff = diff
                max_diff_degree = degree
        
        if max_diff_degree is not None:
            print(f"Largest frequency difference: Degree {max_diff_degree}, " +
                  f"Original: {orig_freq_dict[max_diff_degree]:.6f}, " +
                  f"Config: {config_freq_dict[max_diff_degree]:.6f}, " +
                  f"Diff: {max_diff:.6f}")
    
    # Save the figure
    output_path = os.path.join(MULTIPANEL_DIR, 'network_structure_and_percolation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nMultipanel visualization saved to {output_path}")
    
    # Show the plot
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()

def main():
    """Main function to run the script."""
    print("Creating multi-panel network comparison visualization...")
    create_multipanel_visualization()
    print("Done!")

if __name__ == "__main__":
    main() 