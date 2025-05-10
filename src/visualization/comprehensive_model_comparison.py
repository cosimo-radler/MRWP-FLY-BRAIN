#!/usr/bin/env python3
"""
Comprehensive Network Model Comparison Visualization

This script creates a multi-panel visualization comparing:
1. Degree distributions of original networks, scaled configuration models (1500 nodes), 
   unscaled configuration models, and upscaled configuration models (3500 nodes) (top row)
2. Percolation results for random edge removal (middle row)
3. Targeted attack results for degree centrality and betweenness centrality (bottom two rows)
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
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "unscaled")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "upscaled")
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

# Model types
MODEL_TYPES = ['original', 'scaled_config', 'unscaled_config', 'upscaled_config']
MODEL_LABELS = {
    'original': 'Original',
    'scaled_config': 'Scaled Config (1500)',
    'unscaled_config': 'Unscaled Config',
    'upscaled_config': 'Upscaled Config (3500)'
}

# Visualization parameters
COLORS = {
    'original': 'blue',
    'scaled_config': 'red',
    'unscaled_config': 'green',
    'upscaled_config': 'purple'
}

LINE_STYLES = {
    'original': '-',
    'scaled_config': '--',
    'unscaled_config': ':',
    'upscaled_config': '-.'
}

MARKERS = {
    'original': 'o',
    'scaled_config': 's',
    'unscaled_config': '^',
    'upscaled_config': 'd'
}

def load_network(network_type, model_type='original'):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'scaled_config', 'unscaled_config', or 'upscaled_config'
        
    Returns:
        NetworkX Graph
    """
    if model_type == 'original':
        file_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    elif model_type == 'scaled_config':
        file_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(UPSCALED_CONFIG_MODEL_DIR, f"{network_type}_upscaled_config_model.gexf")
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
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

def load_percolation_results(network_type, model_type='original'):
    """Load random percolation results.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'scaled_config', 'unscaled_config', or 'upscaled_config'
        
    Returns:
        DataFrame with results
    """
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # For random percolation, use percolation results
    if model_type == 'original':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_percolation_results.csv")
    elif model_type == 'scaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_config_model_percolation_results.csv")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_unscaled_config_model_percolation_results.csv")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_upscaled_config_model_percolation_results.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'removal_fraction' in df.columns:
            df['removal_probability'] = df['removal_fraction']
        if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
            df['mean_lcc_size'] = df['lcc_size']
            
        return df
    except Exception as e:
        print(f"Error loading percolation results {file_path}: {e}")
        return None

def load_attack_results(network_type, attack_strategy, model_type='original'):
    """Load targeted attack results.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        attack_strategy: 'betweenness' or 'degree'
        model_type: 'original', 'scaled_config', 'unscaled_config', or 'upscaled_config'
        
    Returns:
        DataFrame with results
    """
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # For targeted attacks, use attack results
    if model_type == 'original':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_{attack_strategy}_attack_results.csv")
    elif model_type == 'scaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_config_{attack_strategy}_attack_results.csv")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_unscaled_config_{attack_strategy}_attack_results.csv")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_upscaled_config_model_{attack_strategy}_attack_results.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'removal_fraction' in df.columns:
            df['removal_probability'] = df['removal_fraction']
        if 'fraction_removed' in df.columns:
            df['removal_probability'] = df['fraction_removed']
        if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
            df['mean_lcc_size'] = df['lcc_size']
            
        return df
    except Exception as e:
        print(f"Error loading attack results {file_path}: {e}")
        return None

def plot_degree_distribution(ax, network_type, title):
    """Plot degree distribution comparison for all model types.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        title: Title for the plot
    """
    # Load networks for all model types
    networks = {}
    degree_data = {}
    
    for model_type in MODEL_TYPES:
        # Load network
        G = load_network(network_type, model_type=model_type)
        networks[model_type] = G
        
        # Get degree distribution
        degrees, freq = get_normalized_degree_distribution(G)
        degree_data[model_type] = {'degrees': degrees, 'freq': freq}
    
    # Print network information
    print(f"\n{title} Degree Distribution Analysis:")
    for model_type, G in networks.items():
        if G:
            print(f"{MODEL_LABELS[model_type]} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Plot each model type
    for model_type in MODEL_TYPES:
        if model_type in degree_data and degree_data[model_type]['degrees']:
            degrees = degree_data[model_type]['degrees']
            freq = degree_data[model_type]['freq']
            
            ax.plot(
                degrees, freq,
                color=COLORS[model_type],
                linestyle=LINE_STYLES[model_type],
                marker=MARKERS[model_type],
                linewidth=1.5,
                markersize=5,
                alpha=0.8,
                label=MODEL_LABELS[model_type]
            )
    
    # Set log scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and formatting
    ax.set_title(title)
    ax.set_xlabel('Degree (log scale)')
    ax.set_ylabel('Normalized Frequency (log scale)\n[P(k) = fraction of nodes with degree k]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return degree_data

def plot_attack_comparison(ax, network_type, attack_strategy, title):
    """Plot targeted attack comparison for all model types.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        attack_strategy: 'betweenness' or 'degree'
        title: Title for the plot
    """
    marker_size = 6
    marker_every = 5
    
    # Plot each model type
    for model_type in MODEL_TYPES:
        # Load attack results
        df = load_attack_results(network_type, attack_strategy, model_type=model_type)
        
        if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
            ax.plot(
                df['removal_probability'],
                df['mean_lcc_size'],
                color=COLORS[model_type],
                linestyle=LINE_STYLES[model_type],
                linewidth=2,
                marker=MARKERS[model_type],
                markersize=marker_size,
                markevery=marker_every,
                label=MODEL_LABELS[model_type]
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

def plot_percolation_comparison(ax, network_type, title):
    """Plot random percolation comparison for all model types.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        title: Title for the plot
    """
    marker_size = 6
    marker_every = 5
    
    # Plot each model type
    for model_type in MODEL_TYPES:
        # Load percolation results
        df = load_percolation_results(network_type, model_type=model_type)
        
        if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
            ax.plot(
                df['removal_probability'],
                df['mean_lcc_size'],
                color=COLORS[model_type],
                linestyle=LINE_STYLES[model_type],
                linewidth=2,
                marker=MARKERS[model_type],
                markersize=marker_size,
                markevery=marker_every,
                label=MODEL_LABELS[model_type]
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

def create_comprehensive_visualization():
    """Create the comprehensive multipanel visualization."""
    # Create figure with GridSpec for more control - 4 rows, 3 columns
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create the panels - 4 rows, 3 columns
    # Row 1: Degree distributions
    ax_deg_eb = fig.add_subplot(gs[0, 0])  # EB degree distribution
    ax_deg_fb = fig.add_subplot(gs[0, 1])  # FB degree distribution
    ax_deg_mb = fig.add_subplot(gs[0, 2])  # MB degree distribution
    
    # Row 2: Random percolation
    ax_rand_eb = fig.add_subplot(gs[1, 0])  # EB random percolation
    ax_rand_fb = fig.add_subplot(gs[1, 1])  # FB random percolation
    ax_rand_mb = fig.add_subplot(gs[1, 2])  # MB random percolation
    
    # Row 3: Degree centrality attack
    ax_deg_att_eb = fig.add_subplot(gs[2, 0])  # EB degree attack
    ax_deg_att_fb = fig.add_subplot(gs[2, 1])  # FB degree attack
    ax_deg_att_mb = fig.add_subplot(gs[2, 2])  # MB degree attack
    
    # Row 4: Betweenness centrality attack
    ax_bet_att_eb = fig.add_subplot(gs[3, 0])  # EB betweenness attack
    ax_bet_att_fb = fig.add_subplot(gs[3, 1])  # FB betweenness attack
    ax_bet_att_mb = fig.add_subplot(gs[3, 2])  # MB betweenness attack
    
    # Plot degree distributions (Row 1)
    degree_data = {}
    degree_data['eb'] = plot_degree_distribution(ax_deg_eb, 'eb', f"{NETWORKS['eb']} Degree Distribution")
    degree_data['fb'] = plot_degree_distribution(ax_deg_fb, 'fb', f"{NETWORKS['fb']} Degree Distribution")
    degree_data['mb_kc'] = plot_degree_distribution(ax_deg_mb, 'mb_kc', f"{NETWORKS['mb_kc']} Degree Distribution")
    
    # Plot random percolation results (Row 2)
    plot_percolation_comparison(ax_rand_eb, 'eb', f"{NETWORKS['eb']} Random Percolation")
    plot_percolation_comparison(ax_rand_fb, 'fb', f"{NETWORKS['fb']} Random Percolation")
    plot_percolation_comparison(ax_rand_mb, 'mb_kc', f"{NETWORKS['mb_kc']} Random Percolation")
    
    # Plot degree centrality attack results (Row 3)
    plot_attack_comparison(ax_deg_att_eb, 'eb', 'degree', f"{NETWORKS['eb']} Degree Centrality Attack")
    plot_attack_comparison(ax_deg_att_fb, 'fb', 'degree', f"{NETWORKS['fb']} Degree Centrality Attack")
    plot_attack_comparison(ax_deg_att_mb, 'mb_kc', 'degree', f"{NETWORKS['mb_kc']} Degree Centrality Attack")
    
    # Plot betweenness centrality attack results (Row 4)
    plot_attack_comparison(ax_bet_att_eb, 'eb', 'betweenness', f"{NETWORKS['eb']} Betweenness Centrality Attack")
    plot_attack_comparison(ax_bet_att_fb, 'fb', 'betweenness', f"{NETWORKS['fb']} Betweenness Centrality Attack")
    plot_attack_comparison(ax_bet_att_mb, 'mb_kc', 'betweenness', f"{NETWORKS['mb_kc']} Betweenness Centrality Attack")
    
    # Add main title
    plt.suptitle('Comprehensive Network Model Comparison: Structure and Robustness Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add descriptive subtitles for rows
    y_positions = [0.76, 0.51, 0.26, 0.01]
    row_titles = [
        'Structural Analysis: Normalized Degree Distributions [P(k)]',
        'Random Percolation Analysis: Network Robustness against Random Failures',
        'Degree Centrality Attack Analysis: Network Robustness against Targeted Attacks',
        'Betweenness Centrality Attack Analysis: Network Robustness against Targeted Attacks'
    ]
    
    for y, title in zip(y_positions, row_titles):
        fig.text(0.5, y, title, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Save the figure
    output_path = os.path.join(MULTIPANEL_DIR, 'comprehensive_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nComprehensive visualization saved to {output_path}")
    
    # Show the plot
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()

def main():
    """Main function to run the script."""
    print("Creating comprehensive network model comparison visualization...")
    create_comprehensive_visualization()
    print("Done!")

if __name__ == "__main__":
    main() 