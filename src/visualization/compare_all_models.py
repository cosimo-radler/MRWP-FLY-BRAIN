#!/usr/bin/env python3
"""
Comprehensive Model Comparison Visualization Script

This script creates comparative visualizations of original networks, scaled configuration models, 
and unscaled configuration models for the ellipsoid-body (EB), fan-shaped-body (FB), 
and mushroom-body (MB) Kenyon-cell networks.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.gridspec import GridSpec

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/unscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
COMPARISON_FIGURES_DIR = os.path.join(FIGURES_DIR, "model_comparisons")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(COMPARISON_FIGURES_DIR, exist_ok=True)

# Network name mapping
NETWORK_NAMES = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body Kenyon Cells'
}

# File name mapping
FILE_NAMES = {
    'eb': 'ellipsoid_body',
    'fb': 'fan-shaped_body',
    'mb_kc': 'mushroom_body_kenyon_cell'
}

def load_network(network_type, model_type='original'):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'scaled_config', or 'unscaled_config'
        
    Returns:
        NetworkX DiGraph
    """
    if model_type == 'original':
        path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    elif model_type == 'scaled_config':
        path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    elif model_type == 'unscaled_config':
        path = os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return nx.read_gexf(path) if os.path.exists(path) else None

def load_attack_parameters(network_type, model_type='original'):
    """Load attack parameters from JSON file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'config', or 'unscaled_config'
        
    Returns:
        Dictionary of attack parameters
    """
    full_name = FILE_NAMES.get(network_type, network_type)
    
    if model_type == 'original':
        path = os.path.join(RESULTS_DIR, f"{full_name}_attack_parameters.json")
    elif model_type == 'config':
        path = os.path.join(RESULTS_DIR, f"{full_name}_config_attack_parameters.json")
    elif model_type == 'unscaled_config':
        path = os.path.join(RESULTS_DIR, f"{full_name}_unscaled_config_attack_parameters.json")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def load_percolation_results(network_type, model_type='original'):
    """Load percolation results from CSV file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'config_model', or 'unscaled_config_model'
        
    Returns:
        DataFrame with percolation results
    """
    try:
        path = os.path.join(RESULTS_DIR, f"{network_type}_{model_type}_percolation_results.csv")
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def load_attack_results(network_type, model_type, attack_type):
    """Load targeted attack results from CSV file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'config', or 'unscaled_config'
        attack_type: 'degree', 'betweenness', or 'random'
        
    Returns:
        DataFrame with attack results
    """
    full_name = FILE_NAMES.get(network_type, network_type)
    
    if model_type == 'original':
        prefix = f"{full_name}_original"
    elif model_type == 'config':
        prefix = f"{full_name}_config"
    elif model_type == 'unscaled_config':
        prefix = f"{full_name}_unscaled_config"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{attack_type}_attack_results.csv")
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def plot_triple_comparison(network_type):
    """Create combined comparison plots for original, scaled, and unscaled models.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
    """
    # Load networks
    G_original = load_network(network_type, 'original')
    G_scaled = load_network(network_type, 'scaled_config')
    G_unscaled = load_network(network_type, 'unscaled_config')
    
    if G_original is None or G_scaled is None or G_unscaled is None:
        print(f"Warning: Could not load all networks for {network_type}")
        return
    
    # Create figure
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Network Summary Statistics
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate basic metrics
    metrics = {
        'Original': {
            'Nodes': G_original.number_of_nodes(),
            'Edges': G_original.number_of_edges(),
            'Density': nx.density(G_original),
            'Avg Degree': np.mean([d for _, d in G_original.degree()]),
            'Avg Clustering': nx.average_clustering(G_original.to_undirected())
        },
        'Scaled Config': {
            'Nodes': G_scaled.number_of_nodes(),
            'Edges': G_scaled.number_of_edges(),
            'Density': nx.density(G_scaled),
            'Avg Degree': np.mean([d for _, d in G_scaled.degree()]),
            'Avg Clustering': nx.average_clustering(G_scaled.to_undirected())
        },
        'Unscaled Config': {
            'Nodes': G_unscaled.number_of_nodes(),
            'Edges': G_unscaled.number_of_edges(),
            'Density': nx.density(G_unscaled),
            'Avg Degree': np.mean([d for _, d in G_unscaled.degree()]),
            'Avg Clustering': nx.average_clustering(G_unscaled.to_undirected())
        }
    }
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics)
    
    # Create a bar plot of metrics
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title(f"{NETWORK_NAMES.get(network_type, network_type.upper())} Network Comparison - Basic Metrics", fontsize=16)
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=12)
    
    # 2. Degree Distribution Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    
    # In-degree distributions
    in_degrees_orig = sorted([d for _, d in G_original.in_degree()])
    in_degrees_scaled = sorted([d for _, d in G_scaled.in_degree()])
    in_degrees_unscaled = sorted([d for _, d in G_unscaled.in_degree()])
    
    # Create histograms
    max_in_degree = max(max(in_degrees_orig), max(in_degrees_scaled), max(in_degrees_unscaled))
    bins = np.logspace(0, np.log10(max_in_degree + 1), 20)
    
    ax2.hist(in_degrees_orig, bins=bins, alpha=0.7, label="Original")
    ax2.set_xscale('log')
    ax2.set_title("In-Degree Distribution")
    ax2.set_xlabel("In-Degree")
    ax2.set_ylabel("Count")
    ax2.legend()
    
    # Out-degree distributions
    out_degrees_orig = sorted([d for _, d in G_original.out_degree()])
    out_degrees_scaled = sorted([d for _, d in G_scaled.out_degree()])
    out_degrees_unscaled = sorted([d for _, d in G_unscaled.out_degree()])
    
    max_out_degree = max(max(out_degrees_orig), max(out_degrees_scaled), max(out_degrees_unscaled))
    bins = np.logspace(0, np.log10(max_out_degree + 1), 20)
    
    ax3.hist(out_degrees_orig, bins=bins, alpha=0.7, label="Original")
    ax3.set_xscale('log')
    ax3.set_title("Out-Degree Distribution")
    ax3.set_xlabel("Out-Degree")
    ax3.set_ylabel("Count")
    ax3.legend()
    
    # Total degree rank plot
    total_degrees_orig = sorted([d for _, d in G_original.degree()], reverse=True)
    total_degrees_scaled = sorted([d for _, d in G_scaled.degree()], reverse=True)
    total_degrees_unscaled = sorted([d for _, d in G_unscaled.degree()], reverse=True)
    
    # Normalize for comparison
    x_orig = np.linspace(0, 1, len(total_degrees_orig))
    x_scaled = np.linspace(0, 1, len(total_degrees_scaled))
    x_unscaled = np.linspace(0, 1, len(total_degrees_unscaled))
    
    ax4.plot(x_orig, total_degrees_orig, 'b-', alpha=0.7, label="Original")
    ax4.plot(x_scaled, total_degrees_scaled, 'g-', alpha=0.7, label="Scaled Config")
    ax4.plot(x_unscaled, total_degrees_unscaled, 'r-', alpha=0.7, label="Unscaled Config")
    ax4.set_title("Degree Rank Plot (Normalized)")
    ax4.set_xlabel("Normalized Rank")
    ax4.set_ylabel("Degree")
    ax4.legend()
    
    # 3. Robustness Comparison
    # Bond Percolation
    ax5 = fig.add_subplot(gs[2, 0])
    
    percolation_orig = load_percolation_results(network_type, 'original')
    percolation_scaled = load_percolation_results(network_type, 'config_model')
    percolation_unscaled = load_percolation_results(network_type, 'unscaled_config_model')
    
    if percolation_orig is not None and percolation_scaled is not None and percolation_unscaled is not None:
        ax5.plot(percolation_orig['removal_probability'], percolation_orig['mean_lcc_size'], 'b-', alpha=0.7, label="Original")
        ax5.plot(percolation_scaled['removal_probability'], percolation_scaled['mean_lcc_size'], 'g-', alpha=0.7, label="Scaled Config")
        ax5.plot(percolation_unscaled['removal_probability'], percolation_unscaled['mean_lcc_size'], 'r-', alpha=0.7, label="Unscaled Config")
        ax5.set_title("Bond Percolation")
        ax5.set_xlabel("Edge Removal Probability")
        ax5.set_ylabel("Largest Connected Component Size")
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # Degree-based attack
    ax6 = fig.add_subplot(gs[2, 1])
    
    degree_orig = load_attack_results(network_type, 'original', 'degree')
    degree_scaled = load_attack_results(network_type, 'config', 'degree')
    degree_unscaled = load_attack_results(network_type, 'unscaled_config', 'degree')
    
    if degree_orig is not None and degree_scaled is not None and degree_unscaled is not None:
        ax6.plot(degree_orig['removal_fraction'], degree_orig['mean_lcc_size'], 'b-', alpha=0.7, label="Original")
        ax6.plot(degree_scaled['removal_fraction'], degree_scaled['mean_lcc_size'], 'g-', alpha=0.7, label="Scaled Config")
        ax6.plot(degree_unscaled['removal_fraction'], degree_unscaled['mean_lcc_size'], 'r-', alpha=0.7, label="Unscaled Config")
        ax6.set_title("Degree-Based Attack")
        ax6.set_xlabel("Edge Removal Fraction")
        ax6.set_ylabel("Largest Connected Component Size")
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Betweenness-based attack
    ax7 = fig.add_subplot(gs[2, 2])
    
    betw_orig = load_attack_results(network_type, 'original', 'betweenness')
    betw_scaled = load_attack_results(network_type, 'config', 'betweenness')
    betw_unscaled = load_attack_results(network_type, 'unscaled_config', 'betweenness')
    
    if betw_orig is not None and betw_scaled is not None and betw_unscaled is not None:
        ax7.plot(betw_orig['removal_fraction'], betw_orig['mean_lcc_size'], 'b-', alpha=0.7, label="Original")
        ax7.plot(betw_scaled['removal_fraction'], betw_scaled['mean_lcc_size'], 'g-', alpha=0.7, label="Scaled Config")
        ax7.plot(betw_unscaled['removal_fraction'], betw_unscaled['mean_lcc_size'], 'r-', alpha=0.7, label="Unscaled Config")
        ax7.set_title("Betweenness-Based Attack")
        ax7.set_xlabel("Edge Removal Fraction")
        ax7.set_ylabel("Largest Connected Component Size")
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_FIGURES_DIR, f"{network_type}_triple_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparative_bar_plots():
    """Create comparative bar plots for all networks and attack strategies."""
    # Load summary data for original and scaled config models
    try:
        scaled_summary = pd.read_csv(os.path.join(RESULTS_DIR, "percolation_comparison_summary.csv"))
    except FileNotFoundError:
        scaled_summary = None
    
    # Load summary data for unscaled config models
    try:
        unscaled_summary = pd.read_csv(os.path.join(RESULTS_DIR, "unscaled_percolation_comparison_summary.csv"))
    except FileNotFoundError:
        unscaled_summary = None
    
    # Load targeted attack summary for unscaled models
    try:
        attack_summary = pd.read_csv(os.path.join(RESULTS_DIR, "unscaled_config_attack_summary.csv"))
    except FileNotFoundError:
        attack_summary = None
    
    if scaled_summary is not None and unscaled_summary is not None:
        # Create combined DataFrames
        combined_summary = pd.concat([
            scaled_summary[scaled_summary['model_type'] == 'original'], 
            scaled_summary[scaled_summary['model_type'] == 'config_model'].assign(model_type='scaled_config_model'),
            unscaled_summary[unscaled_summary['model_type'] == 'unscaled_config_model']
        ])
        
        # Create comparative bar plots for robustness
        plt.figure(figsize=(12, 6))
        
        networks = combined_summary['network_type'].unique()
        models = ['original', 'scaled_config_model', 'unscaled_config_model']
        colors = ['blue', 'green', 'red']
        
        bar_width = 0.25
        index = np.arange(len(networks))
        
        for i, (model, color) in enumerate(zip(models, colors)):
            model_data = combined_summary[combined_summary['model_type'] == model]
            values = [model_data[model_data['network_type'] == net]['robustness_index'].values[0] 
                    for net in networks]
            plt.bar(index + i*bar_width, values, bar_width, label=model.replace('_model', '').title(), color=color)
        
        plt.xlabel('Network Type')
        plt.ylabel('Robustness Index')
        plt.title('Robustness Index Comparison - Bond Percolation')
        plt.xticks(index + bar_width, [NETWORK_NAMES.get(net, net.upper()) for net in networks])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_FIGURES_DIR, "bond_percolation_robustness_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    if attack_summary is not None:
        # Plot for each attack type
        for attack_type in attack_summary['attack_type'].unique():
            attack_data = attack_summary[attack_summary['attack_type'] == attack_type]
            
            plt.figure(figsize=(12, 6))
            
            networks = attack_data['network_type'].unique()
            models = ['original', 'unscaled_config']
            colors = ['blue', 'red']
            
            bar_width = 0.35
            index = np.arange(len(networks))
            
            for i, (model, color) in enumerate(zip(models, colors)):
                model_data = attack_data[attack_data['model_type'] == model]
                values = [model_data[model_data['network_type'] == net]['robustness_index'].values[0] 
                        for net in networks]
                plt.bar(index + i*bar_width, values, bar_width, label=model.replace('_', ' ').title(), color=color)
            
            plt.xlabel('Network Type')
            plt.ylabel('Robustness Index')
            plt.title(f'Robustness Index Comparison - {attack_type.title()} Attack')
            plt.xticks(index + bar_width/2, [NETWORK_NAMES.get(net, net.upper()) for net in networks])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(COMPARISON_FIGURES_DIR, f"{attack_type}_attack_robustness_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to generate all comparative visualizations."""
    print("Generating comprehensive model comparisons...")
    
    # Create individual network comparisons
    for network_type in ['eb', 'fb', 'mb_kc']:
        print(f"Creating comparison for {NETWORK_NAMES.get(network_type, network_type.upper())}...")
        plot_triple_comparison(network_type)
    
    # Create comparative bar plots
    print("Creating comparative bar plots...")
    create_comparative_bar_plots()
    
    print(f"All comparative visualizations saved to: {COMPARISON_FIGURES_DIR}")

if __name__ == "__main__":
    main() 