#!/usr/bin/env python3
"""
Comparison of Targeted Attack Results Between Original Networks and Configuration Models

This script creates visualizations comparing the targeted attack results 
between original networks and their corresponding configuration models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Network types and their display names
NETWORKS = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body KC'
}

# Attack strategies
ATTACK_STRATEGIES = ['degree', 'betweenness', 'random']
STRATEGY_COLORS = {
    'degree': 'red',
    'betweenness': 'blue',
    'random': 'green'
}
STRATEGY_NAMES = {
    'degree': 'Degree-Based',
    'betweenness': 'Betweenness-Based',
    'random': 'Random'
}

def load_results(network_type, attack_strategy, is_config=False):
    """Load results for a specific network and attack strategy.
    
    Args:
        network_type: Short network name ('eb', 'fb', 'mb_kc')
        attack_strategy: Attack strategy ('degree', 'betweenness', 'random')
        is_config: Whether to load results for config model (True) or original (False)
        
    Returns:
        DataFrame with results
    """
    # Get full network name
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # Construct filename
    config_prefix = "config_" if is_config else ""
    filename = f"{full_name.lower().replace(' ', '_')}_{config_prefix}{attack_strategy}_attack_results.csv"
    
    try:
        return pd.read_csv(os.path.join(RESULTS_DIR, filename))
    except FileNotFoundError:
        print(f"Warning: Could not find results file {filename}")
        return None

def plot_network_comparison(network_type, attack_strategy):
    """Create comparison plot for original vs config model for a specific network and attack.
    
    Args:
        network_type: Short network name ('eb', 'fb', 'mb_kc')
        attack_strategy: Attack strategy ('degree', 'betweenness', 'random')
        
    Returns:
        Figure object
    """
    # Load results
    original_df = load_results(network_type, attack_strategy, is_config=False)
    config_df = load_results(network_type, attack_strategy, is_config=True)
    
    if original_df is None or config_df is None:
        print(f"Skipping plot for {network_type} with {attack_strategy} attack (missing data)")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot original network
    ax.errorbar(
        original_df['removal_fraction'], 
        original_df['mean_lcc_size'],
        yerr=original_df['std_lcc_size'] if 'std_lcc_size' in original_df.columns else None,
        marker='o',
        markersize=6,
        markevery=5,
        linewidth=2,
        label=f'Original Network',
        color=STRATEGY_COLORS[attack_strategy],
        alpha=0.8
    )
    
    # Plot configuration model
    ax.errorbar(
        config_df['removal_fraction'], 
        config_df['mean_lcc_size'],
        yerr=config_df['std_lcc_size'] if 'std_lcc_size' in config_df.columns else None,
        marker='s',
        markersize=6,
        markevery=5,
        linewidth=2,
        linestyle='--',
        label=f'Configuration Model',
        color=STRATEGY_COLORS[attack_strategy],
        alpha=0.5
    )
    
    # Add labels and title
    ax.set_xlabel('Edge Removal Fraction')
    ax.set_ylabel('Largest Connected Component Size (Normalized)')
    ax.set_title(f'{NETWORKS[network_type]}: {STRATEGY_NAMES[attack_strategy]} Attack')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotations about critical points and robustness
    # This would require loading the parameters files
    
    plt.tight_layout()
    return fig

def create_multipanel_comparison():
    """Create multipanel comparison of all networks and attack strategies.
    
    Returns:
        Figure object
    """
    # Create figure with subplots
    fig, axes = plt.subplots(len(NETWORKS), len(ATTACK_STRATEGIES), figsize=(15, 12))
    
    # Iterate over networks and strategies
    for i, (net_key, net_name) in enumerate(NETWORKS.items()):
        for j, strategy in enumerate(ATTACK_STRATEGIES):
            ax = axes[i, j]
            
            # Load results
            original_df = load_results(net_key, strategy, is_config=False)
            config_df = load_results(net_key, strategy, is_config=True)
            
            if original_df is None or config_df is None:
                ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
                continue
            
            # Plot original network
            ax.errorbar(
                original_df['removal_fraction'], 
                original_df['mean_lcc_size'],
                yerr=original_df['std_lcc_size'] if 'std_lcc_size' in original_df.columns else None,
                marker='o',
                markersize=4,
                markevery=10,
                linewidth=1.5,
                label=f'Original',
                color=STRATEGY_COLORS[strategy],
                alpha=0.8
            )
            
            # Plot configuration model
            ax.errorbar(
                config_df['removal_fraction'], 
                config_df['mean_lcc_size'],
                yerr=config_df['std_lcc_size'] if 'std_lcc_size' in config_df.columns else None,
                marker='s',
                markersize=4,
                markevery=10,
                linewidth=1.5,
                linestyle='--',
                label=f'Config Model',
                color=STRATEGY_COLORS[strategy],
                alpha=0.5
            )
            
            # Set titles only for the top row
            if i == 0:
                ax.set_title(f'{STRATEGY_NAMES[strategy]}')
            
            # Set y-axis label only for the leftmost column
            if j == 0:
                ax.set_ylabel(f'{net_name}\nLCC Size')
            
            # Set x-axis label only for the bottom row
            if i == len(NETWORKS) - 1:
                ax.set_xlabel('Edge Removal Fraction')
            
            # Set axis limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            
            # Add legend only for the top-right plot
            if i == 0 and j == len(ATTACK_STRATEGIES) - 1:
                ax.legend(loc='upper right')
            
            # Add grid
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Targeted Attacks: Original Networks vs. Configuration Models', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, "targeted_attack_original_vs_config_comparison.png"), 
                dpi=300, bbox_inches='tight')
    
    return fig

def create_bar_chart_comparison():
    """Create bar chart comparing critical thresholds and robustness indices.
    
    Returns:
        Figure object
    """
    # Data structure to hold results
    data = []
    
    # Load parameter files
    for net_key, net_name in NETWORKS.items():
        # Get full network name
        network_names = {
            'eb': 'ellipsoid_body',
            'fb': 'fan-shaped_body',
            'mb_kc': 'mushroom_body_kenyon_cell'
        }
        full_name = network_names.get(net_key, net_key)
        
        # Original network parameters
        try:
            with open(os.path.join(RESULTS_DIR, f"targeted_attack_parameters.json"), 'r') as f:
                original_params = json.load(f)
                for strategy in ATTACK_STRATEGIES:
                    if net_key in original_params and strategy in original_params[net_key]:
                        params = original_params[net_key][strategy]
                        data.append({
                            'network': net_name,
                            'model_type': 'Original',
                            'attack_strategy': STRATEGY_NAMES[strategy],
                            'critical_fraction': params.get('critical_fraction', None),
                            'robustness_index': params.get('robustness_index', None)
                        })
        except FileNotFoundError:
            print(f"Warning: Could not find original parameters file for {net_key}")
        
        # Config model parameters
        try:
            with open(os.path.join(RESULTS_DIR, f"{full_name}_config_attack_parameters.json"), 'r') as f:
                config_params = json.load(f)
                for strategy in ATTACK_STRATEGIES:
                    if strategy in config_params:
                        params = config_params[strategy]
                        data.append({
                            'network': net_name,
                            'model_type': 'Config Model',
                            'attack_strategy': STRATEGY_NAMES[strategy],
                            'critical_fraction': params.get('critical_fraction', None),
                            'robustness_index': params.get('robustness_index', None)
                        })
        except FileNotFoundError:
            print(f"Warning: Could not find config parameters file for {full_name}")
    
    if not data:
        print("No parameter data found for bar chart comparison")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create two subplots: one for critical fraction, one for robustness index
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot critical fraction
    sns.barplot(
        data=df, 
        x='network', 
        y='critical_fraction', 
        hue='model_type', 
        palette=['royalblue', 'lightsteelblue'],
        alpha=0.8,
        dodge=True,
        ax=ax1
    )
    ax1.set_title('Critical Edge Removal Fraction Comparison')
    ax1.set_ylabel('Critical Fraction')
    ax1.set_ylim(0, 1)
    ax1.legend(title='Network Type')
    
    # Plot robustness index
    sns.barplot(
        data=df, 
        x='network', 
        y='robustness_index', 
        hue='model_type', 
        palette=['royalblue', 'lightsteelblue'],
        alpha=0.8,
        dodge=True,
        ax=ax2
    )
    ax2.set_title('Robustness Index Comparison')
    ax2.set_ylabel('Robustness Index')
    ax2.set_ylim(0, 1)
    ax2.legend(title='Network Type')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, "targeted_attack_bar_comparison.png"), 
                dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Main function."""
    print("Creating targeted attack comparison visualizations...")
    
    # Create individual plots for each network and attack strategy
    for net_key in NETWORKS.keys():
        for strategy in ATTACK_STRATEGIES:
            fig = plot_network_comparison(net_key, strategy)
            if fig:
                plt.savefig(os.path.join(FIGURES_DIR, f"{net_key}_{strategy}_original_vs_config.png"), 
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # Create multipanel comparison
    create_multipanel_comparison()
    
    # Create bar chart comparison
    create_bar_chart_comparison()
    
    print(f"Visualizations saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 