#!/usr/bin/env python3
"""
Multi-panel Attack Comparison Graph Generator

This script creates a 3-panel graph comparing different attack strategies 
(random percolation, betweenness centrality, degree centrality) for all brain areas,
comparing original models vs. configuration models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
MULTIPANEL_DIR = os.path.join(FIGURES_DIR, "multipanel")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MULTIPANEL_DIR, exist_ok=True)

def load_attack_data():
    """
    Load all attack data for original and configuration models for all brain areas.
    
    Returns:
        Dictionary containing DataFrames for each attack type, model and brain area
    """
    data = {}
    
    # Brain areas
    areas = ['eb', 'fb', 'mb_kc']
    area_full_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body', 
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    
    # Load random percolation data for each area and model type
    for area in areas:
        area_full = area_full_names.get(area, area)
        
        # Original model - random percolation
        original_file = os.path.join(RESULTS_DIR, f"{area}_original_percolation_results.csv")
        if os.path.exists(original_file):
            data[f"{area}_original_random"] = pd.read_csv(original_file)
        
        # Configuration model - random percolation
        config_file = os.path.join(RESULTS_DIR, f"{area}_config_model_percolation_results.csv")
        if os.path.exists(config_file):
            data[f"{area}_config_random"] = pd.read_csv(config_file)
        
        # Original model - degree attack
        original_degree_file = os.path.join(RESULTS_DIR, f"{area_full}_degree_attack_results.csv")
        if os.path.exists(original_degree_file):
            data[f"{area}_original_degree"] = pd.read_csv(original_degree_file)
        
        # Configuration model - degree attack
        config_degree_file = os.path.join(RESULTS_DIR, f"{area_full}_config_degree_attack_results.csv")
        if os.path.exists(config_degree_file):
            data[f"{area}_config_degree"] = pd.read_csv(config_degree_file)
        
        # Original model - betweenness attack
        original_betweenness_file = os.path.join(RESULTS_DIR, f"{area_full}_betweenness_attack_results.csv")
        if os.path.exists(original_betweenness_file):
            data[f"{area}_original_betweenness"] = pd.read_csv(original_betweenness_file)
        
        # Configuration model - betweenness attack
        config_betweenness_file = os.path.join(RESULTS_DIR, f"{area_full}_config_betweenness_attack_results.csv")
        if os.path.exists(config_betweenness_file):
            data[f"{area}_config_betweenness"] = pd.read_csv(config_betweenness_file)
    
    return data

def standardize_column_names(df):
    """
    Standardize column names to ensure consistent access across different data formats.
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        DataFrame with standardized column names
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check if removal column is present
    if 'removal_fraction' in df.columns:
        df_copy['removal_probability'] = df['removal_fraction']
    
    # Check if LCC size column is present
    if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
        df_copy['mean_lcc_size'] = df['lcc_size']
    
    return df_copy

def create_multipanel_comparison(data):
    """
    Create a 3-panel graph comparing different attack strategies.
    
    Args:
        data: Dictionary containing attack results DataFrames
    """
    # Set up the figure with subplots
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    
    # Create the three panels
    ax1 = fig.add_subplot(gs[0, 0])  # Random percolation
    ax2 = fig.add_subplot(gs[0, 1])  # Betweenness attack
    ax3 = fig.add_subplot(gs[0, 2])  # Degree attack
    
    # Set up color schemes
    colors = {'eb': '#1f77b4', 'fb': '#2ca02c', 'mb_kc': '#d62728'}
    line_styles = {'original': '-', 'config': '--'}
    markers = {'original': 'o', 'config': 's'}
    marker_size = 6
    marker_every = 7
    
    # Brain area full names for legend
    area_names = {
        'eb': 'Ellipsoid Body',
        'fb': 'Fan-shaped Body',
        'mb_kc': 'Mushroom Body Kenyon Cells'
    }
    
    # Attack types
    attack_types = ['random', 'betweenness', 'degree']
    attack_titles = {
        'random': 'Random Percolation',
        'betweenness': 'Betweenness Centrality Attack',
        'degree': 'Degree Centrality Attack'
    }
    axes = [ax1, ax2, ax3]

    # Create lists to store handles and labels for a single unified legend
    handles = []
    labels = []
    
    # Plot each attack type in its panel
    for i, attack in enumerate(attack_types):
        ax = axes[i]
            
        for area in ['eb', 'fb', 'mb_kc']:
            # Original model
            key = f"{area}_original_{attack}"
            if key in data and data[key] is not None:
                # Standardize column names
                df = standardize_column_names(data[key])
                
                if 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
                    line, = ax.plot(
                        df['removal_probability'],
                        df['mean_lcc_size'],
                        color=colors[area],
                        linestyle=line_styles['original'],
                        linewidth=2.5,
                        marker=markers['original'],
                        markersize=marker_size,
                        markevery=marker_every,
                        label=f"{area_names[area]} (Original)"
                    )
                    
                    # Add to legend handles only for the first panel to avoid duplicates
                    if i == 0:
                        handles.append(line)
                        labels.append(f"{area_names[area]} (Original)")
            
            # Configuration model
            key = f"{area}_config_{attack}"
            if key in data and data[key] is not None:
                # Standardize column names
                df = standardize_column_names(data[key])
                
                if 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
                    line, = ax.plot(
                        df['removal_probability'],
                        df['mean_lcc_size'],
                        color=colors[area],
                        linestyle=line_styles['config'],
                        linewidth=2.5,
                        marker=markers['config'],
                        markersize=marker_size,
                        markevery=marker_every,
                        label=f"{area_names[area]} (Configuration)"
                    )
                    
                    # Add to legend handles only for the first panel to avoid duplicates
                    if i == 0:
                        handles.append(line)
                        labels.append(f"{area_names[area]} (Configuration)")
        
        # Add labels and customize the plot for each panel
        ax.set_title(attack_titles[attack], fontsize=16, fontweight='bold')
        if i == 0:  # Only add y-label to the first panel
            ax.set_ylabel('Normalized Largest Connected Component Size', fontsize=14)
        ax.set_xlabel('Edge Removal Probability', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        
        # Add tick marks
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add a horizontal line at y=0.05 to show the threshold
        ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.0)
        ax.text(0.05, 0.07, 'Threshold (5%)', fontsize=10, color='gray')
        
        # Add annotation if targeted attack
        if attack in ['betweenness', 'degree']:
            ax.text(0.5, 0.8, "Configuration models\ndisintegrate immediately\nunder targeted attack", 
                   color='#cc0000', fontsize=11, ha='center', 
                   bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='gray', linewidth=0.5))
    
    # Add a unified legend above the plots
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=6, 
        fontsize=11, 
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # Add a main title to the figure
    fig.suptitle('Neural Networks are Robust to Targeted Attacks, Random Networks are Not', 
                fontsize=18, fontweight='bold', y=1.12)
    
    # Save figures in both locations for backward compatibility
    # 1. Original location
    original_output_path = os.path.join(FIGURES_DIR, 'multipanel_attack_comparison_enhanced.png')
    plt.savefig(original_output_path, dpi=300, bbox_inches='tight')
    
    # 2. New dedicated folder
    dedicated_output_path = os.path.join(MULTIPANEL_DIR, 'attack_comparison.png')
    plt.savefig(dedicated_output_path, dpi=300, bbox_inches='tight')
    
    print(f"Figures saved to:")
    print(f"- {original_output_path} (original location)")
    print(f"- {dedicated_output_path} (dedicated folder)")
    
    # Show the plot
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.show()

def main():
    """Main function to run the script."""
    print("Loading attack data...")
    data = load_attack_data()
    
    print("Creating multi-panel comparison graph...")
    create_multipanel_comparison(data)
    
    print("Done!")

if __name__ == "__main__":
    main() 