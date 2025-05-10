#!/usr/bin/env python3
"""
Percolation Comparison Graph Generator

This script creates a line graph comparing the original models with the configuration models 
for percolation analysis across all three brain areas (EB, FB, MB-KC).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_percolation_data():
    """
    Load percolation results for original and configuration models for all brain areas.
    
    Returns:
        Dictionary containing DataFrames for each model and brain area
    """
    data = {}
    
    # Brain areas
    areas = ['eb', 'fb', 'mb_kc']
    
    # Load data for each area and model type
    for area in areas:
        # Original model
        original_file = os.path.join(RESULTS_DIR, f"{area}_original_percolation_results.csv")
        data[f"{area}_original"] = pd.read_csv(original_file)
        
        # Configuration model
        config_file = os.path.join(RESULTS_DIR, f"{area}_config_model_percolation_results.csv")
        data[f"{area}_config"] = pd.read_csv(config_file)
    
    return data

def create_comparison_graph(data):
    """
    Create a line graph comparing original and configuration models for all three brain areas.
    
    Args:
        data: Dictionary containing percolation results DataFrames
    """
    # Set up the plot with seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 9))
    
    # Colors and line styles
    colors = {'eb': '#1f77b4', 'fb': '#2ca02c', 'mb_kc': '#d62728'}
    line_styles = {'original': '-', 'config': '--'}
    markers = {'original': 'o', 'config': 's'}
    marker_size = 8
    
    # Brain area full names for legend
    area_names = {
        'eb': 'Ellipsoid Body',
        'fb': 'Fan-shaped Body',
        'mb_kc': 'Mushroom Body Kenyon Cells'
    }
    
    # Plot each dataset
    for area in ['eb', 'fb', 'mb_kc']:
        # Original model
        plt.plot(
            data[f"{area}_original"]['removal_probability'],
            data[f"{area}_original"]['mean_lcc_size'],
            color=colors[area],
            linestyle=line_styles['original'],
            linewidth=3.0,
            marker=markers['original'],
            markersize=marker_size,
            markevery=5,
            label=f"{area_names[area]} (Original)"
        )
        
        # Configuration model
        plt.plot(
            data[f"{area}_config"]['removal_probability'],
            data[f"{area}_config"]['mean_lcc_size'],
            color=colors[area],
            linestyle=line_styles['config'],
            linewidth=3.0,
            marker=markers['config'],
            markersize=marker_size,
            markevery=5,
            label=f"{area_names[area]} (Configuration)"
        )
    
    # Add labels and title
    plt.xlabel('Percolation Parameter (Edge Removal Probability)', fontsize=16)
    plt.ylabel('Normalized Largest Connected Component Size', fontsize=16)
    plt.title('Percolation Analysis: Original vs. Configuration Models', fontsize=18, fontweight='bold')
    
    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='best', framealpha=0.9)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    # Add tick marks
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    
    # Add a gray horizontal line at y=0.05 to show the threshold used in the analysis
    plt.axhline(y=0.05, color='gray', linestyle=':', linewidth=2)
    plt.text(0.02, 0.06, 'Threshold (5%)', fontsize=12, color='gray')
    
    # Add annotations for critical thresholds
    # Get summary data
    summary = pd.read_csv(os.path.join(RESULTS_DIR, 'percolation_comparison_summary.csv'))
    
    # Add annotations for critical thresholds
    for area in ['eb', 'fb', 'mb_kc']:
        # Original model threshold
        orig_threshold = summary[(summary['network_type'] == area) & 
                                (summary['model_type'] == 'original')]['threshold_simple'].values[0]
        
        # Config model threshold
        config_threshold = summary[(summary['network_type'] == area) & 
                                  (summary['model_type'] == 'config_model')]['threshold_simple'].values[0]
        
        # Add vertical lines at thresholds
        plt.axvline(x=orig_threshold, color=colors[area], linestyle='-', alpha=0.3)
        plt.axvline(x=config_threshold, color=colors[area], linestyle='--', alpha=0.3)
        
        # Add text annotations
        plt.text(orig_threshold - 0.02, 0.9, f"{area} orig\n(q={orig_threshold:.2f})", 
                 color=colors[area], fontsize=12, ha='right')
        plt.text(config_threshold + 0.02, 0.85, f"{area} config\n(q={config_threshold:.2f})", 
                 color=colors[area], fontsize=12, ha='left')
    
    # Save the figure
    output_path = os.path.join(FIGURES_DIR, 'percolation_model_comparison_enhanced.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the script."""
    print("Loading percolation data...")
    data = load_percolation_data()
    
    print("Creating comparison graph...")
    create_comparison_graph(data)
    
    print("Done!")

if __name__ == "__main__":
    main() 