#!/usr/bin/env python3
"""
Combined Network Figures Generator

This script creates one figure per network (EB, FB, MB-KC) containing:
1. Degree distribution comparison between original and configuration model
2. Percolation and targeted attack performance comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import json
from collections import Counter

# Set plot style
sns.set_style("whitegrid")

# Constants
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SRC_DIR, "data")
CONFIG_MODELS_DIR = os.path.join(SRC_DIR, "config_models")
RESULTS_DIR = os.path.join(SRC_DIR, "results")
FIGURES_DIR = os.path.join(SRC_DIR, "figures")
NEW_FIGURES_DIR = os.path.join(FIGURES_DIR, "combined")

# Ensure output directory exists
os.makedirs(NEW_FIGURES_DIR, exist_ok=True)

# Network names mapping
NETWORK_NAMES = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body Kenyon Cells'
}

# File name mappings (some files use different naming conventions)
FILE_NAME_MAPPINGS = {
    'eb': 'ellipsoid_body',
    'fb': 'fan-shaped_body',
    'mb_kc': 'mushroom_body_kenyon_cell'
}

def load_networks(network_type):
    """
    Load original and configuration model networks.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Tuple of (original_network, config_model)
    """
    # Load original network
    original_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    G_original = nx.read_gexf(original_path)
    
    # Load configuration model
    config_path = os.path.join(CONFIG_MODELS_DIR, f"{network_type}_config_model.gexf")
    G_config = nx.read_gexf(config_path)
    
    return G_original, G_config

def load_percolation_data(network_type):
    """
    Load percolation results for original and configuration models.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Tuple of (original_data, config_data)
    """
    # Original model
    original_file = os.path.join(RESULTS_DIR, f"{network_type}_original_percolation_results.csv")
    original_data = pd.read_csv(original_file)
    
    # Configuration model
    config_file = os.path.join(RESULTS_DIR, f"{network_type}_config_model_percolation_results.csv")
    config_data = pd.read_csv(config_file)
    
    return original_data, config_data

def load_targeted_attack_data(network_type):
    """
    Load targeted attack results for original and configuration models.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Tuple of (degree_attack_data, random_attack_data)
    """
    # Get file name with correct mapping
    file_name = FILE_NAME_MAPPINGS[network_type]
    
    # Degree-based targeted attack
    degree_file = os.path.join(RESULTS_DIR, f"{file_name}_degree_attack_results.csv")
    degree_data = pd.read_csv(degree_file)
    
    # Random attack (for comparison)
    random_file = os.path.join(RESULTS_DIR, f"{file_name}_random_attack_results.csv")
    random_data = pd.read_csv(random_file)
    
    return degree_data, random_data

def load_attack_parameters():
    """
    Load attack parameters with critical thresholds.
    
    Returns:
        Dictionary with attack parameters
    """
    params_file = os.path.join(RESULTS_DIR, "targeted_attack_parameters.json")
    with open(params_file, 'r') as f:
        return json.load(f)

def create_combined_figure(network_type):
    """
    Create a combined figure for a single network with degree distribution, 
    percolation, and targeted attack analysis.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
    """
    # Load networks and data
    G_original, G_config = load_networks(network_type)
    percolation_original, percolation_config = load_percolation_data(network_type)
    targeted_original, random_original = load_targeted_attack_data(network_type)
    
    # Get file name with correct mapping for parameter lookup
    file_name = FILE_NAME_MAPPINGS[network_type].replace('-', '_')
    attack_params = load_attack_parameters()
    
    # Get critical thresholds
    # For percolation
    percolation_summary = pd.read_csv(os.path.join(RESULTS_DIR, 'percolation_comparison_summary.csv'))
    orig_perc_threshold = percolation_summary[(percolation_summary['network_type'] == network_type) & 
                                            (percolation_summary['model_type'] == 'original')]['threshold_simple'].values[0]
    config_perc_threshold = percolation_summary[(percolation_summary['network_type'] == network_type) & 
                                              (percolation_summary['model_type'] == 'config_model')]['threshold_simple'].values[0]
    
    # For targeted attacks
    if file_name == "fan_shaped_body":
        param_key = "Fan_Shaped_Body"
    elif file_name == "mushroom_body_kenyon_cell":
        param_key = "Mushroom_Body_Kenyon_Cell"
    else:
        param_key = "Ellipsoid_Body"
        
    targeted_threshold = attack_params[param_key]["targeted"]["critical_fraction"]
    random_threshold = attack_params[param_key]["random"]["critical_fraction"]
    
    # Create figure with 2 rows and 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Network name for title
    network_name = NETWORK_NAMES[network_type]
    
    # --- Row 1: Degree Distribution ---
    
    # Get degree data
    in_degrees_orig = [d for _, d in G_original.in_degree()]
    out_degrees_orig = [d for _, d in G_original.out_degree()]
    in_degrees_config = [d for _, d in G_config.in_degree()]
    out_degrees_config = [d for _, d in G_config.out_degree()]
    
    # Calculate distributions with normalized axes
    in_degree_count_orig = Counter(in_degrees_orig)
    in_deg_orig, in_cnt_orig = zip(*sorted(in_degree_count_orig.items()))
    in_cnt_orig_norm = [count / len(in_degrees_orig) for count in in_cnt_orig]
    
    in_degree_count_config = Counter(in_degrees_config)
    in_deg_config, in_cnt_config = zip(*sorted(in_degree_count_config.items()))
    in_cnt_config_norm = [count / len(in_degrees_config) for count in in_cnt_config]
    
    out_degree_count_orig = Counter(out_degrees_orig)
    out_deg_orig, out_cnt_orig = zip(*sorted(out_degree_count_orig.items()))
    out_cnt_orig_norm = [count / len(out_degrees_orig) for count in out_cnt_orig]
    
    out_degree_count_config = Counter(out_degrees_config)
    out_deg_config, out_cnt_config = zip(*sorted(out_degree_count_config.items()))
    out_cnt_config_norm = [count / len(out_degrees_config) for count in out_cnt_config]
    
    # Plot In-Degree Distribution (normalized)
    ax1.loglog(in_deg_orig, in_cnt_orig_norm, 'bo-', alpha=0.7, 
              label=f"Original ({G_original.number_of_nodes()} nodes)")
    ax1.loglog(in_deg_config, in_cnt_config_norm, 'go-', alpha=0.7, 
              label=f"Config Model ({G_config.number_of_nodes()} nodes)")
    ax1.set_title(f"{network_name} In-Degree Distribution (Normalized)")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Out-Degree Distribution (normalized)
    ax2.loglog(out_deg_orig, out_cnt_orig_norm, 'ro-', alpha=0.7, 
              label=f"Original ({G_original.number_of_nodes()} nodes)")
    ax2.loglog(out_deg_config, out_cnt_config_norm, 'mo-', alpha=0.7, 
              label=f"Config Model ({G_config.number_of_nodes()} nodes)")
    ax2.set_title(f"{network_name} Out-Degree Distribution (Normalized)")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # --- Row 2, Left: Percolation Performance ---
    
    # Plot Percolation Results
    ax3.errorbar(
        percolation_original['removal_probability'],
        percolation_original['mean_lcc_size'],
        yerr=percolation_original['std_lcc_size'],
        fmt='o-',
        color='blue',
        alpha=0.7,
        label=f"Original Network (qc ≈ {orig_perc_threshold:.3f})"
    )
    ax3.errorbar(
        percolation_config['removal_probability'],
        percolation_config['mean_lcc_size'],
        yerr=percolation_config['std_lcc_size'],
        fmt='s-',
        color='green',
        alpha=0.7,
        label=f"Config Model (qc ≈ {config_perc_threshold:.3f})"
    )
    
    # Add vertical lines at thresholds
    ax3.axvline(x=orig_perc_threshold, color='blue', linestyle='--', alpha=0.5)
    ax3.axvline(x=config_perc_threshold, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_title(f"{network_name} Percolation Analysis")
    ax3.set_xlabel("Edge Removal Probability")
    ax3.set_ylabel("Largest Connected Component Size (Normalized)")
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # --- Row 2, Right: Targeted Attack Analysis ---
    
    # Plot Targeted Attack Results
    ax4.plot(
        targeted_original['removal_fraction'],
        targeted_original['mean_lcc_size'],
        'b-', marker='o', markevery=5, alpha=0.7,
        label=f"Degree Attack (fc ≈ {targeted_threshold:.3f})"
    )
    ax4.plot(
        random_original['removal_fraction'],
        random_original['mean_lcc_size'],
        'g-', marker='s', markevery=5, alpha=0.7,
        label=f"Random Attack (fc ≈ {random_threshold:.3f})"
    )
    
    # Add vertical lines at thresholds
    ax4.axvline(x=targeted_threshold, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(x=random_threshold, color='green', linestyle='--', alpha=0.5)
    
    ax4.set_title(f"{network_name} Targeted Attack Analysis")
    ax4.set_xlabel("Node Removal Fraction")
    ax4.set_ylabel("Largest Connected Component Size (Normalized)")
    ax4.set_xlim(0, 1.0)
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Overall title
    plt.suptitle(f"{network_name} Neural Circuit Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    output_path = os.path.join(NEW_FIGURES_DIR, f"{network_type}_combined_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined figure for {network_name}: {output_path}")

def main():
    """Main function to run the script."""
    print("Creating combined figures for each network...")
    
    for network_type in ['eb', 'fb', 'mb_kc']:
        create_combined_figure(network_type)
    
    print("All combined figures created successfully!")

if __name__ == "__main__":
    main() 