#!/usr/bin/env python3
"""
Node Percolation Analysis Visualization for Network Models

This script performs node percolation analysis (random and degree-based) on:
1. Original neural networks
2. Scaled configuration models (1500 nodes)
3. Unscaled configuration models
4. Upscaled configuration models (3500 nodes)

It then creates a comprehensive visualization comparing the results.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import multiprocessing
from functools import partial

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
NODE_PERC_DIR = os.path.join(RESULTS_DIR, "node_percolation")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(NODE_PERC_DIR, exist_ok=True)

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

# Number of Monte Carlo simulations per probability value
NUM_SIMULATIONS = 20
# Number of probability steps to evaluate
NUM_PROB_STEPS = 50

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

def run_random_node_percolation(G, removal_probability, num_simulations=NUM_SIMULATIONS):
    """Run random node percolation simulation for a specific node removal probability.
    
    Args:
        G: NetworkX graph
        removal_probability: Probability of removing a node
        num_simulations: Number of Monte Carlo simulations to run
        
    Returns:
        Dictionary with simulation results
    """
    # Create an undirected graph for percolation analysis
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # List to store largest connected component sizes for each simulation
    lcc_sizes = []
    
    for _ in range(num_simulations):
        # Create a copy of the original graph for this simulation
        G_sim = G_undirected.copy()
        
        # Randomly remove nodes with the given probability
        nodes_to_remove = []
        for node in G_sim.nodes():
            if np.random.random() < removal_probability:
                nodes_to_remove.append(node)
        
        G_sim.remove_nodes_from(nodes_to_remove)
        
        # Calculate largest connected component size
        if len(G_sim) > 0:  # Check if graph is not empty
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc) / original_size  # Normalized size
        else:
            lcc_size = 0
        
        lcc_sizes.append(lcc_size)
    
    # Calculate statistics across simulations
    mean_lcc = np.mean(lcc_sizes)
    std_lcc = np.std(lcc_sizes)
    
    return {
        'removal_probability': removal_probability,
        'mean_lcc_size': mean_lcc,
        'std_lcc_size': std_lcc
    }

def run_degree_node_percolation(G, removal_fraction, num_simulations=NUM_SIMULATIONS):
    """Run degree-based node percolation simulation for a specific removal fraction.
    
    Args:
        G: NetworkX graph
        removal_fraction: Fraction of nodes to remove (highest degree first)
        num_simulations: Number of Monte Carlo simulations to run
        
    Returns:
        Dictionary with simulation results
    """
    # Create an undirected graph for percolation analysis
    G_undirected = G.to_undirected()
    original_size = G_undirected.number_of_nodes()
    
    # List to store largest connected component sizes for each simulation
    lcc_sizes = []
    
    for _ in range(num_simulations):
        # Create a copy of the original graph for this simulation
        G_sim = G_undirected.copy()
        
        # Get degree centrality
        degree_dict = dict(G_sim.degree())
        
        # Sort nodes by degree (highest first)
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        
        # Calculate number of nodes to remove
        num_nodes_to_remove = int(len(sorted_nodes) * removal_fraction)
        
        # Remove highest degree nodes
        nodes_to_remove = sorted_nodes[:num_nodes_to_remove]
        G_sim.remove_nodes_from(nodes_to_remove)
        
        # Calculate largest connected component size
        if len(G_sim) > 0:  # Check if graph is not empty
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc) / original_size  # Normalized size
        else:
            lcc_size = 0
        
        lcc_sizes.append(lcc_size)
    
    # Calculate statistics across simulations
    mean_lcc = np.mean(lcc_sizes)
    std_lcc = np.std(lcc_sizes)
    
    return {
        'removal_probability': removal_fraction,  # Keep consistent naming with random
        'mean_lcc_size': mean_lcc,
        'std_lcc_size': std_lcc
    }

def process_random_probability(args):
    """Process a single random removal probability (for parallel processing)."""
    G, removal_probability, num_simulations = args
    return run_random_node_percolation(G, removal_probability, num_simulations)

def process_degree_probability(args):
    """Process a single degree-based removal fraction (for parallel processing)."""
    G, removal_fraction, num_simulations = args
    return run_degree_node_percolation(G, removal_fraction, num_simulations)

def run_node_percolation_analysis(G, network_name, model_type, percolation_type='random', 
                                 num_prob_steps=NUM_PROB_STEPS, num_simulations=NUM_SIMULATIONS):
    """Run node percolation analysis across a range of removal probabilities.
    
    Args:
        G: NetworkX graph
        network_name: Network identifier (eb, fb, mb_kc)
        model_type: Model type (original, scaled_config, etc.)
        percolation_type: 'random' or 'degree'
        num_prob_steps: Number of probability steps to evaluate
        num_simulations: Number of simulations per probability value
        
    Returns:
        DataFrame with results for each probability
    """
    if G is None:
        print(f"Cannot run analysis on None graph for {network_name} {model_type}")
        return None
    
    print(f"\nRunning {percolation_type} node percolation analysis for {network_name} {model_type}...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal probabilities
    removal_probabilities = np.linspace(0, 1, num_prob_steps)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Create argument list
    args_list = [(G, p, num_simulations) for p in removal_probabilities]
    
    # Run simulations in parallel
    if percolation_type == 'random':
        results = list(tqdm(
            pool.imap(process_random_probability, args_list),
            total=len(args_list),
            desc=f"Simulating random node removal ({model_type})"
        ))
    else:  # degree-based
        results = list(tqdm(
            pool.imap(process_degree_probability, args_list),
            total=len(args_list),
            desc=f"Simulating degree-based node removal ({model_type})"
        ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    filename = f"{network_name}_{model_type}_{percolation_type}_node_percolation.csv"
    df.to_csv(os.path.join(NODE_PERC_DIR, filename), index=False)
    
    return df

def run_all_percolation_analyses():
    """Run percolation analyses for all network types and model types."""
    results = {}
    
    for network_type in NETWORKS:
        results[network_type] = {}
        
        for model_type in MODEL_TYPES:
            # Load network
            G = load_network(network_type, model_type)
            
            if G is not None:
                # Run random node percolation
                random_df = run_node_percolation_analysis(
                    G, network_type, model_type, percolation_type='random'
                )
                
                # Run degree-based node percolation
                degree_df = run_node_percolation_analysis(
                    G, network_type, model_type, percolation_type='degree'
                )
                
                results[network_type][model_type] = {
                    'random': random_df,
                    'degree': degree_df
                }
            else:
                print(f"Skipping {network_type} {model_type} - graph could not be loaded")
    
    return results

def load_node_percolation_results(network_type, model_type, percolation_type):
    """Load node percolation results from saved files.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'original', 'scaled_config', 'unscaled_config', or 'upscaled_config'
        percolation_type: 'random' or 'degree'
        
    Returns:
        DataFrame with results
    """
    filename = f"{network_type}_{model_type}_{percolation_type}_node_percolation.csv"
    file_path = os.path.join(NODE_PERC_DIR, filename)
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading results from {file_path}: {e}")
        return None

def extract_percolation_threshold(df):
    """Extract percolation threshold from results.
    
    Args:
        df: DataFrame with percolation results
        
    Returns:
        Percolation threshold value
    """
    if df is None or 'removal_probability' not in df.columns or 'mean_lcc_size' not in df.columns:
        return np.nan
    
    # Find threshold where LCC drops below 0.05
    for i in range(len(df) - 1):
        if df['mean_lcc_size'].iloc[i] >= 0.05 and df['mean_lcc_size'].iloc[i+1] < 0.05:
            # Linear interpolation
            x1 = df['removal_probability'].iloc[i]
            x2 = df['removal_probability'].iloc[i+1]
            y1 = df['mean_lcc_size'].iloc[i]
            y2 = df['mean_lcc_size'].iloc[i+1]
            
            threshold = x1 + (x2 - x1) * (0.05 - y1) / (y2 - y1)
            return round(threshold, 3)
    
    return np.nan

def plot_node_percolation_comparison(ax, network_type, percolation_type, title):
    """Plot node percolation comparison for all model types.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
        percolation_type: 'random' or 'degree'
        title: Title for the plot
    """
    marker_size = 6
    marker_every = 5
    
    # Plot each model type
    for model_type in MODEL_TYPES:
        # Load percolation results
        df = load_node_percolation_results(network_type, model_type, percolation_type)
        
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
    ax.set_xlabel('Node Removal Probability')
    ax.set_ylabel('Largest Connected Component')
    
    # Set axis limits based on percolation type
    if percolation_type == 'degree':
        # Zoom in on the threshold region for degree-based plots
        ax.set_xlim(0, 0.1)  # Zoom in to show only 0-10% removal
    else:
        ax.set_xlim(0, 1)  # Full range for random percolation
        
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def create_metrics_table(ax, network_type):
    """Create a table of node percolation thresholds for all model types.
    
    Args:
        ax: Matplotlib axis
        network_type: 'eb', 'fb', or 'mb_kc'
    """
    # Headers for the table
    column_labels = ['Metric', 'Original', 'Scaled Config', 'Unscaled Config', 'Upscaled Config']
    row_labels = [
        'Network Size (Nodes)',
        'Network Size (Edges)',
        'Random Node Percolation Threshold', 
        'Degree-based Node Percolation Threshold'
    ]
    
    # Collect metrics for each model type
    metrics = {}
    for model_type in MODEL_TYPES:
        # Load network
        G = load_network(network_type, model_type=model_type)
        
        metrics[model_type] = {}
        
        if G:
            # Basic network sizes
            metrics[model_type]['Network Size (Nodes)'] = G.number_of_nodes()
            metrics[model_type]['Network Size (Edges)'] = G.number_of_edges()
            
            # Load percolation results and extract thresholds
            random_df = load_node_percolation_results(network_type, model_type, 'random')
            degree_df = load_node_percolation_results(network_type, model_type, 'degree')
            
            metrics[model_type]['Random Node Percolation Threshold'] = extract_percolation_threshold(random_df)
            metrics[model_type]['Degree-based Node Percolation Threshold'] = extract_percolation_threshold(degree_df)
    
    # Create table data
    cell_data = []
    for row_label in row_labels:
        row = [row_label]
        for model_type in MODEL_TYPES:
            if model_type in metrics and row_label in metrics[model_type]:
                row.append(metrics[model_type][row_label])
            else:
                row.append('N/A')
        cell_data.append(row)
    
    # Create the table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=cell_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Define colors with proper alpha values
    header_color = '#E6E6E6'
    highlight_colors = {
        'original': (0.0, 0.0, 1.0, 0.2),       # Blue with alpha
        'scaled_config': (1.0, 0.0, 0.0, 0.2),  # Red with alpha
        'unscaled_config': (0.0, 0.8, 0.0, 0.2), # Green with alpha
        'upscaled_config': (0.5, 0.0, 0.5, 0.2)  # Purple with alpha
    }
    
    # Color the header row
    for j, label in enumerate(column_labels):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold')
    
    # Color the metric names column
    for i in range(len(row_labels)):
        cell = table[(i+1, 0)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold')
    
    # Highlight threshold rows with colors
    highlight_rows = [2, 3]  # Random and Degree thresholds (0-indexed after column labels)
    for i in highlight_rows:
        for j in range(1, len(column_labels)):
            cell = table[(i+1, j)]  # +1 to account for header row
            cell_value = cell.get_text().get_text()
            if cell_value != 'N/A' and cell_value != '':
                model_type = MODEL_TYPES[j-1]
                cell.set_facecolor(highlight_colors[model_type])
    
    ax.set_title(f"{NETWORKS[network_type]} Node Percolation Thresholds", fontsize=14, fontweight='bold', pad=20)
    
    return metrics

def create_node_percolation_visualization():
    """Create the comprehensive node percolation visualization."""
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create the panels - 4 rows, 3 columns
    # Row 1: Random node percolation
    ax_rand_eb = fig.add_subplot(gs[0, 0])  # EB random node percolation
    ax_rand_fb = fig.add_subplot(gs[0, 1])  # FB random node percolation
    ax_rand_mb = fig.add_subplot(gs[0, 2])  # MB random node percolation
    
    # Row 2: Degree-based node percolation
    ax_deg_eb = fig.add_subplot(gs[1, 0])  # EB degree-based node percolation
    ax_deg_fb = fig.add_subplot(gs[1, 1])  # FB degree-based node percolation
    ax_deg_mb = fig.add_subplot(gs[1, 2])  # MB degree-based node percolation
    
    # Row 3: Comparison metrics tables
    ax_metrics_eb = fig.add_subplot(gs[2, 0])  # EB metrics
    ax_metrics_fb = fig.add_subplot(gs[2, 1])  # FB metrics
    ax_metrics_mb = fig.add_subplot(gs[2, 2])  # MB metrics
    
    # Row 4: Extra space for explanation/notes
    ax_notes = fig.add_subplot(gs[3, :])  # Notes spanning all columns
    
    # Plot random node percolation results (Row 1)
    plot_node_percolation_comparison(
        ax_rand_eb, 'eb', 'random', 
        f"{NETWORKS['eb']} Random Node Percolation"
    )
    plot_node_percolation_comparison(
        ax_rand_fb, 'fb', 'random', 
        f"{NETWORKS['fb']} Random Node Percolation"
    )
    plot_node_percolation_comparison(
        ax_rand_mb, 'mb_kc', 'random', 
        f"{NETWORKS['mb_kc']} Random Node Percolation"
    )
    
    # Plot degree-based node percolation results (Row 2)
    plot_node_percolation_comparison(
        ax_deg_eb, 'eb', 'degree', 
        f"{NETWORKS['eb']} Degree-based Node Percolation"
    )
    plot_node_percolation_comparison(
        ax_deg_fb, 'fb', 'degree', 
        f"{NETWORKS['fb']} Degree-based Node Percolation"
    )
    plot_node_percolation_comparison(
        ax_deg_mb, 'mb_kc', 'degree', 
        f"{NETWORKS['mb_kc']} Degree-based Node Percolation"
    )
    
    # Create network metrics tables (Row 3)
    create_metrics_table(ax_metrics_eb, 'eb')
    create_metrics_table(ax_metrics_fb, 'fb')
    create_metrics_table(ax_metrics_mb, 'mb_kc')
    
    # Add notes (Row 4)
    ax_notes.axis('off')
    notes_text = """
    Node Percolation Analysis Notes:
    
    Random Node Percolation: Nodes are removed randomly with equal probability.
    
    Degree-based Node Percolation: Nodes are removed in order of decreasing degree centrality 
    (highest degree nodes removed first).
    
    Threshold Definition: Point at which the largest connected component drops below 5% of original network size.
    
    All results are averaged over 20 Monte Carlo simulations per removal probability.
    """
    ax_notes.text(0.5, 0.5, notes_text, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', alpha=0.5))
    
    # Add main title
    plt.suptitle('Node Percolation Analysis: Network Model Comparison', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add descriptive subtitles for rows
    y_positions = [0.75, 0.5, 0.25]
    row_titles = [
        'Random Node Percolation: Network Robustness against Random Node Failures',
        'Degree-based Node Percolation: Network Robustness against Targeted Node Attacks',
        'Node Percolation Threshold Comparison'
    ]
    
    for y, title in zip(y_positions, row_titles):
        fig.text(0.5, y, title, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Save the figure
    output_path = os.path.join(FIGURES_DIR, 'node_percolation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nNode percolation visualization saved to {output_path}")
    
    # Show the plot
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()

def main():
    """Main function to run the script."""
    print("Starting node percolation analysis...")
    
    # Check if results already exist
    files_exist = True
    for network_type in NETWORKS:
        for model_type in MODEL_TYPES:
            for percolation_type in ['random', 'degree']:
                filename = f"{network_type}_{model_type}_{percolation_type}_node_percolation.csv"
                if not os.path.exists(os.path.join(NODE_PERC_DIR, filename)):
                    files_exist = False
                    break
    
    # Run analyses if results don't exist
    if not files_exist:
        print("Running node percolation analyses for all networks and models...")
        run_all_percolation_analyses()
    else:
        print("Using existing node percolation results...")
    
    # Create visualization
    print("Creating node percolation visualization...")
    create_node_percolation_visualization()
    
    print("Done!")

if __name__ == "__main__":
    main() 