#!/usr/bin/env python3
"""
Upscaled Configuration Model Percolation Analysis for Drosophila Circuit Robustness

This script performs bond percolation analysis on the upscaled configuration models (3500 nodes)
of the ellipsoid-body (EB), fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from scipy.optimize import curve_fit
import warnings
import json

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/upscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Simulation parameters
NUM_SIMULATIONS = 20  # Number of Monte Carlo simulations per probability
NUM_PROB_STEPS = 21   # Number of probability steps from 0 to 1

# Ignore curve fitting warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    data_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    return nx.read_gexf(data_path)

def load_upscaled_config_model(network_type):
    """Load upscaled configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(UPSCALED_CONFIG_MODEL_DIR, f"{network_type}_upscaled_config_model.gexf"))

def run_bond_percolation_simulation(G, removal_probability, num_simulations=NUM_SIMULATIONS):
    """Run bond percolation simulation for a specific edge removal probability.
    
    Args:
        G: NetworkX graph
        removal_probability: Probability of removing an edge
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
        
        # Randomly remove edges with the given probability
        edges_to_remove = []
        for u, v in G_sim.edges():
            if np.random.random() < removal_probability:
                edges_to_remove.append((u, v))
        
        G_sim.remove_edges_from(edges_to_remove)
        
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
        'std_lcc_size': std_lcc,
        'lcc_sizes': lcc_sizes
    }

def process_probability(args):
    """Process a single removal probability (for parallel processing)."""
    G, removal_probability, num_simulations = args
    return run_bond_percolation_simulation(G, removal_probability, num_simulations)

def run_percolation_analysis(G, model_type, network_name, num_prob_steps=NUM_PROB_STEPS, num_simulations=NUM_SIMULATIONS):
    """Run bond percolation analysis across a range of removal probabilities.
    
    Args:
        G: NetworkX graph
        model_type: 'original' or 'upscaled_config_model'
        network_name: Name for the results (e.g., 'eb', 'fb', 'mb_kc')
        num_prob_steps: Number of probability steps to evaluate
        num_simulations: Number of simulations per probability value
        
    Returns:
        DataFrame with results for each probability
    """
    full_name = f"{network_name}_{model_type}"
    print(f"\nRunning bond percolation analysis for {full_name} network...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal probabilities
    removal_probabilities = np.linspace(0, 1, num_prob_steps)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Create arguments list for parallel processing
    args_list = [(G, p, num_simulations) for p in removal_probabilities]
    
    # Run simulations in parallel
    results = list(tqdm(
        pool.imap(process_probability, args_list),
        total=len(removal_probabilities),
        desc="Simulating edge removal"
    ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, f"{full_name}_percolation_results.csv"), index=False)
    
    return df

def percolation_model(x, qc, beta):
    """Percolation model function for curve fitting.
    
    P(q) ~ (qc - q)^beta for q < qc
    P(q) = 0 for q >= qc
    
    Args:
        x: Edge removal probability
        qc: Critical threshold
        beta: Critical exponent
        
    Returns:
        Largest connected component size according to model
    """
    y = np.zeros_like(x)
    below_qc = x < qc
    y[below_qc] = (qc - x[below_qc]) ** beta
    return y

def estimate_critical_threshold(df):
    """Estimate critical threshold and exponent from percolation data.
    
    Args:
        df: DataFrame with percolation results
        
    Returns:
        Dictionary with estimated parameters
    """
    x = df['removal_probability'].values
    y = df['mean_lcc_size'].values
    
    # Simple threshold estimate (where LCC size drops below 0.5)
    threshold_simple = 0
    for i, size in enumerate(y):
        if size < 0.5:
            if i > 0:
                threshold_simple = x[i-1] + (x[i] - x[i-1]) * (0.5 - y[i-1]) / (y[i] - y[i-1])
            else:
                threshold_simple = x[i]
            break
    
    # If we didn't cross 0.5, use the last probability
    if threshold_simple == 0 and len(y) > 0:
        threshold_simple = x[-1]
    
    # Try curve fitting for a more rigorous estimate
    try:
        # Initial parameter guess
        p0 = [threshold_simple, 0.5]
        
        # Only use data points where y > 0
        mask = y > 0
        if sum(mask) > 2:  # Need at least 3 points for fitting
            popt, _ = curve_fit(percolation_model, x[mask], y[mask], p0=p0, bounds=([0, 0], [1, 2]))
            threshold_fit, beta = popt
        else:
            threshold_fit, beta = threshold_simple, 0.5
    except:
        # Fallback if curve fitting fails
        threshold_fit, beta = threshold_simple, 0.5
    
    return {
        'threshold_simple': threshold_simple,
        'threshold_fit': threshold_fit,
        'beta': beta
    }

def plot_percolation_comparison(original_df, upscaled_df, network_type, original_params, upscaled_params):
    """Plot percolation analysis comparison between original and upscaled configuration model.
    
    Args:
        original_df: DataFrame with original network percolation results
        upscaled_df: DataFrame with upscaled configuration model percolation results
        network_type: Type of network for naming
        original_params: Dictionary with original network critical parameters
        upscaled_params: Dictionary with upscaled configuration model critical parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original network results
    plt.errorbar(
        original_df['removal_probability'], 
        original_df['mean_lcc_size'],
        yerr=original_df['std_lcc_size'],
        fmt='o-', 
        color='blue',
        alpha=0.7,
        label=f"Original Network (qc ≈ {original_params['threshold_simple']:.3f})"
    )
    
    # Plot upscaled configuration model results
    plt.errorbar(
        upscaled_df['removal_probability'], 
        upscaled_df['mean_lcc_size'],
        yerr=upscaled_df['std_lcc_size'],
        fmt='s-', 
        color='red',
        alpha=0.7,
        label=f"Upscaled Config Model (3500 nodes) (qc ≈ {upscaled_params['threshold_simple']:.3f})"
    )
    
    # Add vertical lines at critical thresholds
    plt.axvline(x=original_params['threshold_simple'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=upscaled_params['threshold_simple'], color='red', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title(f"{network_type.upper()} Network vs. Upscaled Configuration Model: Bond Percolation Analysis")
    plt.xlabel("Edge Removal Probability")
    plt.ylabel("Largest Connected Component Size (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, f"{network_type}_vs_upscaled_percolation.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved percolation comparison to {output_path}")
    plt.close()

def create_multi_comparison_plot(original_df, config_df, upscaled_df, network_type, 
                                 original_params, config_params, upscaled_params):
    """Create a comparison plot showing original, standard config model, and upscaled config model.
    
    Args:
        original_df: DataFrame with original network percolation results
        config_df: DataFrame with standard config model percolation results (1500 nodes)
        upscaled_df: DataFrame with upscaled config model percolation results (3500 nodes)
        network_type: Type of network for naming
        original_params: Dictionary with original network critical parameters
        config_params: Dictionary with config model critical parameters
        upscaled_params: Dictionary with upscaled config model critical parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original network results
    plt.errorbar(
        original_df['removal_probability'], 
        original_df['mean_lcc_size'],
        yerr=original_df['std_lcc_size'],
        fmt='o-', 
        color='blue',
        alpha=0.7,
        label=f"Original ({original_params['threshold_simple']:.3f})"
    )
    
    # Plot standard configuration model results (if available)
    if config_df is not None:
        plt.errorbar(
            config_df['removal_probability'], 
            config_df['mean_lcc_size'],
            yerr=config_df['std_lcc_size'],
            fmt='s-', 
            color='green',
            alpha=0.7,
            label=f"Config Model 1500 ({config_params['threshold_simple']:.3f})"
        )
    
    # Plot upscaled configuration model results
    plt.errorbar(
        upscaled_df['removal_probability'], 
        upscaled_df['mean_lcc_size'],
        yerr=upscaled_df['std_lcc_size'],
        fmt='d-', 
        color='red',
        alpha=0.7,
        label=f"Config Model 3500 ({upscaled_params['threshold_simple']:.3f})"
    )
    
    # Add vertical lines at critical thresholds
    plt.axvline(x=original_params['threshold_simple'], color='blue', linestyle='--', alpha=0.5)
    if config_df is not None:
        plt.axvline(x=config_params['threshold_simple'], color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=upscaled_params['threshold_simple'], color='red', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title(f"{network_type.upper()} Network: Bond Percolation Analysis Comparison")
    plt.xlabel("Edge Removal Probability")
    plt.ylabel("Largest Connected Component Size")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Network Type (qc value)")
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, f"{network_type}_multi_model_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved multi-model comparison to {output_path}")
    plt.close()

def create_comparison_summary(results):
    """Create a summary DataFrame comparing results across networks.
    
    Args:
        results: Dictionary with results for each network
        
    Returns:
        DataFrame with comparative summary
    """
    summary_data = []
    
    for network in results:
        network_results = results[network]
        
        # Original network
        original = {
            'network': network.upper(),
            'model': 'Original',
            'nodes': network_results['original']['nodes'],
            'edges': network_results['original']['edges'],
            'critical_threshold': network_results['original']['params']['threshold_simple'],
            'critical_exponent': network_results['original']['params']['beta'],
        }
        summary_data.append(original)
        
        # Upscaled configuration model
        upscaled = {
            'network': network.upper(),
            'model': 'Upscaled Config (3500)',
            'nodes': network_results['upscaled_config_model']['nodes'],
            'edges': network_results['upscaled_config_model']['edges'],
            'critical_threshold': network_results['upscaled_config_model']['params']['threshold_simple'],
            'critical_exponent': network_results['upscaled_config_model']['params']['beta'],
        }
        summary_data.append(upscaled)
    
    return pd.DataFrame(summary_data)

def plot_summary_comparison(summary_df):
    """Plot comparative summary of critical thresholds.
    
    Args:
        summary_df: DataFrame with comparative summary
    """
    plt.figure(figsize=(10, 6))
    
    # Group by network and model
    grouped = summary_df.groupby(['network', 'model'])['critical_threshold'].mean().unstack()
    
    # Plot as grouped bar chart
    ax = grouped.plot(kind='bar', width=0.7, alpha=0.7)
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    # Set title and labels
    plt.title("Critical Threshold Comparison Across Networks")
    plt.xlabel("Network")
    plt.ylabel("Critical Threshold (qc)")
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, "upscaled_critical_threshold_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved critical threshold comparison to {output_path}")
    plt.close()

def process_network(network_type):
    """Process a single network and its upscaled configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Dictionary with results
    """
    results = {
        'original': {},
        'upscaled_config_model': {}
    }
    
    # Load original network
    G_original = load_original_network(network_type)
    results['original']['nodes'] = G_original.number_of_nodes()
    results['original']['edges'] = G_original.number_of_edges()
    
    # Load upscaled configuration model
    G_upscaled = load_upscaled_config_model(network_type)
    results['upscaled_config_model']['nodes'] = G_upscaled.number_of_nodes()
    results['upscaled_config_model']['edges'] = G_upscaled.number_of_edges()
    
    # Run percolation analysis on original network
    print(f"\nProcessing original {network_type} network")
    orig_df = run_percolation_analysis(G_original, 'original', network_type)
    orig_params = estimate_critical_threshold(orig_df)
    results['original']['df'] = orig_df
    results['original']['params'] = orig_params
    
    # Run percolation analysis on upscaled configuration model
    print(f"\nProcessing upscaled configuration model of {network_type} network")
    upscaled_df = run_percolation_analysis(G_upscaled, 'upscaled_config_model', network_type)
    upscaled_params = estimate_critical_threshold(upscaled_df)
    results['upscaled_config_model']['df'] = upscaled_df
    results['upscaled_config_model']['params'] = upscaled_params
    
    # Plot comparison
    plot_percolation_comparison(orig_df, upscaled_df, network_type, orig_params, upscaled_params)
    
    # Try to load standard configuration model results
    try:
        config_df = pd.read_csv(os.path.join(RESULTS_DIR, f"{network_type}_config_model_percolation_results.csv"))
        # Load parameters
        with open(os.path.join(RESULTS_DIR, f"{network_type}_config_model_percolation_params.json"), 'r') as f:
            config_params = json.load(f)
        
        # Create multi-model comparison
        create_multi_comparison_plot(orig_df, config_df, upscaled_df, network_type, 
                                    orig_params, config_params, upscaled_params)
    except Exception as e:
        print(f"Could not create multi-model comparison: {e}")
        create_multi_comparison_plot(orig_df, None, upscaled_df, network_type, 
                                    orig_params, None, upscaled_params)
    
    return results

def main():
    """Main function to process all networks."""
    results = {}
    
    # Process EB network and its upscaled configuration model
    results['eb'] = process_network('eb')
    
    # Process FB network and its upscaled configuration model
    results['fb'] = process_network('fb')
    
    # Process MB-KC network and its upscaled configuration model
    results['mb_kc'] = process_network('mb_kc')
    
    # Create summary comparison
    summary_df = create_comparison_summary(results)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "upscaled_percolation_comparison_summary.csv"), index=False)
    
    # Plot summary comparison
    plot_summary_comparison(summary_df)
    
    print(f"\nSummary of results:\n{summary_df}")
    print(f"\nSaved summary to {os.path.join(RESULTS_DIR, 'upscaled_percolation_comparison_summary.csv')}")
    print(f"Saved comparison plots to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 