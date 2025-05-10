#!/usr/bin/env python3
"""
Unscaled Configuration Model Percolation Analysis Script

This script performs bond percolation analysis on the unscaled configuration models
of the ellipsoid-body (EB), fan-shaped-body (FB), and mushroom-body (MB) 
Kenyon-cell subnetworks, and compares their robustness with the original networks.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import multiprocessing
from functools import partial

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models/unscaled")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of Monte Carlo simulations per probability value
NUM_SIMULATIONS = 20

# Number of probability steps to evaluate
NUM_PROB_STEPS = 50

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))

def load_unscaled_config_model(network_type):
    """Load unscaled configuration model from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf"))

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
        model_type: 'original', 'config_model', or 'unscaled_config_model'
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
    """Estimate the critical threshold from percolation results.
    
    Args:
        df: DataFrame with percolation results
        
    Returns:
        Dictionary with critical threshold and other parameters
    """
    x = df['removal_probability'].values
    y = df['mean_lcc_size'].values
    
    # Method 1: Simple interpolation to find where LCC size approaches zero
    threshold1 = None
    for i in range(len(y) - 1):
        if y[i] > 0.05 and y[i+1] < 0.05:  # Threshold at 5% connectivity
            # Linear interpolation
            threshold1 = x[i] + (x[i+1] - x[i]) * (0.05 - y[i]) / (y[i+1] - y[i])
            break
    
    if threshold1 is None and y[-1] > 0.05:
        threshold1 = 1.0  # If never drops below threshold
    elif threshold1 is None:
        threshold1 = 0.0  # If already below threshold at start
    
    # Method 2: Curve fitting using percolation model
    try:
        # Initial parameter guess
        p0 = [0.5, 0.5]  # [qc, beta]
        
        # Curve fitting with bounds
        popt, _ = curve_fit(
            percolation_model, 
            x, y, 
            p0=p0,
            bounds=([0, 0], [1, 2])
        )
        
        qc, beta = popt
        
        # Calculate robustness index (area under LCC curve)
        robustness_index = np.trapz(y, x)
        
        return {
            'threshold_simple': threshold1,
            'threshold_fitted': qc,
            'beta': beta,
            'robustness_index': robustness_index
        }
    except Exception as e:
        print(f"Warning: Curve fitting failed: {e}")
        return {
            'threshold_simple': threshold1,
            'threshold_fitted': None,
            'beta': None,
            'robustness_index': np.trapz(y, x)
        }

def plot_percolation_comparison(original_df, unscaled_df, network_type, original_params, unscaled_params):
    """Plot percolation analysis comparison between original and unscaled configuration model.
    
    Args:
        original_df: DataFrame with original network percolation results
        unscaled_df: DataFrame with unscaled configuration model percolation results
        network_type: Type of network for naming
        original_params: Dictionary with original network critical parameters
        unscaled_params: Dictionary with unscaled configuration model critical parameters
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
    
    # Plot unscaled configuration model results
    plt.errorbar(
        unscaled_df['removal_probability'], 
        unscaled_df['mean_lcc_size'],
        yerr=unscaled_df['std_lcc_size'],
        fmt='s-', 
        color='green',
        alpha=0.7,
        label=f"Unscaled Config Model (qc ≈ {unscaled_params['threshold_simple']:.3f})"
    )
    
    # Add vertical lines at critical thresholds
    plt.axvline(x=original_params['threshold_simple'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=unscaled_params['threshold_simple'], color='green', linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title(f"{network_type.upper()} Network vs. Unscaled Config Model: Bond Percolation Analysis")
    plt.xlabel("Edge Removal Probability")
    plt.ylabel("Largest Connected Component Size (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with robustness index
    plt.text(
        0.05, 0.15,
        f"Robustness Index (Original): {original_params['robustness_index']:.3f}\n"
        f"Robustness Index (Unscaled Config): {unscaled_params['robustness_index']:.3f}",
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_unscaled_config_percolation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_summary(results):
    """Create a summary DataFrame of the comparison results.
    
    Args:
        results: Dictionary of results
        
    Returns:
        DataFrame with comparison summary
    """
    summary_data = []
    
    for network_type in results:
        original_params = results[network_type]['original']['params']
        unscaled_params = results[network_type]['unscaled_config_model']['params']
        
        summary_data.append({
            'network_type': network_type,
            'model_type': 'original',
            'nodes': results[network_type]['original']['nodes'],
            'edges': results[network_type]['original']['edges'],
            'threshold_simple': original_params['threshold_simple'],
            'threshold_fitted': original_params['threshold_fitted'],
            'beta': original_params['beta'],
            'robustness_index': original_params['robustness_index']
        })
        
        summary_data.append({
            'network_type': network_type,
            'model_type': 'unscaled_config_model',
            'nodes': results[network_type]['unscaled_config_model']['nodes'],
            'edges': results[network_type]['unscaled_config_model']['edges'],
            'threshold_simple': unscaled_params['threshold_simple'],
            'threshold_fitted': unscaled_params['threshold_fitted'],
            'beta': unscaled_params['beta'],
            'robustness_index': unscaled_params['robustness_index']
        })
    
    return pd.DataFrame(summary_data)

def process_network(network_type):
    """Process a single network and its unscaled configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
        
    Returns:
        Dictionary with results
    """
    results = {
        'original': {},
        'unscaled_config_model': {}
    }
    
    # Load original network
    G_original = load_original_network(network_type)
    results['original']['nodes'] = G_original.number_of_nodes()
    results['original']['edges'] = G_original.number_of_edges()
    
    # Load unscaled configuration model
    G_unscaled = load_unscaled_config_model(network_type)
    results['unscaled_config_model']['nodes'] = G_unscaled.number_of_nodes()
    results['unscaled_config_model']['edges'] = G_unscaled.number_of_edges()
    
    # Run percolation analysis on original network
    print(f"\nProcessing original {network_type} network")
    orig_df = run_percolation_analysis(G_original, 'original', network_type)
    orig_params = estimate_critical_threshold(orig_df)
    results['original']['df'] = orig_df
    results['original']['params'] = orig_params
    
    # Run percolation analysis on unscaled configuration model
    print(f"\nProcessing unscaled configuration model of {network_type} network")
    unscaled_df = run_percolation_analysis(G_unscaled, 'unscaled_config_model', network_type)
    unscaled_params = estimate_critical_threshold(unscaled_df)
    results['unscaled_config_model']['df'] = unscaled_df
    results['unscaled_config_model']['params'] = unscaled_params
    
    # Plot comparison
    plot_percolation_comparison(orig_df, unscaled_df, network_type, orig_params, unscaled_params)
    
    return results

def main():
    """Main function to process all networks."""
    results = {}
    
    # Process EB network and its unscaled configuration model
    results['eb'] = process_network('eb')
    
    # Process FB network and its unscaled configuration model
    results['fb'] = process_network('fb')
    
    # Process MB-KC network and its unscaled configuration model
    results['mb_kc'] = process_network('mb_kc')
    
    # Create summary comparison
    summary_df = create_comparison_summary(results)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "unscaled_percolation_comparison_summary.csv"), index=False)
    
    print(f"\nSummary of results:\n{summary_df}")
    print(f"\nSaved summary to {os.path.join(RESULTS_DIR, 'unscaled_percolation_comparison_summary.csv')}")
    print(f"Saved comparison plots to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 