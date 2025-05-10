#!/usr/bin/env python3
"""
Bond Percolation Analysis Script for Drosophila Circuit Robustness Analysis

This script performs bond percolation analysis on the ellipsoid-body (EB), 
fan-shaped-body (FB), and mushroom-body (MB) Kenyon-cell subnetworks to 
determine their critical thresholds and compare their robustness to random edge removal.
"""

import os
import json
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Ensure results and figures directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Number of Monte Carlo simulations per probability value
NUM_SIMULATIONS = 20

# Number of probability steps to evaluate
NUM_PROB_STEPS = 50

def load_network(network_type):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    return nx.read_gexf(os.path.join(DATA_DIR, f"{network_type}_network.gexf"))

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
    # (This is standard for percolation analysis as we're interested in connected components)
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

def process_probability(G, removal_probability, num_simulations):
    """Process a single removal probability (for parallel processing)."""
    return run_bond_percolation_simulation(G, removal_probability, num_simulations)

def run_percolation_analysis(G, network_name, num_prob_steps=NUM_PROB_STEPS, num_simulations=NUM_SIMULATIONS):
    """Run bond percolation analysis across a range of removal probabilities.
    
    Args:
        G: NetworkX graph
        network_name: Name for the results
        num_prob_steps: Number of probability steps to evaluate
        num_simulations: Number of simulations per probability value
        
    Returns:
        DataFrame with results for each probability
    """
    print(f"\nRunning bond percolation analysis for {network_name} network...")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate removal probabilities
    removal_probabilities = np.linspace(0, 1, num_prob_steps)
    
    # Use parallel processing to speed up simulations
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Keep one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Create partial function with fixed arguments
    func = partial(process_probability, G, num_simulations=num_simulations)
    
    # Run simulations in parallel
    results = list(tqdm(
        pool.imap(func, removal_probabilities),
        total=len(removal_probabilities),
        desc="Simulating edge removal"
    ))
    
    pool.close()
    pool.join()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, f"{network_name.lower().replace(' ', '_')}_percolation_results.csv"), index=False)
    
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

def plot_percolation_results(eb_df, fb_df, mb_kc_df, eb_params, fb_params, mb_kc_params):
    """Plot percolation analysis results for all three networks.
    
    Args:
        eb_df: DataFrame with EB percolation results
        fb_df: DataFrame with FB percolation results
        mb_kc_df: DataFrame with MB-KC percolation results
        eb_params: Dictionary with EB critical parameters
        fb_params: Dictionary with FB critical parameters
        mb_kc_params: Dictionary with MB-KC critical parameters
    """
    plt.figure(figsize=(10, 6))
    
    # Plot EB results
    plt.errorbar(
        eb_df['removal_probability'], 
        eb_df['mean_lcc_size'],
        yerr=eb_df['std_lcc_size'],
        fmt='o-', 
        color='blue',
        alpha=0.7,
        label=f"EB Network (qc ≈ {eb_params['threshold_simple']:.3f})"
    )
    
    # Plot FB results
    plt.errorbar(
        fb_df['removal_probability'], 
        fb_df['mean_lcc_size'],
        yerr=fb_df['std_lcc_size'],
        fmt='o-', 
        color='green',
        alpha=0.7,
        label=f"FB Network (qc ≈ {fb_params['threshold_simple']:.3f})"
    )
    
    # Plot MB-KC results
    plt.errorbar(
        mb_kc_df['removal_probability'], 
        mb_kc_df['mean_lcc_size'],
        yerr=mb_kc_df['std_lcc_size'],
        fmt='o-', 
        color='red',
        alpha=0.7,
        label=f"MB-KC Network (qc ≈ {mb_kc_params['threshold_simple']:.3f})"
    )
    
    # Add vertical lines for critical thresholds
    plt.axvline(x=eb_params['threshold_simple'], color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=fb_params['threshold_simple'], color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=mb_kc_params['threshold_simple'], color='red', linestyle='--', alpha=0.5)
    
    # Add robustness indices to the legend
    eb_ri = eb_params['robustness_index']
    fb_ri = fb_params['robustness_index']
    mb_kc_ri = mb_kc_params['robustness_index']
    plt.plot([], [], ' ', label=f"EB Robustness: {eb_ri:.3f}")
    plt.plot([], [], ' ', label=f"FB Robustness: {fb_ri:.3f}")
    plt.plot([], [], ' ', label=f"MB-KC Robustness: {mb_kc_ri:.3f}")
    
    plt.xlabel('Edge Removal Probability (p)')
    plt.ylabel('Normalized Largest Connected Component Size')
    plt.title('Bond Percolation Analysis: Network Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, "percolation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap visualization of the LCC size distribution
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # EB Heatmap
    sns.heatmap(
        np.vstack(eb_df['lcc_sizes'].apply(lambda x: np.array(x)).values).T,
        ax=ax1,
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized LCC Size'}
    )
    ax1.set_title('Ellipsoid Body Network: Distribution of LCC Sizes Across Simulations')
    ax1.set_ylabel('Simulation Number')
    
    # FB Heatmap
    sns.heatmap(
        np.vstack(fb_df['lcc_sizes'].apply(lambda x: np.array(x)).values).T,
        ax=ax2,
        cmap='Greens',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized LCC Size'}
    )
    ax2.set_title('Fan-shaped Body Network: Distribution of LCC Sizes Across Simulations')
    ax2.set_ylabel('Simulation Number')
    
    # MB-KC Heatmap
    sns.heatmap(
        np.vstack(mb_kc_df['lcc_sizes'].apply(lambda x: np.array(x)).values).T,
        ax=ax3,
        cmap='Reds',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized LCC Size'}
    )
    ax3.set_title('Mushroom Body Kenyon Cell Network: Distribution of LCC Sizes Across Simulations')
    ax3.set_xlabel('Edge Removal Probability Index')
    ax3.set_ylabel('Simulation Number')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "percolation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    print("Running Bond Percolation Analysis...")
    
    # Load networks
    print("\nLoading networks...")
    eb_network = load_network('eb')
    fb_network = load_network('fb')
    mb_kc_network = load_network('mb_kc')
    
    # Run percolation analysis
    eb_results = run_percolation_analysis(eb_network, "Ellipsoid Body")
    fb_results = run_percolation_analysis(fb_network, "Fan-shaped Body")
    mb_kc_results = run_percolation_analysis(mb_kc_network, "Mushroom Body Kenyon Cell")
    
    # Estimate critical thresholds
    print("\nEstimating critical thresholds...")
    eb_params = estimate_critical_threshold(eb_results)
    fb_params = estimate_critical_threshold(fb_results)
    mb_kc_params = estimate_critical_threshold(mb_kc_results)
    
    # Print results
    print("\nResults:")
    print(f"Ellipsoid Body Network:")
    print(f"  Critical Threshold (simple): {eb_params['threshold_simple']:.4f}")
    if eb_params['threshold_fitted'] is not None:
        print(f"  Critical Threshold (fitted): {eb_params['threshold_fitted']:.4f}")
        print(f"  Critical Exponent (beta): {eb_params['beta']:.4f}")
    print(f"  Robustness Index: {eb_params['robustness_index']:.4f}")
    
    print(f"\nFan-shaped Body Network:")
    print(f"  Critical Threshold (simple): {fb_params['threshold_simple']:.4f}")
    if fb_params['threshold_fitted'] is not None:
        print(f"  Critical Threshold (fitted): {fb_params['threshold_fitted']:.4f}")
        print(f"  Critical Exponent (beta): {fb_params['beta']:.4f}")
    print(f"  Robustness Index: {fb_params['robustness_index']:.4f}")
    
    print(f"\nMushroom Body Kenyon Cell Network:")
    print(f"  Critical Threshold (simple): {mb_kc_params['threshold_simple']:.4f}")
    if mb_kc_params['threshold_fitted'] is not None:
        print(f"  Critical Threshold (fitted): {mb_kc_params['threshold_fitted']:.4f}")
        print(f"  Critical Exponent (beta): {mb_kc_params['beta']:.4f}")
    print(f"  Robustness Index: {mb_kc_params['robustness_index']:.4f}")
    
    # Save parameters
    params = {
        'Ellipsoid_Body': eb_params,
        'Fan_Shaped_Body': fb_params,
        'Mushroom_Body_Kenyon_Cell': mb_kc_params
    }
    with open(os.path.join(RESULTS_DIR, "critical_parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Plot results
    print("\nPlotting results...")
    plot_percolation_results(eb_results, fb_results, mb_kc_results, eb_params, fb_params, mb_kc_params)
    
    print("\nPercolation analysis complete!")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main() 