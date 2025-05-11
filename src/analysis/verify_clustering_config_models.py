#!/usr/bin/env python3
"""
Verify Clustering-Preserved Configuration Models

This script loads the original networks and the clustering-preserved configuration
models to verify that they have similar clustering coefficients.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GEPHI_REAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gephi Graphs/real_models")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
CLUSTERING_CONFIG_MODEL_DIR = os.path.join(CONFIG_MODEL_DIR, "clustering")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    # Try to load from the Gephi real models directory first, then fall back to the data directory
    gephi_path = os.path.join(GEPHI_REAL_MODELS_DIR, f"{network_type}_network.gexf")
    data_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    
    if os.path.exists(gephi_path):
        return nx.read_gexf(gephi_path)
    else:
        return nx.read_gexf(data_path)

def load_config_model(network_type, scale_type="unscaled", model_type="clustering"):
    """Load configuration model from GEXF or pickle file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        scale_type: "unscaled" or "scaled"
        model_type: "clustering" for clustering-preserved config model, or "regular" for standard config model
        
    Returns:
        NetworkX DiGraph
    """
    if model_type == "clustering":
        model_dir = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, scale_type)
        pickle_path = os.path.join(model_dir, f"{network_type}_{scale_type}_clustering_config_model.pkl")
        gexf_path = os.path.join(model_dir, f"{network_type}_{scale_type}_clustering_config_model.gexf")
    else:
        model_dir = CONFIG_MODEL_DIR
        if scale_type == "unscaled":
            gexf_path = os.path.join(model_dir, "unscaled", f"{network_type}_unscaled_config_model.gexf")
        else:
            gexf_path = os.path.join(model_dir, f"{network_type}_config_model.gexf")
        pickle_path = None
    
    # Try to load from pickle first (faster)
    if pickle_path and os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle: {e}")
    
    # Fall back to GEXF
    if os.path.exists(gexf_path):
        return nx.read_gexf(gexf_path)
    
    print(f"Warning: Could not find model file for {network_type} ({scale_type}, {model_type})")
    return None

def calculate_network_metrics(G, name="Unknown"):
    """Calculate key network metrics.
    
    Args:
        G: NetworkX graph
        name: Name for identification
        
    Returns:
        Dictionary of metrics
    """
    if G is None:
        return {
            "name": name,
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "avg_degree": 0,
            "avg_clustering": 0,
        }
    
    # Convert to undirected for some metrics
    G_undirected = G.to_undirected()
    
    metrics = {
        "name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "avg_clustering": nx.average_clustering(G_undirected),
    }
    
    return metrics

def compare_models(network_type, scale_type="unscaled"):
    """Compare original network with configuration models.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        scale_type: "unscaled" or "scaled"
        
    Returns:
        DataFrame with comparison metrics
    """
    # Load networks
    G_original = load_original_network(network_type)
    G_clustering = load_config_model(network_type, scale_type, "clustering")
    G_standard = load_config_model(network_type, scale_type, "regular")
    
    # Calculate metrics
    original_metrics = calculate_network_metrics(G_original, f"Original {network_type.upper()}")
    clustering_metrics = calculate_network_metrics(G_clustering, f"Clustering Config ({scale_type})")
    standard_metrics = calculate_network_metrics(G_standard, f"Standard Config ({scale_type})")
    
    # Create and return DataFrame
    metrics_df = pd.DataFrame([original_metrics, clustering_metrics, standard_metrics])
    return metrics_df

def plot_clustering_comparison(network_type, scale_type="unscaled"):
    """Plot clustering coefficient comparison between original and configuration models.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, or 'mb_kc' for Mushroom Body Kenyon Cells
        scale_type: "unscaled" or "scaled"
    """
    # Load networks
    G_original = load_original_network(network_type)
    G_clustering = load_config_model(network_type, scale_type, "clustering")
    G_standard = load_config_model(network_type, scale_type, "regular")
    
    if G_original is None or G_clustering is None:
        print(f"Error: Could not load all required networks for {network_type} ({scale_type})")
        return
    
    # Convert to undirected for clustering calculations
    G_original_undirected = G_original.to_undirected()
    G_clustering_undirected = G_clustering.to_undirected()
    
    # Calculate node-level clustering
    original_clustering = nx.clustering(G_original_undirected)
    clustering_config_clustering = nx.clustering(G_clustering_undirected)
    
    # Calculate global clustering coefficients
    original_gcc = nx.average_clustering(G_original_undirected)
    clustering_config_gcc = nx.average_clustering(G_clustering_undirected)
    
    standard_config_gcc = 0
    if G_standard is not None:
        G_standard_undirected = G_standard.to_undirected()
        standard_config_clustering = nx.clustering(G_standard_undirected)
        standard_config_gcc = nx.average_clustering(G_standard_undirected)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Distribution of clustering coefficients
    ax1.hist(list(original_clustering.values()), bins=20, alpha=0.7, 
            label=f"Original (GCC={original_gcc:.4f})")
    ax1.hist(list(clustering_config_clustering.values()), bins=20, alpha=0.7, 
            label=f"Clustering Config (GCC={clustering_config_gcc:.4f})")
    
    if G_standard is not None:
        ax1.hist(list(standard_config_clustering.values()), bins=20, alpha=0.7, 
                label=f"Standard Config (GCC={standard_config_gcc:.4f})")
    
    ax1.set_title(f"{network_type.upper()} Network: Clustering Coefficient Distribution ({scale_type})")
    ax1.set_xlabel("Node Clustering Coefficient")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Degree vs Clustering
    original_degree = dict(G_original_undirected.degree())
    clustering_degree = dict(G_clustering_undirected.degree())
    
    original_data = [(d, original_clustering[n]) for n, d in original_degree.items()]
    clustering_data = [(d, clustering_config_clustering[n]) for n, d in clustering_degree.items()]
    
    original_x, original_y = zip(*original_data) if original_data else ([], [])
    clustering_x, clustering_y = zip(*clustering_data) if clustering_data else ([], [])
    
    ax2.scatter(original_x, original_y, alpha=0.7, label="Original")
    ax2.scatter(clustering_x, clustering_y, alpha=0.7, label="Clustering Config")
    
    if G_standard is not None:
        standard_degree = dict(G_standard_undirected.degree())
        standard_data = [(d, standard_config_clustering[n]) for n, d in standard_degree.items()]
        standard_x, standard_y = zip(*standard_data) if standard_data else ([], [])
        ax2.scatter(standard_x, standard_y, alpha=0.7, label="Standard Config")
    
    ax2.set_title(f"{network_type.upper()} Network: Degree vs Clustering ({scale_type})")
    ax2.set_xlabel("Node Degree")
    ax2.set_ylabel("Clustering Coefficient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{network_type.lower()}_{scale_type}_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to verify all network models."""
    print("Verifying clustering-preserved configuration models...")
    
    # Create empty DataFrame to store all results
    all_results = pd.DataFrame()
    
    # Process each network type and scale
    for network_type in ['eb', 'fb', 'mb_kc']:
        for scale_type in ['unscaled', 'scaled']:
            print(f"\nProcessing {network_type.upper()} network ({scale_type})...")
            
            # Compare models and collect metrics
            model_comparison = compare_models(network_type, scale_type)
            all_results = pd.concat([all_results, model_comparison])
            
            # Create visualization
            plot_clustering_comparison(network_type, scale_type)
    
    # Save combined results
    results_path = os.path.join(RESULTS_DIR, "clustering_model_comparison.csv")
    all_results.to_csv(results_path)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\nClustering Coefficient Comparison:")
    print(all_results[['name', 'avg_clustering']])

if __name__ == "__main__":
    main() 