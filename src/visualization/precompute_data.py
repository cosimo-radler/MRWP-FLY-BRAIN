#!/usr/bin/env python3
"""
Data Precomputation Script for Network Comparison Dashboard

This script precomputes all data needed for the dashboard and stores it to disk.
It calculates and saves:
- Degree distributions
- Network metrics
- Percolation and attack thresholds

Run this script before using the dashboard to ensure all data is ready for instant visualization.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import json
from collections import Counter
import time

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "unscaled")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "upscaled")
CLUSTERING_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "clustering")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Create precomputed data directory if it doesn't exist
PRECOMPUTED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precomputed_data")
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

# Network types and their display names
NETWORKS = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body KC'
}

# Model types and their display names
MODEL_TYPES = ['original', 'scaled_config', 'unscaled_config', 'upscaled_config', 'clustering_config', 'unscaled_clustering_config']

# Helper functions adapted from the dashboard script
def load_network(network_type, model_type='original'):
    """Load network from GEXF file."""
    if model_type == 'original':
        file_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    elif model_type == 'scaled_config':
        file_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(UNSCALED_CONFIG_MODEL_DIR, f"{network_type}_unscaled_config_model.gexf")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(UPSCALED_CONFIG_MODEL_DIR, f"{network_type}_upscaled_config_model.gexf")
    elif model_type == 'clustering_config':
        file_path = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, "scaled", f"{network_type}_scaled_clustering_config_model.gexf")
    elif model_type == 'unscaled_clustering_config':
        file_path = os.path.join(CLUSTERING_CONFIG_MODEL_DIR, "unscaled", f"{network_type}_unscaled_clustering_config_model.gexf")
    else:
        return None
    
    try:
        G = nx.read_gexf(file_path)
        return G
    except Exception as e:
        print(f"Error loading network {file_path}: {e}")
        return None

def get_normalized_degree_distribution(G):
    """Get normalized degree distribution of a graph."""
    if G is None:
        return [], []
        
    # Convert to undirected for consistent degree calculation
    G_undirected = G.to_undirected()
    
    # Get degrees and their frequencies
    degrees = [d for n, d in G_undirected.degree()]
    degree_count = Counter(degrees)
    
    # Sort by degree
    sorted_degrees = sorted(degree_count.keys())
    frequencies = [degree_count[d] / G_undirected.number_of_nodes() for d in sorted_degrees]
    
    return sorted_degrees, frequencies

def load_percolation_results(network_type, model_type='original'):
    """Load random percolation results."""
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # For random percolation, use percolation results
    if model_type == 'original':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_percolation_results.csv")
    elif model_type == 'scaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_config_model_percolation_results.csv")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_unscaled_config_model_percolation_results.csv")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_upscaled_config_model_percolation_results.csv")
    elif model_type == 'clustering_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_clustering_config_model_percolation_results.csv")
    elif model_type == 'unscaled_clustering_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_unscaled_clustering_config_model_percolation_results.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'removal_fraction' in df.columns:
            df['removal_probability'] = df['removal_fraction']
        if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
            df['mean_lcc_size'] = df['lcc_size']
            
        return df
    except Exception as e:
        print(f"Error loading percolation results {file_path}: {e}")
        return None

def load_attack_results(network_type, attack_strategy, model_type='original'):
    """Load targeted attack results."""
    network_names = {
        'eb': 'ellipsoid_body',
        'fb': 'fan-shaped_body',
        'mb_kc': 'mushroom_body_kenyon_cell'
    }
    full_name = network_names.get(network_type, network_type)
    
    # For targeted attacks, use attack results
    if model_type == 'original':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_{attack_strategy}_attack_results.csv")
    elif model_type == 'scaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_config_{attack_strategy}_attack_results.csv")
    elif model_type == 'unscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{full_name}_unscaled_config_{attack_strategy}_attack_results.csv")
    elif model_type == 'upscaled_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_upscaled_config_model_{attack_strategy}_attack_results.csv")
    elif model_type == 'clustering_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_clustering_config_model_{attack_strategy}_attack_results.csv")
    elif model_type == 'unscaled_clustering_config':
        file_path = os.path.join(RESULTS_DIR, f"{network_type}_unscaled_clustering_config_model_{attack_strategy}_attack_results.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'removal_fraction' in df.columns:
            df['removal_probability'] = df['removal_fraction']
        if 'fraction_removed' in df.columns:
            df['removal_probability'] = df['fraction_removed']
        if 'lcc_size' in df.columns and 'mean_lcc_size' not in df.columns:
            df['mean_lcc_size'] = df['lcc_size']
            
        return df
    except Exception as e:
        print(f"Error loading attack results {file_path}: {e}")
        return None

def calculate_network_metrics(G):
    """Calculate key network metrics for a graph."""
    if G is None:
        return None
    
    # Convert to undirected for consistent calculations
    G_undirected = G.to_undirected()
    
    # Basic metrics
    n_nodes = G_undirected.number_of_nodes()
    n_edges = G_undirected.number_of_edges()
    
    # Average degree
    avg_degree = 2 * n_edges / n_nodes
    
    try:
        # Clustering coefficient - measures triangle density
        clustering = nx.average_clustering(G_undirected)
    except:
        clustering = np.nan
    
    try:
        # Path length metrics
        if nx.is_connected(G_undirected):
            avg_path_length = nx.average_shortest_path_length(G_undirected)
            diameter = nx.diameter(G_undirected)
        else:
            # If graph is not connected, calculate for largest component
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            largest_subnet = G_undirected.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(largest_subnet)
            diameter = nx.diameter(largest_subnet)
    except:
        avg_path_length = np.nan
        diameter = np.nan
    
    # Return metrics dictionary
    return {
        'Nodes': n_nodes,
        'Edges': n_edges,
        'Avg Degree': round(avg_degree, 2),
        'Clustering': round(clustering, 3),
        'Avg Path Length': round(avg_path_length, 2) if not np.isnan(avg_path_length) else 'N/A',
        'Diameter': diameter if not np.isnan(diameter) else 'N/A'
    }

def extract_percolation_threshold(network_type, model_type='original'):
    """Extract percolation threshold from results."""
    df = load_percolation_results(network_type, model_type)
    
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

def extract_attack_threshold(network_type, attack_strategy, model_type='original'):
    """Extract attack threshold from results."""
    df = load_attack_results(network_type, attack_strategy, model_type)
    
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

def precompute_all_data():
    """Precompute all data needed for the dashboard."""
    start_time = time.time()
    
    # Initialize data stores
    degree_distributions = {}
    percolation_results = {}
    attack_results = {'degree': {}, 'betweenness': {}}
    network_metrics = {}
    
    # Process each network type and model type
    for network_type in NETWORKS.keys():
        print(f"Processing {NETWORKS[network_type]}...")
        
        # Initialize network level stores
        degree_distributions[network_type] = {}
        percolation_results[network_type] = {}
        attack_results['degree'][network_type] = {}
        attack_results['betweenness'][network_type] = {}
        network_metrics[network_type] = {}
        
        for model_type in MODEL_TYPES:
            print(f"  - Model: {model_type}")
            
            # Load network
            G = load_network(network_type, model_type)
            
            if G is not None:
                # Get degree distribution
                degrees, frequencies = get_normalized_degree_distribution(G)
                degree_distributions[network_type][model_type] = {
                    'degrees': degrees,
                    'frequencies': frequencies
                }
                
                # Calculate network metrics
                metrics = calculate_network_metrics(G)
                
                if metrics:
                    # Add percolation threshold
                    perc_threshold = extract_percolation_threshold(network_type, model_type)
                    metrics['Percolation Threshold'] = perc_threshold
                    
                    # Add attack thresholds
                    deg_threshold = extract_attack_threshold(network_type, 'degree', model_type)
                    metrics['Degree Attack Threshold'] = deg_threshold
                    
                    bet_threshold = extract_attack_threshold(network_type, 'betweenness', model_type)
                    metrics['Betweenness Attack Threshold'] = bet_threshold
                    
                    network_metrics[network_type][model_type] = metrics
            
            # Load percolation results
            df_perc = load_percolation_results(network_type, model_type)
            if df_perc is not None:
                percolation_results[network_type][model_type] = {
                    'removal_probability': df_perc['removal_probability'].tolist(),
                    'mean_lcc_size': df_perc['mean_lcc_size'].tolist()
                }
            
            # Load attack results
            for attack in ['degree', 'betweenness']:
                df_attack = load_attack_results(network_type, attack, model_type)
                if df_attack is not None:
                    attack_results[attack][network_type][model_type] = {
                        'removal_probability': df_attack['removal_probability'].tolist(),
                        'mean_lcc_size': df_attack['mean_lcc_size'].tolist()
                    }
    
    # Save all precomputed data
    with open(os.path.join(PRECOMPUTED_DIR, 'degree_distributions.pkl'), 'wb') as f:
        pickle.dump(degree_distributions, f)
    
    with open(os.path.join(PRECOMPUTED_DIR, 'percolation_results.pkl'), 'wb') as f:
        pickle.dump(percolation_results, f)
    
    with open(os.path.join(PRECOMPUTED_DIR, 'attack_results.pkl'), 'wb') as f:
        pickle.dump(attack_results, f)
    
    with open(os.path.join(PRECOMPUTED_DIR, 'network_metrics.pkl'), 'wb') as f:
        pickle.dump(network_metrics, f)
    
    # Also save metrics as JSON for potential external use
    # Convert any non-JSON serializable values first
    for network_type in network_metrics:
        for model_type in network_metrics[network_type]:
            for key, value in network_metrics[network_type][model_type].items():
                if isinstance(value, np.float64) or isinstance(value, np.float32):
                    network_metrics[network_type][model_type][key] = float(value)
                elif isinstance(value, np.int64) or isinstance(value, np.int32):
                    network_metrics[network_type][model_type][key] = int(value)
    
    with open(os.path.join(PRECOMPUTED_DIR, 'network_metrics.json'), 'w') as f:
        json.dump(network_metrics, f, indent=2)
    
    end_time = time.time()
    print(f"All data precomputed and saved in {end_time - start_time:.2f} seconds!")
    print(f"Data stored in: {PRECOMPUTED_DIR}")

if __name__ == "__main__":
    precompute_all_data() 