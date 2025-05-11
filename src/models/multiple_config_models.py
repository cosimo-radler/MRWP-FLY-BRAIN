#!/usr/bin/env python3
"""
Multiple Configuration Models Generation Script

This script generates multiple instances of configuration models for:
1. Scaled configuration models (1500 nodes)
2. Unscaled configuration models (original node count)

Each model is saved as a separate GEXF file for later analysis.
"""

import os
import numpy as np
import networkx as nx
from tqdm import tqdm

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
SCALED_MODELS_DIR = os.path.join(CONFIG_MODELS_DIR, "multiple_scaled")
UNSCALED_MODELS_DIR = os.path.join(CONFIG_MODELS_DIR, "multiple_unscaled")

# Ensure directories exist
os.makedirs(SCALED_MODELS_DIR, exist_ok=True)
os.makedirs(UNSCALED_MODELS_DIR, exist_ok=True)

# Parameters
NUM_INSTANCES = 10  # Number of model instances to generate
SCALED_SIZE = 1500  # Size of scaled models

# Network types to process
NETWORK_TYPES = ['eb', 'fb', 'mb_kc']

def load_original_network(network_type):
    """Load original network from GEXF file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        
    Returns:
        NetworkX Graph
    """
    file_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
    try:
        G = nx.read_gexf(file_path)
        return G
    except Exception as e:
        print(f"Error loading network {file_path}: {e}")
        return None

def create_scaled_config_model(G_original, target_size=SCALED_SIZE):
    """Create a scaled configuration model.
    
    Args:
        G_original: Original NetworkX graph
        target_size: Target number of nodes
        
    Returns:
        NetworkX Graph of scaled configuration model
    """
    # Convert to undirected for degree sequence
    G_undirected = G_original.to_undirected()
    
    # Get degree sequence
    degree_sequence = [d for _, d in G_undirected.degree()]
    
    # Calculate scaling factors
    original_size = len(G_undirected)
    size_ratio = target_size / original_size
    
    # Scale the degree sequence
    # Each node's degree is scaled by the size ratio, and an edge requires two nodes
    scaled_degrees = []
    for degree in degree_sequence:
        scaled_degree = max(1, int(degree * np.sqrt(size_ratio)))  # Ensure minimum degree of 1
        scaled_degrees.append(scaled_degree)
    
    # Generate additional degrees for new nodes
    if target_size > original_size:
        # Sample from the existing distribution for the new nodes
        additional_degrees = np.random.choice(scaled_degrees, target_size - original_size)
        scaled_degrees.extend(additional_degrees)
    
    # Make sum of degrees even (required for configuration model)
    if sum(scaled_degrees) % 2 != 0:
        scaled_degrees[0] += 1
    
    # Create configuration model
    G_config = nx.configuration_model(scaled_degrees, seed=np.random.randint(10000))
    G_config = nx.Graph(G_config)  # Remove parallel edges
    G_config.remove_edges_from(nx.selfloop_edges(G_config))  # Remove self-loops
    
    return G_config

def create_unscaled_config_model(G_original):
    """Create an unscaled configuration model (same node count as original).
    
    Args:
        G_original: Original NetworkX graph
        
    Returns:
        NetworkX Graph of unscaled configuration model
    """
    # Convert to undirected for degree sequence
    G_undirected = G_original.to_undirected()
    
    # Get degree sequence
    degree_sequence = [d for _, d in G_undirected.degree()]
    
    # Make sum of degrees even (required for configuration model)
    if sum(degree_sequence) % 2 != 0:
        degree_sequence[0] += 1
    
    # Create configuration model
    G_config = nx.configuration_model(degree_sequence, seed=np.random.randint(10000))
    G_config = nx.Graph(G_config)  # Remove parallel edges
    G_config.remove_edges_from(nx.selfloop_edges(G_config))  # Remove self-loops
    
    return G_config

def generate_multiple_models():
    """Generate multiple instances of configuration models for all network types."""
    for network_type in NETWORK_TYPES:
        print(f"\nProcessing {network_type} network...")
        
        # Load original network
        G_original = load_original_network(network_type)
        if G_original is None:
            print(f"Skipping {network_type} network - could not load original.")
            continue
        
        print(f"Original network: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
        
        # Generate multiple scaled configuration models
        print(f"Generating {NUM_INSTANCES} scaled configuration models...")
        for i in tqdm(range(NUM_INSTANCES)):
            # Create scaled model
            G_scaled = create_scaled_config_model(G_original)
            
            # Save model
            output_path = os.path.join(SCALED_MODELS_DIR, f"{network_type}_scaled_config_model_{i+1:02d}.gexf")
            nx.write_gexf(G_scaled, output_path)
        
        # Generate multiple unscaled configuration models
        print(f"Generating {NUM_INSTANCES} unscaled configuration models...")
        for i in tqdm(range(NUM_INSTANCES)):
            # Create unscaled model
            G_unscaled = create_unscaled_config_model(G_original)
            
            # Save model
            output_path = os.path.join(UNSCALED_MODELS_DIR, f"{network_type}_unscaled_config_model_{i+1:02d}.gexf")
            nx.write_gexf(G_unscaled, output_path)

def main():
    """Main function."""
    print("Generating multiple configuration models...")
    generate_multiple_models()
    print("Done!")

if __name__ == "__main__":
    main() 