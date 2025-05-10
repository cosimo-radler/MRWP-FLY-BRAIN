#!/usr/bin/env python3
"""
Export Enhanced Configuration Models

This script enhances the configuration model GEXF files with additional node attributes
for better visualization in Gephi, including degree, betweenness centrality, and clustering.
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Constants
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
GEPHI_CONFIG_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gephi Graphs/config_models")
ENHANCED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures/enhanced_models")

# Ensure directories exist
os.makedirs(CONFIG_MODEL_DIR, exist_ok=True)
os.makedirs(GEPHI_CONFIG_MODELS_DIR, exist_ok=True)
os.makedirs(ENHANCED_MODELS_DIR, exist_ok=True)

def load_network(network_type):
    """Load network from GEXF file.
    
    Args:
        network_type: 'eb' for Ellipsoid Body, 'fb' for Fan-shaped Body, 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        NetworkX DiGraph
    """
    # Try to load from Gephi directory first, then fall back to config_models directory
    gephi_path = os.path.join(GEPHI_CONFIG_MODELS_DIR, f"{network_type}_config_model.gexf")
    config_path = os.path.join(CONFIG_MODEL_DIR, f"{network_type}_config_model.gexf")
    
    if os.path.exists(gephi_path):
        return nx.read_gexf(gephi_path)
    else:
        return nx.read_gexf(config_path)

def enhance_network(G, network_type):
    """Add additional attributes to network for better visualization.
    
    Args:
        G: NetworkX DiGraph
        network_type: Type of network for naming
        
    Returns:
        Enhanced NetworkX DiGraph
    """
    print(f"Enhancing {network_type} configuration model...")
    
    # Get the undirected version for some metrics
    G_undirected = G.to_undirected()
    
    # Calculate node degrees
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = dict(G.degree())
    
    # Calculate betweenness centrality (on largest WCC to avoid errors)
    try:
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        largest_subgraph = G.subgraph(largest_wcc)
        betweenness = nx.betweenness_centrality(largest_subgraph)
        
        # Add 0 betweenness for nodes not in largest WCC
        for node in G.nodes():
            if node not in betweenness:
                betweenness[node] = 0.0
    except Exception as e:
        print(f"Error calculating betweenness centrality: {e}")
        betweenness = {node: 0.0 for node in G.nodes()}
    
    # Calculate clustering coefficients
    clustering = nx.clustering(G_undirected)
    
    # Add attributes to nodes
    for node in G.nodes():
        G.nodes[node]['in_degree'] = in_degrees.get(node, 0)
        G.nodes[node]['out_degree'] = out_degrees.get(node, 0)
        G.nodes[node]['total_degree'] = total_degrees.get(node, 0)
        G.nodes[node]['betweenness'] = betweenness.get(node, 0.0)
        G.nodes[node]['clustering'] = clustering.get(node, 0.0)
        
        # Set size and color based on degree and betweenness for visualization
        G.nodes[node]['size'] = 1 + 4 * np.sqrt(total_degrees.get(node, 0))
        G.nodes[node]['r'] = min(255, int(255 * betweenness.get(node, 0) * 10))
        G.nodes[node]['g'] = min(255, int(100 + 155 * clustering.get(node, 0)))
        G.nodes[node]['b'] = min(255, int(100 + 155 * (in_degrees.get(node, 0) / max(in_degrees.values()))))
        
        # Set node labels
        G.nodes[node]['label'] = f"Node {node}"
    
    return G

def save_enhanced_network(G, network_type):
    """Save enhanced network to GEXF file.
    
    Args:
        G: NetworkX DiGraph
        network_type: Type of network for naming
    """
    # Save to both locations for backward compatibility
    figure_path = os.path.join(ENHANCED_MODELS_DIR, f"{network_type}_enhanced_config_model.gexf")
    gephi_path = os.path.join(GEPHI_CONFIG_MODELS_DIR, f"{network_type}_enhanced_config_model.gexf")
    
    nx.write_gexf(G, figure_path)
    nx.write_gexf(G, gephi_path)
    
    print(f"Saved enhanced configuration model to:")
    print(f"- {figure_path}")
    print(f"- {gephi_path}")

def process_network(network_type):
    """Process a single network to create an enhanced configuration model.
    
    Args:
        network_type: Type of network ('eb', 'fb', or 'mb_kc')
    """
    # Load network
    G = load_network(network_type)
    
    # Enhance network
    G_enhanced = enhance_network(G, network_type)
    
    # Save enhanced network
    save_enhanced_network(G_enhanced, network_type)
    
    return G_enhanced

def main():
    """Main function to process selected networks."""
    # Process EB network
    eb_enhanced = process_network('eb')
    
    # Process FB network
    fb_enhanced = process_network('fb')
    
    # Process MB-KC network
    mb_kc_enhanced = process_network('mb_kc')
    
    print("\nEnhanced configuration models have been exported to:")
    print(f"- {ENHANCED_MODELS_DIR}")
    print(f"- {GEPHI_CONFIG_MODELS_DIR}")
    print("\nThese files include additional attributes for visualization in Gephi:")
    print("- Node size based on degree")
    print("- Node color based on betweenness centrality and clustering")
    print("- Additional attributes: in_degree, out_degree, total_degree, betweenness, clustering")

if __name__ == "__main__":
    main() 