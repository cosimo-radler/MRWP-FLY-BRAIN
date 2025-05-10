#!/usr/bin/env python3
"""
Gephi Enhancement Script for Drosophila Circuit Robustness Analysis

This script enhances the GEXF files for better visualization in Gephi
by adding additional attributes like node size, color, and edge weight.
"""

import os
import networkx as nx
import numpy as np
from tqdm import tqdm

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GEPHI_REAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gephi Graphs/real_models")
ENHANCED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "enhanced_data")

# Ensure directories exist
os.makedirs(ENHANCED_DIR, exist_ok=True)
os.makedirs(GEPHI_REAL_MODELS_DIR, exist_ok=True)

def enhance_network_for_gephi(G, network_type):
    """Enhance network with additional attributes for better Gephi visualization.
    
    Args:
        G: NetworkX graph
        network_type: 'eb' for Ellipsoid Body or 'mb_kc' for Mushroom Body Kenyon Cells
        
    Returns:
        Enhanced NetworkX graph
    """
    # Create a copy to avoid modifying the original
    G_enhanced = G.copy()
    
    # Calculate degree and other centrality measures
    degree = dict(G.degree())
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    try:
        # Calculate additional centrality measures (may be computationally intensive for large networks)
        if G.number_of_nodes() < 1000:  # Only for reasonably sized networks
            print(f"Calculating centrality measures for {network_type} network...")
            
            # PageRank
            pagerank = nx.pagerank(G, alpha=0.85)
            
            # Betweenness centrality (approximate for large networks)
            if G.number_of_nodes() > 500:
                betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes() // 2))
            else:
                betweenness = nx.betweenness_centrality(G)
                
            # Add these as node attributes
            nx.set_node_attributes(G_enhanced, pagerank, 'pagerank')
            nx.set_node_attributes(G_enhanced, betweenness, 'betweenness')
    except Exception as e:
        print(f"Warning: Couldn't calculate some centrality measures: {e}")
        
    # Set node attributes
    nx.set_node_attributes(G_enhanced, degree, 'degree')
    nx.set_node_attributes(G_enhanced, in_degree, 'in_degree')
    nx.set_node_attributes(G_enhanced, out_degree, 'out_degree')
    
    # Set node size attribute based on degree for visualization
    node_size = {node: 1 + 5 * degree[node] for node in G.nodes()}
    nx.set_node_attributes(G_enhanced, node_size, 'size')
    
    # Add random node positions for initial layout
    pos = nx.spring_layout(G, seed=42)
    pos_x = {node: float(pos[node][0]) for node in G.nodes()}
    pos_y = {node: float(pos[node][1]) for node in G.nodes()}
    
    nx.set_node_attributes(G_enhanced, pos_x, 'x')
    nx.set_node_attributes(G_enhanced, pos_y, 'y')
    
    # Add node colors based on degree (gradient from blue to red)
    max_degree = max(degree.values()) if degree else 1
    colors = {
        node: f"#{int(255 * min(1, degree[node] / max_degree)):02x}00{int(255 * (1 - min(1, degree[node] / max_degree))):02x}" 
        for node in G.nodes()
    }
    nx.set_node_attributes(G_enhanced, colors, 'color')
    
    # Add edge weights if not already present
    for u, v, data in G_enhanced.edges(data=True):
        if 'weight' not in data:
            G_enhanced[u][v]['weight'] = 1.0
    
    # Add edge colors
    for u, v in G_enhanced.edges():
        G_enhanced[u][v]['color'] = '#555555'
    
    return G_enhanced

def main():
    """Main execution function."""
    print("Enhancing networks for Gephi...")
    
    # Load EB network
    print("\nProcessing Ellipsoid Body (EB) network...")
    try:
        # Try to load from Gephi directory first, then fall back to data directory
        eb_path = os.path.join(GEPHI_REAL_MODELS_DIR, "eb_network.gexf")
        data_path = os.path.join(DATA_DIR, "eb_network.gexf")
        
        if os.path.exists(eb_path):
            eb_network = nx.read_gexf(eb_path)
        else:
            eb_network = nx.read_gexf(data_path)
            
        print(f"EB Network: {eb_network.number_of_nodes()} nodes, {eb_network.number_of_edges()} edges")
        
        # Enhance network
        eb_enhanced = enhance_network_for_gephi(eb_network, 'eb')
        
        # Save enhanced network to both locations
        nx.write_gexf(eb_enhanced, os.path.join(ENHANCED_DIR, "eb_network_enhanced.gexf"))
        nx.write_gexf(eb_enhanced, os.path.join(GEPHI_REAL_MODELS_DIR, "eb_network_enhanced.gexf"))
        print(f"Enhanced EB network saved to enhanced_data and Gephi Graphs/real_models directories")
        
    except Exception as e:
        print(f"Error processing EB network: {e}")
    
    # Load FB network
    print("\nProcessing Fan-shaped Body (FB) network...")
    try:
        # Try to load from Gephi directory first, then fall back to data directory
        fb_path = os.path.join(GEPHI_REAL_MODELS_DIR, "fb_network.gexf")
        data_path = os.path.join(DATA_DIR, "fb_network.gexf")
        
        if os.path.exists(fb_path):
            fb_network = nx.read_gexf(fb_path)
        else:
            fb_network = nx.read_gexf(data_path)
            
        print(f"FB Network: {fb_network.number_of_nodes()} nodes, {fb_network.number_of_edges()} edges")
        
        # Enhance network
        fb_enhanced = enhance_network_for_gephi(fb_network, 'fb')
        
        # Save enhanced network to both locations
        nx.write_gexf(fb_enhanced, os.path.join(ENHANCED_DIR, "fb_network_enhanced.gexf"))
        nx.write_gexf(fb_enhanced, os.path.join(GEPHI_REAL_MODELS_DIR, "fb_network_enhanced.gexf"))
        print(f"Enhanced FB network saved to enhanced_data and Gephi Graphs/real_models directories")
        
    except Exception as e:
        print(f"Error processing FB network: {e}")
    
    # Load MB-KC network
    print("\nProcessing Mushroom Body Kenyon Cell (MB-KC) network...")
    try:
        # Try to load from Gephi directory first, then fall back to data directory
        mb_kc_path = os.path.join(GEPHI_REAL_MODELS_DIR, "mb_kc_network.gexf")
        data_path = os.path.join(DATA_DIR, "mb_kc_network.gexf")
        
        if os.path.exists(mb_kc_path):
            mb_kc_network = nx.read_gexf(mb_kc_path)
        else:
            mb_kc_network = nx.read_gexf(data_path)
            
        print(f"MB-KC Network: {mb_kc_network.number_of_nodes()} nodes, {mb_kc_network.number_of_edges()} edges")
        
        # Enhance network
        mb_kc_enhanced = enhance_network_for_gephi(mb_kc_network, 'mb_kc')
        
        # Save enhanced network to both locations
        nx.write_gexf(mb_kc_enhanced, os.path.join(ENHANCED_DIR, "mb_kc_network_enhanced.gexf"))
        nx.write_gexf(mb_kc_enhanced, os.path.join(GEPHI_REAL_MODELS_DIR, "mb_kc_network_enhanced.gexf"))
        print(f"Enhanced MB-KC network saved to enhanced_data and Gephi Graphs/real_models directories")
        
    except Exception as e:
        print(f"Error processing MB-KC network: {e}")
    
    print("\nNetwork enhancement complete!")

if __name__ == "__main__":
    main() 