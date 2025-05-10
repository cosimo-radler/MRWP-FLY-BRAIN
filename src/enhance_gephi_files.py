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
ENHANCED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "enhanced_data")

# Ensure enhanced data directory exists
os.makedirs(ENHANCED_DIR, exist_ok=True)

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
        eb_network = nx.read_gexf(os.path.join(DATA_DIR, "eb_network.gexf"))
        print(f"EB Network: {eb_network.number_of_nodes()} nodes, {eb_network.number_of_edges()} edges")
        
        # Enhance network
        eb_enhanced = enhance_network_for_gephi(eb_network, 'eb')
        
        # Save enhanced network
        nx.write_gexf(eb_enhanced, os.path.join(ENHANCED_DIR, "eb_network_enhanced.gexf"))
        print(f"Enhanced EB network saved to {os.path.join(ENHANCED_DIR, 'eb_network_enhanced.gexf')}")
        
    except Exception as e:
        print(f"Error processing EB network: {e}")
    
    # Load MB-KC network
    print("\nProcessing Mushroom Body Kenyon Cell (MB-KC) network...")
    try:
        mb_kc_network = nx.read_gexf(os.path.join(DATA_DIR, "mb_kc_network.gexf"))
        print(f"MB-KC Network: {mb_kc_network.number_of_nodes()} nodes, {mb_kc_network.number_of_edges()} edges")
        
        # Enhance network
        mb_kc_enhanced = enhance_network_for_gephi(mb_kc_network, 'mb_kc')
        
        # Save enhanced network
        nx.write_gexf(mb_kc_enhanced, os.path.join(ENHANCED_DIR, "mb_kc_network_enhanced.gexf"))
        print(f"Enhanced MB-KC network saved to {os.path.join(ENHANCED_DIR, 'mb_kc_network_enhanced.gexf')}")
        
    except Exception as e:
        print(f"Error processing MB-KC network: {e}")
    
    print("\nNetwork enhancement complete!")

if __name__ == "__main__":
    main() 