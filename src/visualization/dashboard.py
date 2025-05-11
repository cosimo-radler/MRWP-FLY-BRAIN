#!/usr/bin/env python3
"""
Interactive Network Comparison Dashboard

This dashboard visualizes the comparison of different network models:
- Original networks
- Scaled configuration models (1500 nodes)
- Unscaled configuration models
- Upscaled configuration models (3500 nodes)
- Clustering-preserved configuration models

Features:
- Interactive toggling of models
- Visualization of degree distributions
- Visualization of percolation and attack results
- Network metrics comparison
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
from collections import Counter
import functools
import time
from flask_caching import Cache

# Constants from the original visualization script
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models")
UNSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "unscaled")
UPSCALED_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "upscaled")
CLUSTERING_CONFIG_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_models", "clustering")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Network types and their display names
NETWORKS = {
    'eb': 'Ellipsoid Body',
    'fb': 'Fan-shaped Body',
    'mb_kc': 'Mushroom Body KC'
}

# Model types and their display names
MODEL_TYPES = ['original', 'scaled_config', 'unscaled_config', 'upscaled_config', 'clustering_config', 'unscaled_clustering_config']
MODEL_LABELS = {
    'original': 'Original',
    'scaled_config': 'Scaled Config (1500)',
    'unscaled_config': 'Unscaled Config',
    'upscaled_config': 'Upscaled Config (3500)',
    'clustering_config': 'Clustering Config (1500)',
    'unscaled_clustering_config': 'Unscaled Clustering Config'
}

# Visualization parameters
COLORS = {
    'original': 'blue',
    'scaled_config': 'red',
    'unscaled_config': 'green',
    'upscaled_config': 'purple',
    'clustering_config': 'orange',
    'unscaled_clustering_config': 'brown'
}

# Helper functions adapted from the visualization script
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

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Setup caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes in seconds
})

# Add caching decorator for network loading
@cache.memoize()
def cached_load_network(network_type, model_type):
    """Cached version of load_network"""
    return load_network(network_type, model_type)

# Add caching decorator for degree distribution
@cache.memoize()
def cached_get_normalized_degree_distribution(network_type, model_type):
    """Cached version of get_normalized_degree_distribution"""
    G = cached_load_network(network_type, model_type)
    return get_normalized_degree_distribution(G)

# Add caching decorator for percolation results
@cache.memoize()
def cached_load_percolation_results(network_type, model_type):
    """Cached version of load_percolation_results"""
    return load_percolation_results(network_type, model_type)

# Add caching decorator for attack results
@cache.memoize()
def cached_load_attack_results(network_type, attack_strategy, model_type):
    """Cached version of load_attack_results"""
    return load_attack_results(network_type, attack_strategy, model_type)

# Update the degree distribution figure creation function to use caching
def create_degree_distribution_figure(network_type, selected_models):
    """Create an interactive degree distribution plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        # Use cached function to get degree distribution
        degrees, freq = cached_get_normalized_degree_distribution(network_type, model_type)
        
        if degrees and freq:
            fig.add_trace(go.Scatter(
                x=degrees,
                y=freq,
                mode='lines+markers',
                name=MODEL_LABELS[model_type],
                line=dict(color=COLORS[model_type]),
                marker=dict(size=8)
            ))
    
    # Set axis scales to logarithmic
    fig.update_xaxes(type="log", title="Degree (log scale)")
    fig.update_yaxes(type="log", title="Normalized Frequency [P(k)]")
    
    # Update layout
    fig.update_layout(
        title=f"{NETWORKS[network_type]} Degree Distribution",
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

# Update the percolation figure creation function to use caching
def create_percolation_figure(network_type, selected_models):
    """Create an interactive percolation plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        # Use cached function to load percolation results
        df = cached_load_percolation_results(network_type, model_type)
        
        if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['removal_probability'],
                y=df['mean_lcc_size'],
                mode='lines+markers',
                name=MODEL_LABELS[model_type],
                line=dict(color=COLORS[model_type]),
                marker=dict(size=8)
            ))
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=0, y0=0.05, x1=1, y1=0.05,
        line=dict(color="gray", width=1, dash="dot"),
    )
    
    fig.add_annotation(
        x=0.1, y=0.07,
        text="Threshold (5%)",
        showarrow=False,
        font=dict(color="gray")
    )
    
    # Update layout
    fig.update_layout(
        title=f"{NETWORKS[network_type]} Random Percolation",
        xaxis_title="Edge Removal Probability",
        yaxis_title="Largest Connected Component",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

# Update the attack figure creation function to use caching
def create_attack_figure(network_type, attack_strategy, selected_models):
    """Create an interactive attack plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        # Use cached function to load attack results
        df = cached_load_attack_results(network_type, attack_strategy, model_type)
        
        if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['removal_probability'],
                y=df['mean_lcc_size'],
                mode='lines+markers',
                name=MODEL_LABELS[model_type],
                line=dict(color=COLORS[model_type]),
                marker=dict(size=8)
            ))
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=0, y0=0.05, x1=1, y1=0.05,
        line=dict(color="gray", width=1, dash="dot"),
    )
    
    fig.add_annotation(
        x=0.1, y=0.07,
        text="Threshold (5%)",
        showarrow=False,
        font=dict(color="gray")
    )
    
    attack_name = "Degree Centrality" if attack_strategy == "degree" else "Betweenness Centrality"
    
    # Update layout
    fig.update_layout(
        title=f"{NETWORKS[network_type]} {attack_name} Attack",
        xaxis_title="Edge Removal Probability",
        yaxis_title="Largest Connected Component",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

# Add a caching decorator for network metrics
@cache.memoize()
def cached_calculate_network_metrics(network_type, model_type):
    """Cached version of calculate_network_metrics"""
    G = cached_load_network(network_type, model_type)
    metrics = calculate_network_metrics(G)
    
    if metrics:
        # Add percolation and attack thresholds
        perc_threshold = extract_percolation_threshold(network_type, model_type)
        metrics['Percolation Threshold'] = perc_threshold
        
        deg_threshold = extract_attack_threshold(network_type, 'degree', model_type)
        metrics['Degree Attack Threshold'] = deg_threshold
        
        bet_threshold = extract_attack_threshold(network_type, 'betweenness', model_type)
        metrics['Betweenness Attack Threshold'] = bet_threshold
    
    return metrics

# Update the metrics table creation function to use caching
def create_metrics_table(network_type, selected_models):
    """Create a metrics comparison table for the selected models."""
    # Headers for the table
    row_labels = [
        'Nodes', 
        'Edges', 
        'Avg Degree', 
        'Clustering', 
        'Avg Path Length', 
        'Diameter',
        'Percolation Threshold',
        'Degree Attack Threshold',
        'Betweenness Attack Threshold'
    ]
    
    # Collect metrics for each model type
    table_data = []
    for row_label in row_labels:
        row = {'Metric': row_label}
        for model_type in selected_models:
            metrics = cached_calculate_network_metrics(network_type, model_type)
            
            if metrics is not None and row_label in metrics:
                value = metrics[row_label]
                if isinstance(value, float) and not np.isnan(value):
                    row[MODEL_LABELS[model_type]] = value
                else:
                    row[MODEL_LABELS[model_type]] = str(value)
            else:
                row[MODEL_LABELS[model_type]] = 'N/A'
        table_data.append(row)
    
    return table_data

# Add new functions for region comparison

@cache.memoize()
def create_region_comparison_figure(selected_regions, model_type, plot_type):
    """Create a figure comparing different brain regions using the same model type.
    
    Args:
        selected_regions: List of brain region types to include
        model_type: Which model type to use for comparison
        plot_type: 'degree', 'percolation', 'degree_attack', or 'betweenness_attack'
    """
    fig = go.Figure()
    
    for network_type in selected_regions:
        if plot_type == 'degree':
            # Degree distribution comparison
            degrees, freq = cached_get_normalized_degree_distribution(network_type, model_type)
            
            if degrees and freq:
                fig.add_trace(go.Scatter(
                    x=degrees,
                    y=freq,
                    mode='lines+markers',
                    name=NETWORKS[network_type],
                    marker=dict(size=8)
                ))
                
            # Set axis scales to logarithmic
            fig.update_xaxes(type="log", title="Degree (log scale)")
            fig.update_yaxes(type="log", title="Normalized Frequency [P(k)]")
            title = f"{MODEL_LABELS[model_type]} - Degree Distribution Comparison"
            
        elif plot_type == 'percolation':
            # Percolation comparison
            df = cached_load_percolation_results(network_type, model_type)
            
            if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['removal_probability'],
                    y=df['mean_lcc_size'],
                    mode='lines+markers',
                    name=NETWORKS[network_type],
                    marker=dict(size=8)
                ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=0, y0=0.05, x1=1, y1=0.05,
                line=dict(color="gray", width=1, dash="dot"),
            )
            
            fig.update_xaxes(title="Edge Removal Probability", range=[0, 1])
            fig.update_yaxes(title="Largest Connected Component", range=[0, 1.05])
            title = f"{MODEL_LABELS[model_type]} - Random Percolation Comparison"
            
        elif plot_type == 'degree_attack':
            # Degree attack comparison
            df = cached_load_attack_results(network_type, 'degree', model_type)
            
            if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['removal_probability'],
                    y=df['mean_lcc_size'],
                    mode='lines+markers',
                    name=NETWORKS[network_type],
                    marker=dict(size=8)
                ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=0, y0=0.05, x1=1, y1=0.05,
                line=dict(color="gray", width=1, dash="dot"),
            )
            
            fig.update_xaxes(title="Edge Removal Probability", range=[0, 1])
            fig.update_yaxes(title="Largest Connected Component", range=[0, 1.05])
            title = f"{MODEL_LABELS[model_type]} - Degree Attack Comparison"
            
        elif plot_type == 'betweenness_attack':
            # Betweenness attack comparison
            df = cached_load_attack_results(network_type, 'betweenness', model_type)
            
            if df is not None and 'removal_probability' in df.columns and 'mean_lcc_size' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['removal_probability'],
                    y=df['mean_lcc_size'],
                    mode='lines+markers',
                    name=NETWORKS[network_type],
                    marker=dict(size=8)
                ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=0, y0=0.05, x1=1, y1=0.05,
                line=dict(color="gray", width=1, dash="dot"),
            )
            
            fig.update_xaxes(title="Edge Removal Probability", range=[0, 1])
            fig.update_yaxes(title="Largest Connected Component", range=[0, 1.05])
            title = f"{MODEL_LABELS[model_type]} - Betweenness Attack Comparison"
    
    # Update layout
    fig.update_layout(
        title=title,
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

@cache.memoize()
def create_metrics_bar_chart(metric_name, selected_regions, selected_models):
    """Create a bar chart comparing a specific metric across brain regions and models.
    
    Args:
        metric_name: Name of the metric to compare
        selected_regions: List of brain region types to include
        selected_models: List of model types to include
    """
    # Prepare data for bar chart
    regions = []
    model_types = []
    values = []
    
    for network_type in selected_regions:
        for model_type in selected_models:
            metrics = cached_calculate_network_metrics(network_type, model_type)
            
            if metrics is not None and metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    regions.append(NETWORKS[network_type])
                    model_types.append(MODEL_LABELS[model_type])
                    values.append(value)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Brain Region': regions,
        'Model Type': model_types,
        'Value': values
    })
    
    # Create grouped bar chart
    fig = px.bar(
        df, 
        x='Brain Region', 
        y='Value', 
        color='Model Type',
        barmode='group',
        title=f'Comparison of {metric_name} Across Brain Regions and Models',
        height=500
    )
    
    # Update layout
    fig.update_layout(
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Define app layout with interactive components
app.layout = dbc.Container([
    html.H1("Network Model Comparison Dashboard", className="text-center my-4"),
    
    # Network type selector
    dbc.Row([
        dbc.Col([
            html.H4("Select Brain Region Network:"),
            dcc.RadioItems(
                id='network-selector',
                options=[{'label': NETWORKS[k], 'value': k} for k in NETWORKS.keys()],
                value='eb',
                className="mb-4",
                inputStyle={"margin-right": "10px", "margin-left": "20px"}
            )
        ], width=12)
    ]),
    
    # Model type selector with checkboxes
    dbc.Row([
        dbc.Col([
            html.H4("Toggle Models:"),
            dcc.Checklist(
                id='model-selector',
                options=[{'label': MODEL_LABELS[k], 'value': k} for k in MODEL_TYPES],
                value=['original', 'scaled_config'],  # Default selected models
                className="mb-4",
                inputStyle={"margin-right": "10px", "margin-left": "20px"}
            )
        ], width=12)
    ]),
    
    # Tabs for different visualizations
    dbc.Tabs([
        # Degree Distribution Tab
        dbc.Tab(label="Degree Distribution", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='degree-distribution-plot')
                ], width=12)
            ])
        ]),
        
        # Percolation Tab
        dbc.Tab(label="Random Percolation", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='percolation-plot')
                ], width=12)
            ])
        ]),
        
        # Degree Attack Tab
        dbc.Tab(label="Degree Centrality Attack", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='degree-attack-plot')
                ], width=12)
            ])
        ]),
        
        # Betweenness Attack Tab
        dbc.Tab(label="Betweenness Centrality Attack", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='betweenness-attack-plot')
                ], width=12)
            ])
        ]),
        
        # Network Metrics Tab
        dbc.Tab(label="Network Metrics", children=[
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id='metrics-table',
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'padding': '10px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 6},  # Percolation threshold
                                'backgroundColor': 'rgba(255, 235, 205, 0.3)'
                            },
                            {
                                'if': {'row_index': 7},  # Degree attack threshold
                                'backgroundColor': 'rgba(255, 235, 205, 0.3)'
                            },
                            {
                                'if': {'row_index': 8},  # Betweenness attack threshold
                                'backgroundColor': 'rgba(255, 235, 205, 0.3)'
                            },
                        ],
                        style_table={'overflowX': 'auto'}
                    )
                ], width=12)
            ])
        ]),
        
        # New tab for region comparison
        dbc.Tab(label="Region Comparison", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Select Regions to Compare:"),
                    dcc.Checklist(
                        id='region-comparison-selector',
                        options=[{'label': NETWORKS[k], 'value': k} for k in NETWORKS.keys()],
                        value=['eb', 'fb', 'mb_kc'],  # Default all regions
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=6),
                dbc.Col([
                    html.H4("Select Model Type:"),
                    dcc.RadioItems(
                        id='region-comparison-model',
                        options=[{'label': MODEL_LABELS[k], 'value': k} for k in MODEL_TYPES],
                        value='original',  # Default original model
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=6)
            ], className="mt-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Select Plot Type:"),
                    dcc.RadioItems(
                        id='region-comparison-plot-type',
                        options=[
                            {'label': 'Degree Distribution', 'value': 'degree'},
                            {'label': 'Random Percolation', 'value': 'percolation'},
                            {'label': 'Degree Attack', 'value': 'degree_attack'},
                            {'label': 'Betweenness Attack', 'value': 'betweenness_attack'}
                        ],
                        value='percolation',  # Default percolation
                        className="mb-4",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='region-comparison-plot')
                ], width=12)
            ])
        ]),
        
        # New tab for metric bar charts
        dbc.Tab(label="Metrics Bar Charts", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Select Regions to Compare:"),
                    dcc.Checklist(
                        id='bar-chart-region-selector',
                        options=[{'label': NETWORKS[k], 'value': k} for k in NETWORKS.keys()],
                        value=['eb', 'fb', 'mb_kc'],  # Default all regions
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=6),
                dbc.Col([
                    html.H4("Select Models to Compare:"),
                    dcc.Checklist(
                        id='bar-chart-model-selector',
                        options=[{'label': MODEL_LABELS[k], 'value': k} for k in MODEL_TYPES],
                        value=['original', 'scaled_config'],  # Default original and scaled
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=6)
            ], className="mt-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Select Metric:"),
                    dcc.RadioItems(
                        id='bar-chart-metric',
                        options=[
                            {'label': 'Clustering Coefficient', 'value': 'Clustering'},
                            {'label': 'Average Path Length', 'value': 'Avg Path Length'},
                            {'label': 'Average Degree', 'value': 'Avg Degree'},
                            {'label': 'Percolation Threshold', 'value': 'Percolation Threshold'},
                            {'label': 'Degree Attack Threshold', 'value': 'Degree Attack Threshold'},
                            {'label': 'Betweenness Attack Threshold', 'value': 'Betweenness Attack Threshold'}
                        ],
                        value='Clustering',  # Default clustering
                        className="mb-4",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='metrics-bar-chart')
                ], width=12)
            ])
        ]),
    ]),
    
    html.Hr(),
    
    # Footer with info
    dbc.Row([
        dbc.Col([
            html.P("Dashboard for comparing network models and their robustness properties.", className="text-center"),
            html.P("Toggle models on/off using the checkboxes above to compare their characteristics.", className="text-center")
        ], width=12)
    ])
], fluid=True)

# Update callback to show loading state
@app.callback(
    Output('degree-distribution-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False,  # Load on initial page load
)
def update_degree_distribution(network_type, selected_models):
    # Add small delay for more responsive UI
    return create_degree_distribution_figure(network_type, selected_models)

@app.callback(
    Output('percolation-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=True,  # Only load when tab is active
)
def update_percolation_plot(network_type, selected_models):
    return create_percolation_figure(network_type, selected_models)

@app.callback(
    Output('degree-attack-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=True,  # Only load when tab is active
)
def update_degree_attack_plot(network_type, selected_models):
    return create_attack_figure(network_type, 'degree', selected_models)

@app.callback(
    Output('betweenness-attack-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=True,  # Only load when tab is active
)
def update_betweenness_attack_plot(network_type, selected_models):
    return create_attack_figure(network_type, 'betweenness', selected_models)

@app.callback(
    Output('metrics-table', 'data'),
    Output('metrics-table', 'columns'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=True,  # Only load when tab is active
)
def update_metrics_table(network_type, selected_models):
    table_data = create_metrics_table(network_type, selected_models)
    
    # Create columns based on selected models
    columns = [{"name": "Metric", "id": "Metric"}]
    for model_type in selected_models:
        columns.append({"name": MODEL_LABELS[model_type], "id": MODEL_LABELS[model_type]})
    
    return table_data, columns

# Add callbacks for new region comparison tab
@app.callback(
    Output('region-comparison-plot', 'figure'),
    [
        Input('region-comparison-selector', 'value'),
        Input('region-comparison-model', 'value'),
        Input('region-comparison-plot-type', 'value')
    ],
    prevent_initial_call=True
)
def update_region_comparison(selected_regions, model_type, plot_type):
    return create_region_comparison_figure(selected_regions, model_type, plot_type)

# Add callback for metrics bar chart
@app.callback(
    Output('metrics-bar-chart', 'figure'),
    [
        Input('bar-chart-metric', 'value'),
        Input('bar-chart-region-selector', 'value'),
        Input('bar-chart-model-selector', 'value')
    ],
    prevent_initial_call=True
)
def update_metrics_bar_chart(metric_name, selected_regions, selected_models):
    return create_metrics_bar_chart(metric_name, selected_regions, selected_models)

# Run the app
if __name__ == '__main__':
    # Preload some data for faster initial rendering
    print("Preloading most common data configurations...")
    for network_type in ['eb', 'fb', 'mb_kc']:
        for model_type in ['original', 'scaled_config']:
            cached_load_network(network_type, model_type)
            cached_get_normalized_degree_distribution(network_type, model_type)
            cached_load_percolation_results(network_type, model_type)
            for attack in ['degree', 'betweenness']:
                cached_load_attack_results(network_type, attack, model_type)
    print("Preloading complete. Starting server...")
    
    app.run(debug=True, port=8052) 