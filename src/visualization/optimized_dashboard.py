#!/usr/bin/env python3
"""
Optimized Network Comparison Dashboard

This dashboard visualizes the comparison of different network models using
precomputed data stored on disk. This optimized version provides instant
visualization by loading precomputed data instead of performing calculations
on the fly.

Features:
- Instant visualization with no calculation delays
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
import pickle
import json
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import time

# Path to precomputed data
PRECOMPUTED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precomputed_data")

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

# Load all precomputed data
print("Loading precomputed data...")
start_load = time.time()

# Load degree distributions
with open(os.path.join(PRECOMPUTED_DIR, 'degree_distributions.pkl'), 'rb') as f:
    DEGREE_DISTRIBUTIONS = pickle.load(f)

# Load percolation results
with open(os.path.join(PRECOMPUTED_DIR, 'percolation_results.pkl'), 'rb') as f:
    PERCOLATION_RESULTS = pickle.load(f)

# Load attack results
with open(os.path.join(PRECOMPUTED_DIR, 'attack_results.pkl'), 'rb') as f:
    ATTACK_RESULTS = pickle.load(f)

# Load network metrics
with open(os.path.join(PRECOMPUTED_DIR, 'network_metrics.pkl'), 'rb') as f:
    NETWORK_METRICS = pickle.load(f)

print(f"Data loaded in {time.time() - start_load:.2f} seconds!")

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create visualization functions that use precomputed data
def create_degree_distribution_figure(network_type, selected_models):
    """Create an interactive degree distribution plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        if (
            network_type in DEGREE_DISTRIBUTIONS and 
            model_type in DEGREE_DISTRIBUTIONS[network_type]
        ):
            data = DEGREE_DISTRIBUTIONS[network_type][model_type]
            degrees = data['degrees']
            freq = data['frequencies']
            
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

def create_percolation_figure(network_type, selected_models):
    """Create an interactive percolation plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        if (
            network_type in PERCOLATION_RESULTS and 
            model_type in PERCOLATION_RESULTS[network_type]
        ):
            data = PERCOLATION_RESULTS[network_type][model_type]
            x = data['removal_probability']
            y = data['mean_lcc_size']
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
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

def create_attack_figure(network_type, attack_strategy, selected_models):
    """Create an interactive attack plot for the selected models."""
    fig = go.Figure()
    
    for model_type in selected_models:
        if (
            network_type in ATTACK_RESULTS[attack_strategy] and 
            model_type in ATTACK_RESULTS[attack_strategy][network_type]
        ):
            data = ATTACK_RESULTS[attack_strategy][network_type][model_type]
            x = data['removal_probability']
            y = data['mean_lcc_size']
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
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

def create_metrics_table(network_type, selected_models):
    """Create a table of network metrics for the selected models."""
    # List of metrics to show in the table
    metrics_to_show = [
        'Nodes', 'Edges', 'Avg Degree', 'Clustering', 'Avg Path Length', 'Diameter',
        'Percolation Threshold', 'Degree Attack Threshold', 'Betweenness Attack Threshold'
    ]
    
    # Create a list of dictionaries for the table
    table_data = []
    
    # Add a row for each metric
    for metric in metrics_to_show:
        row = {'Metric': metric}
        
        # Add a column for each selected model
        for model_type in selected_models:
            if (
                network_type in NETWORK_METRICS and 
                model_type in NETWORK_METRICS[network_type] and
                metric in NETWORK_METRICS[network_type][model_type]
            ):
                row[MODEL_LABELS[model_type]] = NETWORK_METRICS[network_type][model_type][metric]
            else:
                row[MODEL_LABELS[model_type]] = 'N/A'
        
        table_data.append(row)
    
    return table_data

def create_region_comparison_figure(selected_regions, model_type, plot_type):
    """Create a comparison plot for multiple regions using a single model type."""
    fig = go.Figure()
    
    # Different plotting logic based on the plot type
    if plot_type == 'degree':
        # Degree distribution comparison
        for network_type in selected_regions:
            if (
                network_type in DEGREE_DISTRIBUTIONS and 
                model_type in DEGREE_DISTRIBUTIONS[network_type]
            ):
                data = DEGREE_DISTRIBUTIONS[network_type][model_type]
                degrees = data['degrees']
                freq = data['frequencies']
                
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
        fig.update_layout(title=f"Degree Distribution Comparison ({MODEL_LABELS[model_type]})")
        
    elif plot_type == 'percolation':
        # Percolation comparison
        for network_type in selected_regions:
            if (
                network_type in PERCOLATION_RESULTS and 
                model_type in PERCOLATION_RESULTS[network_type]
            ):
                data = PERCOLATION_RESULTS[network_type][model_type]
                x = data['removal_probability']
                y = data['mean_lcc_size']
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
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
        fig.update_layout(title=f"Random Percolation Comparison ({MODEL_LABELS[model_type]})")
        
    elif plot_type == 'degree_attack' or plot_type == 'betweenness_attack':
        # Attack comparison
        attack_strategy = 'degree' if plot_type == 'degree_attack' else 'betweenness'
        attack_name = "Degree Centrality" if attack_strategy == "degree" else "Betweenness Centrality"
        
        for network_type in selected_regions:
            if (
                network_type in ATTACK_RESULTS[attack_strategy] and 
                model_type in ATTACK_RESULTS[attack_strategy][network_type]
            ):
                data = ATTACK_RESULTS[attack_strategy][network_type][model_type]
                x = data['removal_probability']
                y = data['mean_lcc_size']
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
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
        fig.update_layout(title=f"{attack_name} Attack Comparison ({MODEL_LABELS[model_type]})")
    
    # Common layout settings
    fig.update_layout(
        legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

def create_metrics_bar_chart(metric_name, selected_regions, selected_models):
    """Create a bar chart comparing a specific metric across regions and models."""
    # Prepare data for bar chart
    regions = []
    models = []
    values = []
    
    for network_type in selected_regions:
        for model_type in selected_models:
            if (
                network_type in NETWORK_METRICS and 
                model_type in NETWORK_METRICS[network_type] and
                metric_name in NETWORK_METRICS[network_type][model_type]
            ):
                val = NETWORK_METRICS[network_type][model_type][metric_name]
                
                # Skip non-numeric values
                if val == 'N/A':
                    continue
                    
                regions.append(NETWORKS[network_type])
                models.append(MODEL_LABELS[model_type])
                values.append(val)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Region': regions,
        'Model': models,
        'Value': values
    })
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Region', 
        y='Value', 
        color='Model',
        barmode='group',
        title=f'{metric_name} Comparison',
        labels={'Value': metric_name, 'Region': 'Brain Region'}
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

# Define the app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Network Analysis Dashboard (Optimized)", className="text-center"),
            html.H4("Compare Neural Network Models", className="text-center")
        ], width=12)
    ], className="mt-4 mb-4"),
    
    # Main content in tabs
    dbc.Tabs([
        # First tab - Model comparison for a single network
        dbc.Tab(label="Single Network Comparison", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Select Network:"),
                    dcc.RadioItems(
                        id='network-selector',
                        options=[{'label': NETWORKS[k], 'value': k} for k in NETWORKS.keys()],
                        value='eb',  # Default ellipsoid body
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=4),
                dbc.Col([
                    html.H4("Select Models to Compare:"),
                    dcc.Checklist(
                        id='model-selector',
                        options=[{'label': MODEL_LABELS[k], 'value': k} for k in MODEL_TYPES],
                        value=['original', 'scaled_config'],  # Default original and scaled
                        className="mb-2",
                        inputStyle={"margin-right": "10px", "margin-left": "20px"}
                    )
                ], width=8)
            ], className="mt-3"),
            
            # Degree distribution plot
            dbc.Row([
                dbc.Col([
                    html.H4("Degree Distribution", className="text-center"),
                    dcc.Graph(id='degree-distribution-plot')
                ], width=12)
            ], className="mt-3"),
            
            # Percolation plot
            dbc.Row([
                dbc.Col([
                    html.H4("Random Percolation", className="text-center"),
                    dcc.Graph(id='percolation-plot')
                ], width=12)
            ], className="mt-3"),
            
            # Attack plots
            dbc.Row([
                dbc.Col([
                    html.H4("Degree-based Attack", className="text-center"),
                    dcc.Graph(id='degree-attack-plot')
                ], width=6),
                dbc.Col([
                    html.H4("Betweenness-based Attack", className="text-center"),
                    dcc.Graph(id='betweenness-attack-plot')
                ], width=6)
            ], className="mt-3"),
            
            # Metrics table
            dbc.Row([
                dbc.Col([
                    html.H4("Network Metrics", className="text-center"),
                    dash_table.DataTable(
                        id='metrics-table',
                        style_cell={
                            'textAlign': 'center',
                            'padding': '10px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                ], width=12)
            ], className="mt-3 mb-5")
        ]),
        
        # Second tab - Region comparison with a fixed model
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
        
        # Third tab for metric bar charts
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
            html.P("Optimized version using precomputed data for instant visualization.", className="text-center")
        ], width=12)
    ])
], fluid=True)

# Define callbacks
@app.callback(
    Output('degree-distribution-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False  # Load on initial page load
)
def update_degree_distribution(network_type, selected_models):
    return create_degree_distribution_figure(network_type, selected_models)

@app.callback(
    Output('percolation-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False  # Load immediately
)
def update_percolation_plot(network_type, selected_models):
    return create_percolation_figure(network_type, selected_models)

@app.callback(
    Output('degree-attack-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False  # Load immediately
)
def update_degree_attack_plot(network_type, selected_models):
    return create_attack_figure(network_type, 'degree', selected_models)

@app.callback(
    Output('betweenness-attack-plot', 'figure'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False  # Load immediately
)
def update_betweenness_attack_plot(network_type, selected_models):
    return create_attack_figure(network_type, 'betweenness', selected_models)

@app.callback(
    Output('metrics-table', 'data'),
    Output('metrics-table', 'columns'),
    [Input('network-selector', 'value'),
     Input('model-selector', 'value')],
    prevent_initial_call=False  # Load immediately
)
def update_metrics_table(network_type, selected_models):
    table_data = create_metrics_table(network_type, selected_models)
    
    # Create columns based on selected models
    columns = [{"name": "Metric", "id": "Metric"}]
    for model_type in selected_models:
        columns.append({"name": MODEL_LABELS[model_type], "id": MODEL_LABELS[model_type]})
    
    return table_data, columns

@app.callback(
    Output('region-comparison-plot', 'figure'),
    [
        Input('region-comparison-selector', 'value'),
        Input('region-comparison-model', 'value'),
        Input('region-comparison-plot-type', 'value')
    ],
    prevent_initial_call=False  # Load immediately
)
def update_region_comparison(selected_regions, model_type, plot_type):
    return create_region_comparison_figure(selected_regions, model_type, plot_type)

@app.callback(
    Output('metrics-bar-chart', 'figure'),
    [
        Input('bar-chart-metric', 'value'),
        Input('bar-chart-region-selector', 'value'),
        Input('bar-chart-model-selector', 'value')
    ],
    prevent_initial_call=False  # Load immediately
)
def update_metrics_bar_chart(metric_name, selected_regions, selected_models):
    return create_metrics_bar_chart(metric_name, selected_regions, selected_models)

# Run the app
if __name__ == '__main__':
    print("All data loaded and dashboard ready!")
    app.run(debug=True, port=8053) 