# Network Model Comparison Dashboard

This interactive dashboard allows you to compare different network models for the brain region networks:
- Original networks
- Scaled configuration models (1500 nodes)
- Unscaled configuration models
- Upscaled configuration models (3500 nodes)
- Clustering-preserved configuration models (both scaled and unscaled variants)

## Features

- Interactive toggling of models via checkboxes
- Select different brain regions (Ellipsoid Body, Fan-shaped Body, Mushroom Body KC)
- Compare:
  - Degree distributions (log-log scale)
  - Random percolation robustness 
  - Targeted attack robustness (degree centrality and betweenness centrality)
  - Network metrics (nodes, edges, clustering, etc.)

## Running the Dashboard

1. Ensure you have the required dependencies installed:
   ```
   pip install dash dash-bootstrap-components plotly pandas networkx
   ```

2. Run the dashboard from the project root directory:
   ```
   python src/visualization/dashboard.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8050/
   ```

## Usage Guide

1. **Select a brain region network** using the radio buttons at the top
2. **Toggle models on/off** using the checkboxes to control which models are displayed
3. **Navigate between tabs** to view different visualizations:
   - Degree Distribution: Structural analysis of the networks
   - Random Percolation: Network robustness against random failures
   - Degree Centrality Attack: Network robustness against targeted attacks based on node degree
   - Betweenness Centrality Attack: Network robustness against targeted attacks based on node betweenness
   - Network Metrics: Table of structural properties and critical thresholds

## Understanding the Visualizations

- **Degree Distribution**: Log-log plot showing the probability of nodes having a certain degree. The slope and shape reveal the network structure.
- **Percolation/Attack Plots**: Shows how the largest connected component (LCC) shrinks as edges are removed. The threshold where LCC reaches 5% is highlighted.
- **Network Metrics Table**: Compares structural properties and robustness thresholds across models. Highlighted rows indicate key thresholds. 