# Drosophila Neural Circuit Robustness Analysis

This project analyzes the robustness of neural circuits in the Drosophila brain, specifically focusing on:
- Ellipsoid Body (EB)
- Fan-shaped Body (FB)
- Mushroom Body Kenyon Cells (MB-KC)

## Project Overview

This research investigates the structural robustness of neural circuits in the Drosophila brain by 
analyzing how these networks respond to targeted attacks and random failures. We compare real neural 
networks to configuration models with preserved degree distributions to identify unique properties
that might contribute to robustness.

## Project Structure

### Source Code (`src/`)

- **data/**: Data acquisition from the hemibrain dataset
  - `data_acquisition.py`: Fetches connectome data using the neuPrint API

- **models/**: Configuration model creation
  - `configuration_model.py`: Creates configuration models preserving degree distributions
  - `config_model_percolation.py`: Percolation analysis on configuration models
  - `targeted_attack_config_models.py`: Targeted attack simulations on configuration models
  - `export_enhanced_config_models.py`: Exports models with enhanced properties

- **analysis/**: Network analysis
  - `targeted_attack_analysis.py`: Analysis of targeted attacks on networks
  - `percolation_analysis.py`: Percolation analysis on real networks
  - `betweenness_attack_analysis.py`: Betweenness-based attack simulations
  - `network_metrics_comparison.py`: Comparison of network metrics

- **visualization/**: Visualizing results
  - `network_visualization.py`: Visualization of network structures
  - `create_percolation_comparison.py`: Creates percolation comparison visualizations
  - `create_multipanel_attack_comparison.py`: Creates multipanel attack comparison visualizations
  - `create_combined_network_figures.py`: Creates combined analysis figures per network

### Data & Output Directories (Project Root)

- **data/**: Contains raw and processed network data
- **config_models/**: Contains configuration model network files
- **Gephi Graphs/**: Contains Gephi visualization files
  - `real_models/`: Gephi files for real neural networks
  - `config_models/`: Gephi files for configuration models
- **results/**: Contains analysis results in CSV and JSON format
- **figures/**: Contains visualization outputs and plots
  - `combined/`: Contains the simplified combined analysis figures

## Key Findings

- Neural circuits in the Drosophila brain show different levels of robustness to targeted attacks
- Configuration models reveal that degree distribution alone does not fully explain the observed robustness
- The Mushroom Body Kenyon Cell network shows particularly high resilience to targeted attacks

## Getting Started

### Full Analysis Pipeline

To run the full analysis, execute each step sequentially:

```
python -m src.data.data_acquisition
python -m src.models.configuration_model
python -m src.analysis.targeted_attack_analysis
python -m src.analysis.percolation_analysis
python -m src.visualization.create_multipanel_attack_comparison
```

### Simplified Analysis Pipeline

For a simplified analysis that generates only the essential combined figures, run:

```
./run_essential_analysis.py
```

This script will:
1. Acquire data from the hemibrain dataset (or generate sample data)
2. Create configuration models preserving degree distributions
3. Perform percolation analysis
4. Run targeted attack simulations
5. Generate one combined figure per network showing:
   - Normalized degree distribution comparison
   - Percolation performance comparison
   - Targeted attack analysis

The combined figures will be saved in `src/figures/combined/`.

## Combined Analysis Figures

Each combined figure contains:
- Top row: Normalized in-degree and out-degree distributions comparing original and configuration models
- Bottom left: Percolation analysis comparing the robustness of original and configuration models
- Bottom right: Targeted attack analysis comparing degree-based vs random node removal

These figures provide a concise visual comparison of the structural properties and robustness characteristics of each neural circuit, including both uniform (percolation) and targeted attack scenarios.

## Requirements

- Python 3.7+
- NetworkX
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Requests

See `requirements.txt` for exact version requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Interactive Dashboard for Network Model Comparison

A new interactive web dashboard has been added to visualize and compare different network models!

## Running the Dashboard

1. Install dependencies:
   ```
   pip install dash dash-bootstrap-components plotly
   ```

2. Run the dashboard:
   ```
   python src/visualization/dashboard.py
   ```

3. Open your browser to:
   ```
   http://127.0.0.1:8050/
   ```

4. Or simply open `index.html` in your browser for project information and a link to the dashboard.

## Features

- Interactive toggling of models
- Multiple visualization tabs for degree distributions, percolation results, and targeted attacks
- Comprehensive metrics comparison
- Select different brain regions (EB, FB, MB-KC)

## Dashboard Screenshots

| Degree Distribution | Network Metrics |
|:-----------:|:-----------:|
| ![Degree Distribution](figures/dashboard_screenshot_1.png) | ![Network Metrics](figures/dashboard_screenshot_2.png) |

--- 