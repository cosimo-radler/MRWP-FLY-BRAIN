# Drosophila Neural Circuit Robustness Analysis

This project analyzes the robustness of neural circuits in the Drosophila brain, specifically focusing on:
- Ellipsoid Body (EB)
- Fan-shaped Body (FB)
- Mushroom Body Kenyon Cells (MB-KC)

## Project Structure

### Source Code (`src/`)

- **data/**: Data acquisition from the hemibrain dataset
  - `data_acquisition.py`: Fetches connectome data using the neuPrint API

- **models/**: Configuration model creation
  - `configuration_model.py`: Creates configuration models preserving degree distributions
  - `clustering_config_model.py`: Creates configuration models preserving both degree distributions and clustering coefficients
  - `config_model_percolation.py`: Percolation analysis on configuration models
  - `targeted_attack_config_models.py`: Targeted attack simulations on configuration models
  - `export_enhanced_config_models.py`: Exports models with enhanced properties

- **analysis/**: Network analysis
  - `targeted_attack_analysis.py`: Analysis of targeted attacks on networks
  - `percolation_analysis.py`: Percolation analysis on real networks
  - `betweenness_attack_analysis.py`: Betweenness-based attack simulations
  - `network_metrics_comparison.py`: Comparison of network metrics
  - `verify_clustering_config_models.py`: Verification of clustering-preserved configuration models

- **visualization/**: Visualizing results
  - `network_visualization.py`: Visualization of network structures
  - `create_percolation_comparison.py`: Creates percolation comparison visualizations
  - `create_multipanel_attack_comparison.py`: Creates multipanel attack comparison visualizations

- **utils/**: Utility functions
  - `enhance_gephi_files.py`: Enhances Gephi files with additional metrics

### Data & Output Directories (Project Root)

- **data/**: Contains raw and processed network data
- **config_models/**: Contains configuration model network files
  - `clustering/`: Contains configuration models preserving clustering coefficients
    - `unscaled/`: Unscaled models with original node count
    - `scaled/`: Scaled models with 1500 nodes
- **Gephi Graphs/**: Contains Gephi visualization files
  - `real_models/`: Gephi files for real neural networks
  - `config_models/`: Gephi files for configuration models
- **results/**: Contains analysis results in CSV and JSON format
- **figures/**: Contains visualization outputs and plots

## Getting Started

To run the analysis, first acquire the data using:

```
python -m src.data.data_acquisition
```

Then create configuration models:

```
python -m src.models.configuration_model
```

To create configuration models that preserve clustering coefficient:

```
python src/run_clustering_config_model.py
```

Run the analyses:

```
python -m src.analysis.targeted_attack_analysis
python -m src.analysis.percolation_analysis
```

Verify clustering-preserved configuration models:

```
python src/analysis/verify_clustering_config_models.py
```

And finally visualize the results:

```
python -m src.visualization.create_multipanel_attack_comparison
``` 