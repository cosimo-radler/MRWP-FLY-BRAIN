# Clustering-Preserved Configuration Models

This directory contains configuration models of Drosophila brain regions that preserve both the degree distribution and the clustering coefficient of the original networks.

## Overview

Standard configuration models preserve only the degree distribution of the original networks, but typically have much lower clustering coefficients. These models provide a more realistic null model by maintaining both properties.

## Directories

- `unscaled/`: Contains configuration models with the same number of nodes as the original networks
- `scaled/`: Contains configuration models scaled to 1500 nodes

## File Format

For each brain region (eb, fb, mb_kc) and scale type, the following files are generated:

- `{region}_{scale_type}_clustering_config_model.gexf`: Network in GEXF format
- `{region}_{scale_type}_clustering_config_model.pkl`: Network in pickle format for faster loading
- `{region}_{scale_type}_metrics_comparison.csv`: Comparison metrics between original and config models

## Algorithm

The models are created using a rewiring process that preserves the degree distribution while targeting a specific clustering coefficient:

1. First, a standard configuration model is created that preserves the degree distribution
2. Then, an edge-swapping procedure iteratively rewires edges to:
   - Increase clustering (by creating triangles) when current clustering < target clustering
   - Decrease clustering (by breaking triangles) when current clustering > target clustering
3. The process continues until the clustering coefficient is within a specified tolerance of the target

## Scripts

The following scripts are used to generate and analyze these models:

- `src/models/clustering_config_model.py`: Main script for generating the models
- `src/run_clustering_config_model.py`: Runner script to execute the model generation
- `src/analysis/verify_clustering_config_models.py`: Script to verify and compare model properties

## Example Usage

To generate the models:

```
python src/run_clustering_config_model.py
```

To analyze and compare models:

```
python src/analysis/verify_clustering_config_models.py
```

## References

This implementation is inspired by the following methods:

1. Bansal, S., Khandelwal, S., & Meyers, L. A. (2009). Exploring biological network structure with clustered random networks. BMC Bioinformatics, 10, 405.
2. Milo, R., Kashtan, N., Itzkovitz, S., Newman, M. E. J., & Alon, U. (2003). On the uniform generation of random graphs with prescribed degree sequences. arXiv:cond-mat/0312028. 