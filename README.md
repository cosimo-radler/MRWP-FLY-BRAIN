# Drosophila Circuit Robustness Analysis

This project compares the bond-percolation thresholds of two Drosophila learning circuits to analyze their relative robustness to random synaptic loss.

## Research Question

How does the critical bond-percolation threshold qc of the ellipsoid-body subnetwork compare to that of the mushroom-body (Kenyon-cell) subnetwork under random edge removal, and what does this reveal about the relative robustness of visual- versus olfactory-learning circuits in Drosophila?

## Project Structure

- `src/`: Python scripts for data acquisition, processing, and analysis
- `data/`: Raw and processed connectome data
- `results/`: Analysis outputs (thresholds, statistics)
- `figures/`: Generated plots and visualizations

## Setup

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run data acquisition script:
   ```
   python src/data_acquisition.py
   ```

3. Run percolation analysis:
   ```
   python src/percolation_analysis.py
   ```

## Data Sources

- Hemibrain EM connectome
- Virtual Fly Brain / neuPrint

## Methods

- Network construction using NetworkX
- Bond-percolation simulation
- LCC (Largest Connected Component) analysis
- Statistical comparison of critical thresholds 