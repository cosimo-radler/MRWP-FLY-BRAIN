#!/usr/bin/env python3
"""
Path Utilities for Drosophila Circuit Robustness Analysis

This module provides consistent paths to important directories
and files across the project.
"""

import os

# Base project directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_MODELS_DIR = os.path.join(PROJECT_ROOT, "config_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Gephi directories
GEPHI_DIR = os.path.join(PROJECT_ROOT, "Gephi Graphs")
GEPHI_REAL_MODELS_DIR = os.path.join(GEPHI_DIR, "real_models")
GEPHI_CONFIG_MODELS_DIR = os.path.join(GEPHI_DIR, "config_models")

def ensure_directories_exist():
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, CONFIG_MODELS_DIR, RESULTS_DIR, FIGURES_DIR, 
                     GEPHI_DIR, GEPHI_REAL_MODELS_DIR, GEPHI_CONFIG_MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_network_path(network_type, model_type="real"):
    """Get path to a network file.
    
    Args:
        network_type: 'eb', 'fb', or 'mb_kc'
        model_type: 'real' or 'config'
        
    Returns:
        Path to the GEXF file
    """
    if model_type == "real":
        # Try Gephi directory first, then fall back to data directory
        gephi_path = os.path.join(GEPHI_REAL_MODELS_DIR, f"{network_type}_network.gexf")
        data_path = os.path.join(DATA_DIR, f"{network_type}_network.gexf")
        
        return gephi_path if os.path.exists(gephi_path) else data_path
    else:
        # Try Gephi config directory first, then fall back to config_models directory
        gephi_path = os.path.join(GEPHI_CONFIG_MODELS_DIR, f"{network_type}_config_model.gexf")
        config_path = os.path.join(CONFIG_MODELS_DIR, f"{network_type}_config_model.gexf")
        
        return gephi_path if os.path.exists(gephi_path) else config_path

def get_results_path(filename):
    """Get path to a results file.
    
    Args:
        filename: Name of the results file
        
    Returns:
        Path to the results file
    """
    return os.path.join(RESULTS_DIR, filename)

def get_figure_path(filename):
    """Get path to a figure file.
    
    Args:
        filename: Name of the figure file
        
    Returns:
        Path to the figure file
    """
    return os.path.join(FIGURES_DIR, filename) 