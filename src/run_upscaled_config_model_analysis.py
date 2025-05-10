#!/usr/bin/env python3
"""
Upscaled Configuration Model Analysis Pipeline

This script runs the complete upscaled configuration model analysis pipeline:
1. Generate upscaled configuration models (3500 nodes)
2. Run percolation analysis on the models
3. Run targeted attack analysis (degree and betweenness) on the models
"""

import os
import sys
import time
import subprocess

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def run_module(module_path, description):
    """Run a Python module and handle errors.
    
    Args:
        module_path: Path to the Python module to run
        description: Description of the module for output
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"Running {description}")
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, module_path], check=True)
        elapsed_time = time.time() - start_time
        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {description}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def main():
    """Main function to run the entire pipeline."""
    # Get the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define module paths
    upscaled_config_model_path = os.path.join(base_dir, "models", "upscaled_configuration_model.py")
    upscaled_percolation_path = os.path.join(base_dir, "models", "upscaled_config_model_percolation.py")
    upscaled_attack_path = os.path.join(base_dir, "models", "upscaled_config_model_targeted_attack.py")
    
    # Run upscaled configuration model generation
    if not run_module(upscaled_config_model_path, "Upscaled Configuration Model Generation"):
        print("Upscaled configuration model generation failed. Stopping pipeline.")
        return
    
    # Run upscaled percolation analysis
    if not run_module(upscaled_percolation_path, "Upscaled Configuration Model Percolation Analysis"):
        print("Upscaled percolation analysis failed.")
        return
    
    # Run upscaled targeted attack analysis
    if not run_module(upscaled_attack_path, "Upscaled Configuration Model Targeted Attack Analysis"):
        print("Upscaled targeted attack analysis failed.")
        return
    
    print_header("Upscaled Configuration Model Analysis Pipeline Complete")
    print("""
Results can be found in:
- config_models/upscaled/ (Generated models)
- results/ (Analysis CSV files and parameters)
- figures/ (Visualizations)
    """)

if __name__ == "__main__":
    main() 