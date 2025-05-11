#!/usr/bin/env python3
"""
Run Script for Clustering-Preserving Configuration Models

This script runs the clustering_config_model.py script to generate
configuration models that preserve both the degree distribution and 
the clustering coefficient of the original brain region networks.
"""

import os
import sys
import subprocess
import time

def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the configuration model script
    model_script = os.path.join(script_dir, "models", "clustering_config_model.py")
    
    # Check if the script exists
    if not os.path.exists(model_script):
        print(f"Error: Script not found at {model_script}")
        sys.exit(1)
    
    print(f"Running clustering-preserving configuration model generation...")
    print(f"Script path: {model_script}")
    
    # Run the script
    start_time = time.time()
    
    try:
        subprocess.run([sys.executable, model_script], check=True)
        
        # Calculate and print elapsed time
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds")
        print("Configuration models with preserved clustering coefficients have been generated successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 