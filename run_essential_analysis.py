#!/usr/bin/env python3
"""
Essential Analysis Pipeline

This script runs only the essential components of the analysis pipeline to generate
the combined network figures with degree distribution, percolation, and targeted attack analysis.
"""

import os
import subprocess
import time

# Define the script directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')

def print_section(title):
    """Print a section header with the given title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_script(module_path, description):
    """Run a Python module with the given description."""
    print_section(description)
    cmd = ["python", "-m", module_path]
    subprocess.run(cmd, check=True)
    print(f"\n✓ Completed: {description}")
    
    # Small delay to ensure file writes complete
    time.sleep(1)

def main():
    """Run the essential analysis pipeline."""
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('config_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('Gephi Graphs/real_models', exist_ok=True)
    os.makedirs('Gephi Graphs/config_models', exist_ok=True)
    
    # Step 1: Data Acquisition
    run_script("src.data.data_acquisition", "Data Acquisition")
    
    # Step 2: Configuration Model Creation
    run_script("src.models.configuration_model", "Configuration Model Creation")
    
    # Step 3: Percolation Analysis
    run_script("src.models.config_model_percolation", "Percolation Analysis")
    
    # Step 4: Targeted Attack Analysis
    run_script("src.analysis.targeted_attack_analysis", "Targeted Attack Analysis")
    
    # Step 5: Create combined figures
    run_script("src.visualization.create_combined_network_figures", "Creating Combined Network Figures")
    
    print_section("ANALYSIS COMPLETE")
    print("The essential analysis pipeline has completed successfully.")
    print("Combined network figures are available in: src/figures/combined/")
    print("  - eb_combined_analysis.png")
    print("  - fb_combined_analysis.png")
    print("  - mb_kc_combined_analysis.png")
    print("\nEach figure includes:")
    print("  - Normalized degree distribution comparison")
    print("  - Percolation analysis")
    print("  - Targeted attack analysis")

if __name__ == "__main__":
    main() 