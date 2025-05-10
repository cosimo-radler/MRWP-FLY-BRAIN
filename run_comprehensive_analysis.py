#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline Script

This script runs the entire analysis pipeline:
1. Creating scaled configuration models
2. Creating unscaled configuration models
3. Running percolation analysis on all models
4. Running targeted attack analysis on all models
5. Generating comprehensive visualizations

This allows for a complete analysis of original networks, scaled configuration models, 
and unscaled configuration models to understand the impact of scaling on network properties.
"""

import os
import subprocess
import time
import argparse

# Path constants
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")
VISUALIZATION_DIR = os.path.join(SRC_DIR, "visualization")

# Define the analysis steps
ANALYSIS_STEPS = [
    {
        "name": "Create scaled configuration models",
        "script": os.path.join(MODELS_DIR, "configuration_model.py"),
        "description": "Generate scaled configuration models with 1500 nodes"
    },
    {
        "name": "Create unscaled configuration models",
        "script": os.path.join(MODELS_DIR, "unscaled_configuration_model.py"),
        "description": "Generate unscaled configuration models that preserve original node counts"
    },
    {
        "name": "Run bond percolation on scaled models",
        "script": os.path.join(MODELS_DIR, "config_model_percolation.py"),
        "description": "Run bond percolation analysis on original networks and scaled config models"
    },
    {
        "name": "Run bond percolation on unscaled models",
        "script": os.path.join(MODELS_DIR, "unscaled_config_model_percolation.py"),
        "description": "Run bond percolation analysis on original networks and unscaled config models"
    },
    {
        "name": "Run targeted attacks on scaled models",
        "script": os.path.join(MODELS_DIR, "targeted_attack_config_models.py"),
        "description": "Run targeted attacks (degree and betweenness) on scaled configuration models"
    },
    {
        "name": "Run targeted attacks on unscaled models",
        "script": os.path.join(MODELS_DIR, "targeted_attack_unscaled_config_models.py"),
        "description": "Run targeted attacks (degree and betweenness) on unscaled configuration models"
    },
    {
        "name": "Create comprehensive visualizations",
        "script": os.path.join(VISUALIZATION_DIR, "compare_all_models.py"),
        "description": "Generate comprehensive comparisons between all model types"
    }
]

def run_analysis_step(step, args):
    """Run a single analysis step.
    
    Args:
        step: Dictionary containing step information
        args: Command line arguments
        
    Returns:
        Boolean indicating success or failure
    """
    print(f"\n{'=' * 80}")
    print(f"STEP: {step['name']}")
    print(f"DESCRIPTION: {step['description']}")
    print(f"{'=' * 80}\n")
    
    # Check if the script exists
    if not os.path.exists(step['script']):
        print(f"ERROR: Script not found: {step['script']}")
        return False
    
    # Run the script with appropriate Python interpreter
    try:
        if args.python_path:
            cmd = [args.python_path, step['script']]
        else:
            cmd = ["python3", step['script']]
            
        start_time = time.time()
        process = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Step failed with exit code {e.returncode}")
        return False
    
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        return False

def main():
    """Main function to run the entire analysis pipeline."""
    parser = argparse.ArgumentParser(description="Run comprehensive network analysis pipeline")
    parser.add_argument("--python-path", help="Path to Python interpreter (default: uses 'python3')")
    parser.add_argument("--start-step", type=int, default=1, 
                        help="Start from a specific step (1-indexed)")
    parser.add_argument("--end-step", type=int, 
                        help="End at a specific step (1-indexed)")
    parser.add_argument("--steps", type=int, nargs="+", 
                        help="Run only specific steps (1-indexed)")
    args = parser.parse_args()
    
    print(f"{'=' * 80}")
    print(f"STARTING COMPREHENSIVE ANALYSIS PIPELINE")
    print(f"{'=' * 80}")
    
    # List available steps
    print("\nAvailable analysis steps:")
    for i, step in enumerate(ANALYSIS_STEPS, 1):
        print(f"{i}. {step['name']} - {step['description']}")
    
    # Determine which steps to run
    steps_to_run = []
    if args.steps:
        # Run only specific steps
        steps_to_run = [i-1 for i in args.steps if 0 < i <= len(ANALYSIS_STEPS)]
    else:
        # Run range of steps
        start = max(0, args.start_step - 1)
        end = args.end_step if args.end_step and args.end_step <= len(ANALYSIS_STEPS) else len(ANALYSIS_STEPS)
        steps_to_run = range(start, end)
    
    # Run the selected steps
    results = []
    total_start_time = time.time()
    
    for i in steps_to_run:
        if i < 0 or i >= len(ANALYSIS_STEPS):
            print(f"WARNING: Step {i+1} is out of range, skipping")
            continue
            
        step = ANALYSIS_STEPS[i]
        success = run_analysis_step(step, args)
        results.append((i+1, step['name'], success))
    
    total_end_time = time.time()
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS PIPELINE SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds\n")
    
    for step_num, step_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"Step {step_num}: {step_name} - {status}")
    
    # Check for any failures
    if any(not success for _, _, success in results):
        print("\nWARNING: Some analysis steps failed. Check the output for details.")
        return 1
    else:
        print("\nAll requested analysis steps completed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 