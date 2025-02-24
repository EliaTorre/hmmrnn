#!/usr/bin/env python
"""
Script for running multiple HMM-RNN experiments in sequence.
"""

import os
import argparse
import json
import datetime
import importlib
import subprocess
from pathlib import Path

import scripts.config as config_module

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run multiple HMM-RNN experiments in sequence")
    
    # Experiment directory
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='Base directory for all experiments')
    
    # Configuration selection
    parser.add_argument('--configs', type=str, nargs='+', default=['HMMSmall', 'HMMMedium', 'HMMLarge'],
                        help='List of configuration classes to run (default: HMMSmall HMMMedium HMMLarge)')
    
    # Experiment parameters
    parser.add_argument('--batch_name', type=str, default=None,
                        help='Batch name for this set of experiments (default: timestamp)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs for all experiments (overrides config)')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=None,
                        help='Learning rates to try for all experiments (overrides config)')
    
    # Execution method
    parser.add_argument('--method', choices=['subprocess', 'direct'], default='direct',
                        help='Method to run experiments: subprocess (calls main.py) or direct (imports main)')
    
    # Additional parameters to pass to main.py
    parser.add_argument('--additional_args', type=str, default='',
                        help='Additional arguments to pass to main.py (for subprocess method)')
    
    return parser.parse_args()

def setup_batch_directory(args):
    """Set up the batch experiment directory"""
    # Create base experiments directory
    base_dir = Path(args.experiments_dir)
    base_dir.mkdir(exist_ok=True)
    
    # Create batch directory with timestamp if not specified
    if args.batch_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.batch_name = f"batch_{timestamp}"
    
    batch_dir = base_dir / args.batch_name
    batch_dir.mkdir(exist_ok=True)
    
    # Create a summary file
    summary = {
        "batch_name": args.batch_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "configs": args.configs,
        "overrides": {
            "epochs": args.epochs,
            "learning_rates": args.learning_rates
        },
        "experiments": []
    }
    
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    return batch_dir, summary

def run_experiment_subprocess(config_name, batch_dir, args):
    """Run an experiment as a subprocess (calling main.py)"""
    # Build command
    experiment_dir = batch_dir / config_name
    cmd = [
        "python", "main.py",
        "--config", config_name,
        "--exp_name", str(experiment_dir)
    ]
    
    # Add overrides if specified
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.learning_rates is not None:
        cmd.extend(["--lr"] + [str(lr) for lr in args.learning_rates])
    
    # Add any additional arguments
    if args.additional_args:
        cmd.extend(args.additional_args.split())
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the subprocess
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save output
    with open(experiment_dir / "process_output.txt", "w") as f:
        f.write("STDOUT:\n")
        f.write(process.stdout)
        f.write("\nSTDERR:\n")
        f.write(process.stderr)
    
    return {
        "config": config_name,
        "exit_code": process.returncode,
        "success": process.returncode == 0
    }

def run_experiment_direct(config_name, batch_dir, args):
    """Run an experiment directly (importing main)"""
    import sys
    from io import StringIO
    import contextlib
    
    # Capture stdout
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    # Create the experiment directory
    experiment_dir = batch_dir / config_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Store original sys.argv and set new one for main.py
    original_argv = sys.argv
    
    # Build new argv
    sys.argv = [
        "main.py",
        "--config", config_name,
        "--exp_name", str(experiment_dir)
    ]
    
    # Add overrides if specified
    if args.epochs is not None:
        sys.argv.extend(["--epochs", str(args.epochs)])
    if args.learning_rates is not None:
        sys.argv.extend(["--lr"] + [str(lr) for lr in args.learning_rates])
    
    # Add any additional arguments
    if args.additional_args:
        sys.argv.extend(args.additional_args.split())
    
    print(f"Running with args: {sys.argv}")
    
    # Import and run main function
    success = False
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Dynamically import main
            main_module = importlib.import_module("main")
            main_module.main()
        success = True
    except Exception as e:
        stderr_capture.write(f"Error running experiment: {str(e)}\n")
        import traceback
        stderr_capture.write(traceback.format_exc())
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    # Save output
    with open(experiment_dir / "process_output.txt", "w") as f:
        f.write("STDOUT:\n")
        f.write(stdout_capture.getvalue())
        f.write("\nSTDERR:\n")
        f.write(stderr_capture.getvalue())
    
    return {
        "config": config_name,
        "success": success
    }

def update_batch_summary(batch_dir, summary, experiment_result):
    """Update the batch summary with experiment results"""
    summary["experiments"].append(experiment_result)
    
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

def main():
    """Main function to run multiple experiments"""
    args = parse_arguments()
    print(f"Starting batch run with {len(args.configs)} configurations")
    
    # Set up the batch directory
    batch_dir, summary = setup_batch_directory(args)
    print(f"Batch directory: {batch_dir}")
    
    # Run each experiment
    for config_name in args.configs:
        print(f"="*60)
        print(f"Running experiment with config: {config_name}")
        print(f"="*60)
        
        # Check if the config exists
        if not hasattr(config_module, config_name):
            print(f"Configuration '{config_name}' not found in config_module. Skipping.")
            experiment_result = {
                "config": config_name,
                "success": False,
                "error": "Configuration not found"
            }
        else:
            # Run the experiment
            if args.method == 'subprocess':
                experiment_result = run_experiment_subprocess(config_name, batch_dir, args)
            else:  # direct method
                experiment_result = run_experiment_direct(config_name, batch_dir, args)
        
        # Update the batch summary
        update_batch_summary(batch_dir, summary, experiment_result)
        
        print(f"Experiment {config_name} completed. Success: {experiment_result['success']}")
    
    print(f"="*60)
    print(f"All experiments completed.")
    print(f"Results saved in: {batch_dir}")
    print(f"="*60)

if __name__ == "__main__":
    main()
