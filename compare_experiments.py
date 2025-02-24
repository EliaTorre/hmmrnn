#!/usr/bin/env python
"""
Script for comparing results from multiple HMM-RNN experiments.
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compare multiple HMM-RNN experiment results")
    
    # Batch directory
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Directory containing the batch of experiments')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save comparison results (default: batch_dir)')
    parser.add_argument('--format', choices=['csv', 'json', 'md', 'html', 'all'], default='all',
                        help='Output format for comparison table')
    
    # Visualization options
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate comparison plots')
    
    return parser.parse_args()

def load_experiment_summaries(batch_dir):
    """Load summaries from all experiments in the batch"""
    batch_dir = Path(batch_dir)
    batch_summary_path = batch_dir / "batch_summary.json"
    
    if not batch_summary_path.exists():
        raise FileNotFoundError(f"Batch summary file not found: {batch_summary_path}")
    
    with open(batch_summary_path, "r") as f:
        batch_summary = json.load(f)
    
    experiments = []
    for exp_info in batch_summary["experiments"]:
        config_name = exp_info["config"]
        exp_dir = batch_dir / config_name
        
        if not exp_dir.exists():
            print(f"Warning: Experiment directory not found: {exp_dir}")
            continue
        
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            print(f"Warning: Summary file not found for experiment: {config_name}")
            continue
        
        with open(summary_path, "r") as f:
            exp_summary = json.load(f)
            exp_summary["config_name"] = config_name
            experiments.append(exp_summary)
    
    return batch_summary, experiments

def create_comparison_table(experiments):
    """Create a pandas DataFrame comparing experiment results"""
    data = []
    
    for exp in experiments:
        row = {
            "Config": exp["config_name"],
            "States": int(exp["config"].get("states", 0)),
            "Outputs": int(exp["config"].get("outputs", 0)),
            "Stay Prob": float(exp["config"].get("stay_prob", 0)),
            "Emission Method": exp["config"].get("emission_method", ""),
            "Hidden Size": int(exp["config"].get("hidden_size", 0)),
            "Learning Rate": exp["best_lr"],
            "Best Loss": exp["best_loss"],
            "Duration (min)": exp["experiment_duration_minutes"],
            "Explained Variance (Top 3)": sum(exp["explained_variance"][:3]) if "explained_variance" in exp else 0
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def save_comparison_table(df, output_dir, format='all'):
    """Save the comparison table in various formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    formats = ['csv', 'json', 'md', 'html'] if format == 'all' else [format]
    
    for fmt in formats:
        if fmt == 'csv':
            df.to_csv(output_dir / "comparison.csv", index=False)
        elif fmt == 'json':
            df.to_json(output_dir / "comparison.json", orient="records", indent=4)
        elif fmt == 'md':
            with open(output_dir / "comparison.md", "w") as f:
                f.write(df.to_markdown(index=False))
        elif fmt == 'html':
            with open(output_dir / "comparison.html", "w") as f:
                f.write(df.to_html(index=False))
    
    print(f"Comparison table saved to {output_dir}")

def generate_comparison_plots(experiments, output_dir):
    """Generate plots comparing different experiment parameters and results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting
    configs = [exp["config_name"] for exp in experiments]
    states = [int(exp["config"].get("states", 0)) for exp in experiments]
    losses = [exp["best_loss"] for exp in experiments]
    durations = [exp["experiment_duration_minutes"] for exp in experiments]
    
    # 1. States vs. Loss
    plt.figure(figsize=(10, 6))
    plt.scatter(states, losses, c=durations, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Duration (minutes)')
    
    for i, config in enumerate(configs):
        plt.annotate(config, (states[i], losses[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    plt.xlabel('Number of HMM States')
    plt.ylabel('Best Validation Loss')
    plt.title('Loss vs. Number of States')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "states_vs_loss.pdf")
    plt.savefig(output_dir / "states_vs_loss.png", dpi=300)
    plt.close()
    
    # 2. Collect explained variance data if available
    explained_variances = []
    for exp in experiments:
        if "explained_variance" in exp:
            ev = exp["explained_variance"]
            if isinstance(ev, list) and len(ev) > 0:
                # Only use the first 10 components or less
                ev = ev[:min(10, len(ev))]
                explained_variances.append((exp["config_name"], ev))
    
    if explained_variances:
        # Plot explained variance by principal component
        plt.figure(figsize=(12, 6))
        
        for config, ev in explained_variances:
            components = range(1, len(ev) + 1)
            plt.plot(components, np.cumsum(ev), 'o-', label=config)
        
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance by Configuration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "explained_variance.pdf")
        plt.savefig(output_dir / "explained_variance.png", dpi=300)
        plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

def create_html_dashboard(experiments, comparison_df, output_dir):
    """Create an HTML dashboard summarizing the experiments"""
    output_dir = Path(output_dir)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HMM-RNN Experiments Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .container { margin-bottom: 30px; }
            .images { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
            .image-container { max-width: 45%; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>HMM-RNN Experiments Dashboard</h1>
        
        <div class="container">
            <h2>Experiment Comparison</h2>
            {comparison_table}
        </div>
        
        <div class="container">
            <h2>Comparison Plots</h2>
            <div class="images">
                <div class="image-container">
                    <h3>States vs. Loss</h3>
                    <img src="states_vs_loss.png" alt="States vs. Loss">
                </div>
                <div class="image-container">
                    <h3>Explained Variance</h3>
                    <img src="explained_variance.png" alt="Explained Variance">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Insert the comparison table
    html = html.replace("{comparison_table}", comparison_df.to_html(index=False))
    
    # Write the HTML file
    with open(output_dir / "dashboard.html", "w") as f:
        f.write(html)
    
    print(f"HTML dashboard saved to {output_dir / 'dashboard.html'}")

def main():
    """Main function to compare experiment results"""
    args = parse_arguments()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(args.batch_dir) / "comparison"
    
    print(f"Loading experiment summaries from {args.batch_dir}")
    batch_summary, experiments = load_experiment_summaries(args.batch_dir)
    
    if not experiments:
        print("No valid experiment summaries found.")
        return
    
    print(f"Found {len(experiments)} experiments to compare")
    
    # Create comparison table
    df = create_comparison_table(experiments)
    save_comparison_table(df, args.output_dir, args.format)
    
    # Generate comparison plots
    if args.plot:
        generate_comparison_plots(experiments, args.output_dir)
        create_html_dashboard(experiments, df, args.output_dir)
    
    print("Comparison completed successfully.")

if __name__ == "__main__":
    main()
