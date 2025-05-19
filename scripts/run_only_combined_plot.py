import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import seaborn as sns

# Add the current directory to the path so we can import from scripts
sys.path.append('.')

from scripts.pipeline import create_combined_grid_plot, extract_rnn_models, run_model_and_compute_pca
from scripts.pipeline import generate_trajectories_no_input, generate_trajectories_with_input

def run_only_combined_plot():
    """
    Run only the part of the pipeline that creates the combined grid plot.
    This will extract the necessary data from the models but skip generating all the individual plots.
    """
    print("Extracting RNN models...")
    grouped_models = extract_rnn_models()
    
    # Process each model type
    model_types = ["HMMTwo", "HMMThree", "HMMFour", "HMMFive"]
    
    # For the special plot (Hidden=150, Input=100)
    special_trajectories_no_in = {}
    special_color_labels_no_in = {}
    special_fixed_points_no_in = {}
    special_trajectories_in = {}
    special_color_labels_in = {}
    special_rnn_pca_dict = {}
    
    # Process each model type to extract data for the combined plot
    for model_type in model_types:
        print(f"\nProcessing {model_type} model for combined plot...")
        
        # Check if the model exists
        if 150 not in grouped_models[model_type] or 100 not in grouped_models[model_type][150]:
            print(f"No model found for {model_type} with hidden_size=150, input_size=100")
            continue
        
        # Process seed 0
        seed = 0
        if seed not in grouped_models[model_type][150][100]:
            print(f"No model found for {model_type} with hidden_size=150, input_size=100, seed={seed}")
            continue
        
        # Get the first model for this configuration
        model_path, config = grouped_models[model_type][150][100][seed][0]
        print(f"Processing model: {os.path.basename(model_path)}")
        
        # Run model and compute PCA
        result = run_model_and_compute_pca(
            model_path, input_size=100, hidden_size=150
        )
        
        # Skip this model if there was an error
        if result is None:
            print(f"Skipping model: {os.path.basename(model_path)}")
            continue
            
        rnn, pca, hidden_states, pca_result = result
        
        # Generate trajectories without input and compute fixed point
        trajectories_no_in, color_labels_no_in, fixed_point_no_in = generate_trajectories_no_input(
            rnn, pca, num_samples=100, trajectory_length=500
        )
        
        # Generate trajectories with input (without fixed point)
        trajectories_in, color_labels_in, _ = generate_trajectories_with_input(
            rnn, pca, num_samples=1, trajectory_length=500, fixed_point_pca=None
        )
        
        # Store data for special plot
        special_trajectories_no_in[model_type] = trajectories_no_in
        special_color_labels_no_in[model_type] = color_labels_no_in
        special_fixed_points_no_in[model_type] = fixed_point_no_in
        special_trajectories_in[model_type] = trajectories_in
        special_color_labels_in[model_type] = color_labels_in
        special_rnn_pca_dict[model_type] = (rnn, pca)
    
    # Create the combined grid plot
    print("\nCreating combined grid plot...")
    create_combined_grid_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, 
                             special_fixed_points_no_in, special_trajectories_in, special_color_labels_in, 
                             special_rnn_pca_dict, ".")
    
    print("\nCombined grid plot created successfully!")

if __name__ == "__main__":
    run_only_combined_plot()
