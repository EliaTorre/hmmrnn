import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import os
import seaborn as sns
import pickle

from scripts.pipeline import create_combined_grid_plot

def regenerate_combined_plot():
    """
    Regenerate only the combined grid plot without re-running the entire pipeline.
    This assumes that the necessary data has already been generated.
    """
    # Check if the necessary data exists
    if not os.path.exists("plot_data.pkl"):
        print("Error: plot_data.pkl not found. Need to save the data first.")
        return False
    
    # Load the saved data
    with open("plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Extract the data
    model_types = data["model_types"]
    special_trajectories_no_in = data["special_trajectories_no_in"]
    special_color_labels_no_in = data["special_color_labels_no_in"]
    special_fixed_points_no_in = data["special_fixed_points_no_in"]
    special_trajectories_in = data["special_trajectories_in"]
    special_color_labels_in = data["special_color_labels_in"]
    special_rnn_pca_dict = data["special_rnn_pca_dict"]
    
    # Create the combined grid plot
    create_combined_grid_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, 
                             special_fixed_points_no_in, special_trajectories_in, special_color_labels_in, 
                             special_rnn_pca_dict, ".")
    
    return True

def save_plot_data():
    """
    Extract the necessary data from the existing pipeline results and save it to a pickle file.
    This function should be run once to save the data needed for regenerate_combined_plot.
    """
    # Process each model type
    model_types = ["HMMTwo", "HMMThree", "HMMFour", "HMMFive"]
    
    # For the special plot (Hidden=150, Input=100)
    special_trajectories_no_in = {}
    special_color_labels_no_in = {}
    special_fixed_points_no_in = {}
    special_trajectories_in = {}
    special_color_labels_in = {}
    special_rnn_pca_dict = {}
    
    # Load data from existing files
    for model_type in model_types:
        # Load trajectories without input
        traj_no_in_path = f"traj_no_in/special_traj_no_in_grid.png"
        if not os.path.exists(traj_no_in_path):
            print(f"Warning: {traj_no_in_path} not found. Some data may be missing.")
        
        # Load trajectories with input
        traj_in_path = f"traj_in/special_traj_in_grid.png"
        if not os.path.exists(traj_in_path):
            print(f"Warning: {traj_in_path} not found. Some data may be missing.")
        
        # Load variance contour plots
        var_contour_path = f"variance_contours/special_variance_contour_grid.png"
        if not os.path.exists(var_contour_path):
            print(f"Warning: {var_contour_path} not found. Some data may be missing.")
    
    # Since we can't directly extract the data from the PNG files,
    # we need to run a modified version of the pipeline to extract just the data we need
    
    print("To save the plot data, you need to run a modified version of the pipeline.")
    print("Please add the following code at the end of the run_pipeline function in scripts/pipeline.py:")
    print("\n# Save data for regenerating the combined grid plot")
    print("import pickle")
    print("data = {")
    print("    'model_types': model_types,")
    print("    'special_trajectories_no_in': special_trajectories_no_in,")
    print("    'special_color_labels_no_in': special_color_labels_no_in,")
    print("    'special_fixed_points_no_in': special_fixed_points_no_in,")
    print("    'special_trajectories_in': special_trajectories_in,")
    print("    'special_color_labels_in': special_color_labels_in,")
    print("    'special_rnn_pca_dict': special_rnn_pca_dict")
    print("}")
    print("with open('plot_data.pkl', 'wb') as f:")
    print("    pickle.dump(data, f)")
    
    return False

if __name__ == "__main__":
    # Check if plot_data.pkl exists
    if os.path.exists("plot_data.pkl"):
        print("Regenerating combined grid plot...")
        regenerate_combined_plot()
    else:
        print("Plot data not found. Need to save the data first.")
        save_plot_data()
