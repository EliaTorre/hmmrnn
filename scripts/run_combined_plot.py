import os
import pickle
import numpy as np
import torch

from scripts.pipeline import create_combined_grid_plot

def run_combined_plot():
    """
    Run only the combined grid plot function from the pipeline.
    This assumes that the necessary data files already exist.
    """
    # Check if the necessary directories exist
    required_dirs = ["traj_no_in", "traj_in", "variance_contours"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Error: Directory '{dir_name}' not found. Make sure you've run the pipeline before.")
            return False
    
    # Define model types
    model_types = ["HMMTwo", "HMMThree", "HMMFour", "HMMFive"]
    
    # Load data from existing files
    # Since we can't directly load the data from the pipeline without re-running it,
    # we'll use a simplified approach that loads the necessary data from the special plot files
    
    print("Loading data from existing files...")
    
    # Create empty dictionaries to store the data
    special_trajectories_no_in = {}
    special_color_labels_no_in = {}
    special_fixed_points_no_in = {}
    special_trajectories_in = {}
    special_color_labels_in = {}
    special_rnn_pca_dict = {}
    
    # Check if the special plot files exist
    special_no_in_path = "traj_no_in/special_traj_no_in_grid.png"
    special_in_path = "traj_in/special_traj_in_grid.png"
    special_var_path = "variance_contours/special_variance_contour_grid.png"
    
    if not os.path.exists(special_no_in_path) or not os.path.exists(special_in_path) or not os.path.exists(special_var_path):
        print("Warning: Special plot files not found. The combined grid plot may not be accurate.")
    
    # Since we can't extract the data from the PNG files directly,
    # we'll use a workaround by running the create_combined_grid_plot function with dummy data
    
    # Create dummy data for each model type
    for model_type in model_types:
        # Create dummy trajectories (2D arrays)
        dummy_traj = np.zeros((1, 10, 2))
        dummy_colors = np.zeros((1, 10), dtype=int)
        dummy_fixed_point = np.zeros(2)
        
        # Add to dictionaries
        special_trajectories_no_in[model_type] = dummy_traj
        special_color_labels_no_in[model_type] = dummy_colors
        special_fixed_points_no_in[model_type] = dummy_fixed_point
        special_trajectories_in[model_type] = dummy_traj
        special_color_labels_in[model_type] = dummy_colors
        
        # Create dummy RNN and PCA
        class DummyRNN:
            def __init__(self):
                self.rnn = type('obj', (object,), {
                    'weight_ih_l0': type('obj', (object,), {'data': torch.zeros(1, 1)}),
                    'weight_hh_l0': type('obj', (object,), {'data': torch.zeros(1, 1)})
                })
                self.hidden_size = 150
                self.input_size = 100
                self.device = 'cpu'
        
        class DummyPCA:
            def __init__(self):
                self.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
                
            def transform(self, data):
                return np.zeros((data.shape[0], 2))
        
        special_rnn_pca_dict[model_type] = (DummyRNN(), DummyPCA())
    
    # Run the create_combined_grid_plot function
    print("Creating combined grid plot...")
    create_combined_grid_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, 
                             special_fixed_points_no_in, special_trajectories_in, special_color_labels_in, 
                             special_rnn_pca_dict, ".")
    
    print("Done! The combined grid plot has been created with the new 3x4 layout.")
    print("Note: Since we used dummy data, the plot may not show actual trajectories.")
    print("To see the actual trajectories, you can use the regenerate_combined_plot_direct.py script instead.")
    
    return True

if __name__ == "__main__":
    run_combined_plot()
