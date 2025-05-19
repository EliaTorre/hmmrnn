import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import os

from scripts.mechint import load_model

def run_model_and_compute_pca(model_path, input_size, hidden_size, time_steps=30000):
    """
    Run a model for the specified number of timesteps and compute PCA.
    
    Args:
        model_path (str): Path to the model file
        input_size (int): Input size for the model
        hidden_size (int): Hidden size for the model
        time_steps (int): Number of timesteps to run the model
        
    Returns:
        tuple: (rnn, pca, hidden_states, pca_result) or None if error occurs
    """
    try:
        # Load the model
        rnn = load_model(model_path, input_size=input_size, hidden_size=hidden_size)
        
        # Generate sequences
        print(f"Generating {time_steps} timesteps for model: {os.path.basename(model_path)}")
        rnn_data = rnn.gen_seq(time_steps=time_steps, dynamics_mode="full")
        
        # Extract hidden states
        hidden_states = rnn_data["h"]
        
        # Check for NaN values
        if np.isnan(hidden_states).any():
            print(f"Warning: NaN values detected in hidden states for model: {os.path.basename(model_path)}. Skipping.")
            return None
        
        # Compute PCA
        print("Computing PCA...")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(hidden_states)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return rnn, pca, hidden_states, pca_result
    except Exception as e:
        print(f"Error processing model {os.path.basename(model_path)}: {str(e)}")
        return None

def generate_trajectories_with_input(rnn, pca, num_samples=1, trajectory_length=500):
    """
    Generate trajectories with input, starting from random initial conditions.
    
    Args:
        rnn: The RNN model
        pca: Fitted PCA model
        num_samples (int): Number of initial conditions to sample
        trajectory_length (int): Length of each trajectory
        
    Returns:
        tuple: (trajectories, color_labels, outputs)
            trajectories: Array of shape (num_samples, trajectory_length, 2) for 2D PCA projections
            color_labels: Array of shape (num_samples, trajectory_length) for coloring
            outputs: Array of shape (num_samples, trajectory_length, output_size) for outputs
    """
    # Generate 30k timesteps to sample from
    print("Generating 30k timesteps to sample initial conditions...")
    rnn_data = rnn.gen_seq(time_steps=30000, dynamics_mode="full")
    hidden_states = rnn_data["h"]
    
    # Sample random initial conditions
    indices = np.random.choice(len(hidden_states), num_samples, replace=False)
    initial_states = hidden_states[indices]
    
    # Initialize arrays for trajectories, colors, and outputs
    trajectories = np.zeros((num_samples, trajectory_length, 2))
    color_labels = np.zeros((num_samples, trajectory_length), dtype=int)
    outputs = np.zeros((num_samples, trajectory_length, rnn.output_size))
    
    # Generate trajectories with input
    print(f"Generating {num_samples} trajectories with input...")
    for i in range(num_samples):
        h_current = torch.tensor(initial_states[i], dtype=torch.float32).to(rnn.device)
        trajectory = np.zeros((trajectory_length, rnn.hidden_size))
        
        # Run the model with input
        for t in range(trajectory_length):
            # Generate random input
            x = torch.normal(mean=0, std=1, size=(rnn.input_size,)).float().to(rnn.device)
            
            # Full dynamics (with input)
            h_current = torch.relu(x @ rnn.rnn.weight_ih_l0.data.T + h_current @ rnn.rnn.weight_hh_l0.data.T)
            trajectory[t] = h_current.cpu().detach().numpy()
            
            # Compute logits for coloring
            logits = h_current @ rnn.fc.weight.data.T
            
            # Calculate outputs using gumbel_softmax as specified
            output = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1)
            outputs[i, t] = output.cpu().detach().numpy()
            
            # Use argmax of gumbel_softmax output for coloring (not logits)
            color_labels[i, t] = torch.argmax(output).item()
        
        # Project trajectory to PCA space
        trajectories[i] = pca.transform(trajectory)[:, :2]  # Keep only first 2 PCs
    
    return trajectories, color_labels, outputs

def create_grid_combined_plot(model_paths, output_path):
    """
    Create a 1x4 grid plot showing trajectories for different models.
    
    Args:
        model_paths (list): List of model paths to process
        output_path (str): Path to save the output plot
    """
    # Define model parameters
    hidden_size = 150
    input_size = 100
    
    # Create figure with 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Define model titles
    model_titles = ["2 States", "3 States", "4 States", "5 States"]
    
    # Process each model
    for j, model_path in enumerate(model_paths):
        ax = axes[j]
        
        # Run model and compute PCA
        result = run_model_and_compute_pca(model_path, input_size, hidden_size)
        
        if result is None:
            print(f"Skipping model: {os.path.basename(model_path)}")
            continue
            
        rnn, pca, hidden_states, pca_result = result
        
        # Generate trajectories with input
        trajectories, color_labels, outputs = generate_trajectories_with_input(
            rnn, pca, num_samples=1, trajectory_length=500
        )
        
        # Loop through each trajectory
        for k in range(trajectories.shape[0]):
            traj = trajectories[k]
            color_label = color_labels[k]
            
            # Plot each segment with its corresponding color
            for l in range(len(traj) - 1):
                start = traj[l]
                end = traj[l + 1]
                color = colors[color_label[l]]
                
                # Plot the segment with thicker line
                ax.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=1.0)
                
                # Add an arrow at each timestep (between current and next point) with thicker head
                ax.arrow(
                    start[0], start[1],  # Starting point of the arrow
                    end[0] - start[0],  # x-direction
                    end[1] - start[1],  # y-direction
                    color=color,  # Arrow color matches the segment color
                    head_width=0.08,  # Increased size of the arrow
                    head_length=0.08,  # Increased length of the arrow head
                    length_includes_head=True,
                    lw=0.5  # Increased line width
                )
        
        # Set title for each subplot
        ax.set_title(model_titles[j], fontsize=14)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set labels
        if j == 0:
            ax.set_ylabel('PC2', fontsize=16)
        ax.set_xlabel('PC1', fontsize=16)
    
    # Create legend patches
    green_patch = Patch(color='darkgreen', label='Output 0')
    blue_patch = Patch(color='royalblue', label='Output 1')
    red_patch = Patch(color='darkred', label='Output 2')
    
    # Add the legend to the figure
    legend_handles = [green_patch, blue_patch, red_patch]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define model paths
    model_paths = [
        "Experiments/grid_search_20250412_021154/HMMTwo/hidden_150/input_100/seed_0/models/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth",
        "Experiments/grid_search_20250412_021154/HMMThree/hidden_150/input_100/seed_0/models/3HMM_3Outputs_linear_30kData_0.001lr_4.1Loss.pth",
        "Experiments/grid_search_20250412_021154/HMMFour/hidden_150/input_100/seed_0/models/4HMM_3Outputs_linear_30kData_0.001lr_4.6Loss.pth",
        "Experiments/grid_search_20250412_021154/HMMFive/hidden_150/input_100/seed_0/models/5HMM_3Outputs_linear_30kData_0.001lr_7.8Loss.pth"
    ]
    
    # Define output path
    output_path = "model_trajectories_grid.png"
    
    # Create the grid plot
    create_grid_combined_plot(model_paths, output_path)
    
    print(f"Plot saved to {output_path}")
