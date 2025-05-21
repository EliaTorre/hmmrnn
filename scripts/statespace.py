import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import torch
from collections import defaultdict
import os
import seaborn as sns

from scripts.metrics import get_specific_models
from scripts.mechint import load_model

def extract_rnn_models():
    """Extract all RNN models trained, similar to what's done in scripts/plots.py."""
    # Get models with their configurations
    models_with_configs = get_specific_models(with_config=True)
    
    # Group models by type, hidden size, input size, and seed
    grouped_models = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )
    
    for model_path, config in models_with_configs:
        if config is None:
            continue
            
        # Extract model type from path
        path_parts = model_path.split('/')
        model_type_idx = path_parts.index("TrainedModels") + 1
        model_type = path_parts[model_type_idx]
        hidden_size = path_parts[model_type_idx + 1]
        input_size = path_parts[model_type_idx + 2]
        seed = path_parts[model_type_idx + 3]
        
        # Extract numeric values from the path components
        hidden_size_val = int(hidden_size.split('_')[1])
        input_size_val = int(input_size.split('_')[1])
        seed_val = int(seed.split('_')[1])
        
        # Add model to the grouped dictionary
        grouped_models[model_type][hidden_size_val][input_size_val][seed_val].append((model_path, config))
    
    return grouped_models

def run_model_and_compute_pca(model_path, input_size, hidden_size, time_steps=30000):
    """Run a model for the specified number of timesteps and compute PCA."""
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

def generate_trajectories_no_input(rnn, pca, num_samples=100, trajectory_length=500):
    """Generate trajectories without input, starting from random initial conditions."""
    # Generate 30k timesteps to sample from
    print("Generating 30k timesteps to sample initial conditions...")
    rnn_data = rnn.gen_seq(time_steps=30000, dynamics_mode="full")
    hidden_states = rnn_data["h"]
    
    # Sample random initial conditions
    indices = np.random.choice(len(hidden_states), num_samples, replace=False)
    initial_states = hidden_states[indices]
    
    # Initialize arrays for trajectories and colors
    trajectories = np.zeros((num_samples, trajectory_length, 2))
    color_labels = np.zeros((num_samples, trajectory_length), dtype=int)
    
    # Array to store final hidden states for fixed point calculation
    final_hidden_states = np.zeros((num_samples, rnn.hidden_size))
    
    # Generate trajectories without input
    print(f"Generating {num_samples} trajectories without input...")
    for i in range(num_samples):
        h_current = torch.tensor(initial_states[i], dtype=torch.float32).to(rnn.device)
        trajectory = np.zeros((trajectory_length, rnn.hidden_size))
        
        # Run the model without input (recurrence only)
        for t in range(trajectory_length):
            # Use recurrence_only mode
            h_current = torch.relu(h_current @ rnn.rnn.weight_hh_l0.data.T)
            trajectory[t] = h_current.cpu().detach().numpy()
            
            # Compute logits for coloring
            logits = h_current @ rnn.fc.weight.data.T
            color_labels[i, t] = torch.argmax(logits).item()
        
        # Store the final hidden state for fixed point calculation
        final_hidden_states[i] = trajectory[-1]
        
        # Project trajectory to PCA space
        trajectories[i] = pca.transform(trajectory)[:, :2]  # Keep only first 2 PCs
    
    # Calculate the fixed point by averaging the final hidden states
    fixed_point_hidden = np.mean(final_hidden_states, axis=0)
    
    # Project the fixed point to PCA space
    fixed_point_pca = pca.transform(fixed_point_hidden.reshape(1, -1))[0, :2]
    
    print(f"Fixed point in PCA space: {fixed_point_pca}")
    
    return trajectories, color_labels, fixed_point_pca

def generate_trajectories_with_input(rnn, pca, num_samples=1, trajectory_length=500, fixed_point_pca=None):
    """Generate trajectories with input, starting from random initial conditions."""
    # Generate 30k timesteps to sample from
    print("Generating 30k timesteps to sample initial conditions...")
    rnn_data = rnn.gen_seq(time_steps=30000, dynamics_mode="full")
    hidden_states = rnn_data["h"]
    
    # Sample random initial conditions
    indices = np.random.choice(len(hidden_states), num_samples, replace=False)
    initial_states = hidden_states[indices]
    
    # Initialize arrays for trajectories and colors
    trajectories = np.zeros((num_samples, trajectory_length, 2))
    color_labels = np.zeros((num_samples, trajectory_length), dtype=int)
    
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
            color_labels[i, t] = torch.argmax(logits).item()
        
        # Project trajectory to PCA space
        trajectories[i] = pca.transform(trajectory)[:, :2]  # Keep only first 2 PCs
    
    return trajectories, color_labels, fixed_point_pca

def plot_trajectories_2d(trajectories, color_labels, plot_title, save_path, fixed_point=None):
    """Plot 2D trajectories with colors based on the argmax of the logits."""
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Loop through each trajectory
    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        color_label = color_labels[i]
        
        # Plot each segment with its corresponding color
        for j in range(len(traj) - 1):
            start = traj[j]
            end = traj[j + 1]
            color = colors[color_label[j]]
            
            # Plot the segment with thicker line
            ax.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=1.5)
            
            # Add an arrow at each timestep (between current and next point) with thicker head
            ax.arrow(
                start[0], start[1],  # Starting point of the arrow
                end[0] - start[0],  # x-direction
                end[1] - start[1],  # y-direction
                color=color,  # Arrow color matches the segment color
                head_width=0.15,  # Increased size of the arrow
                head_length=0.15,  # Increased length of the arrow head
                length_includes_head=True,
                lw=0.8  # Increased line width
            )
    
    # Plot the fixed point as a black X if provided
    if fixed_point is not None:
        ax.scatter(fixed_point[0], fixed_point[1], c='black', marker='x', s=100, linewidth=2, zorder=10)
    
    # Set labels
    ax.set_xlabel('PC1', fontsize=16, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=16, fontweight='bold')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    
    # Create legend patches
    green_patch = Patch(color='darkgreen', label='Logit 0')
    blue_patch = Patch(color='royalblue', label='Logit 1')
    red_patch = Patch(color='darkred', label='Logit 2')
    black_x = plt.Line2D([0], [0], marker='x', color='black', markersize=10, label='Fixed Point')
    
    # Add the legend to the plot
    legend_handles = [green_patch, blue_patch, red_patch]
    if fixed_point is not None:
        legend_handles.append(black_x)
    
    ax.legend(handles=legend_handles, loc='upper left', fontsize=16)
    
    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def create_grid_plot(model_type, trajectories_dict, color_labels_dict, fixed_points_dict, output_folder, with_input=False):
    """Create a 4x3 grid of plots for a specific model type."""
    # Create figure with 4x3 grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Define input sizes and hidden sizes
    input_sizes = [1, 10, 100, 200]
    hidden_sizes = [50, 150, 200]
    
    # Plot each configuration
    for i, hidden_size in enumerate(hidden_sizes):
        for j, input_size in enumerate(input_sizes):
            ax = axes[i, j]
            
            # Check if we have data for this configuration
            if hidden_size in trajectories_dict and input_size in trajectories_dict[hidden_size]:
                trajectories = trajectories_dict[hidden_size][input_size]
                color_labels = color_labels_dict[hidden_size][input_size]
                fixed_point = fixed_points_dict.get(hidden_size, {}).get(input_size, None)
                
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
                
                # Plot the fixed point as a black X if available
                if fixed_point is not None:
                    ax.scatter(fixed_point[0], fixed_point[1], c='black', marker='x', s=80, linewidth=2, zorder=10)
            
            # Set title for each subplot
            ax.set_title(f"Input: {input_size}, Hidden: {hidden_size}", fontsize=10)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set labels for the leftmost plots
            if j == 0:
                ax.set_ylabel('PC2', fontsize=16)
            
            # Set labels for the bottom plots
            if i == 2:
                ax.set_xlabel('PC1', fontsize=16)
    
    # Create legend patches
    green_patch = Patch(color='darkgreen', label='Logit 0')
    blue_patch = Patch(color='royalblue', label='Logit 1')
    red_patch = Patch(color='darkred', label='Logit 2')
    black_x = plt.Line2D([0], [0], marker='x', color='black', markersize=10, label='Fixed Point')
    
    # Add the legend to the figure
    legend_handles = [green_patch, blue_patch, red_patch]
    # Add fixed point to legend only if at least one subplot has a fixed point
    has_fixed_point = any(fixed_points_dict.get(h, {}).get(i, None) is not None 
                          for h in trajectories_dict for i in trajectories_dict[h])
    if has_fixed_point:
        legend_handles.append(black_x)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=16)
    
    # Set overall title
    input_type = "with Input" if with_input else "without Input"
    fig.suptitle(f"{model_type} Trajectories {input_type}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    mode = "traj_in" if with_input else "traj_no_in"
    save_path = os.path.join(output_folder, f"{model_type}_{mode}_grid.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def generate_variance_contour_plots(rnn, pca, plot_title, save_path):
    """Generate contour plots showing where 95% of the variance of the trajectory resides for different input variances."""
    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    device = ih.device
    
    # Define variance levels to test
    variances = [0.1, 1.0, 2.0, 3.0, 4.0]
    steps = 10000
    
    # Generate hidden states for different variances
    h = torch.zeros((len(variances), steps, rnn.hidden_size)).to(device)
    for var_idx, var in enumerate(variances):
        for i in range(1, steps):
            x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
            h[var_idx, i] = torch.relu(x @ ih.T + h[var_idx, i-1] @ hh.T)
    
    # Define colors from the inferno colormap (dark purple/black to yellow/orange)
    colors = plt.cm.inferno(np.linspace(0.9, 0.1, 5))  # Sample 5 colors from the inferno colormap
    handles = []
    labels = []
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot contours for each variance
    for i in range(len(variances)):
        h_pca = pca.transform(h[i].cpu().numpy())
        sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.05], color=colors[i], 
                   label=f'Variance={variances[i]} (95% Contour)', bw_adjust=3.0, linestyles='solid')
        sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.01, 0.2], color=colors[i], 
                   alpha=0.2, bw_adjust=3.0, fill=True, label=None)
        handles.append(plt.Line2D([0], [0], color=colors[i], linestyle='solid'))
        labels.append(f'Variance={variances[i]} (95% Contour)')
    
    # Add legend and labels
    plt.legend(handles, labels, fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=16)
    plt.ylabel('PCA Component 2', fontsize=16)
    plt.title(plot_title)
    
    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def create_variance_contour_grid(model_type, rnn_pca_dict, output_folder):
    """Create a 4x3 grid of variance contour plots for a specific model type."""
    # Create figure with 4x3 grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Define input sizes and hidden sizes
    input_sizes = [1, 10, 100, 200]
    hidden_sizes = [50, 150, 200]
    
    # Define variance levels to test
    variances = [0.1, 1.0, 2.0, 3.0, 4.0]
    steps = 5000  # Reduced for grid plot
    
    # Define colors from the inferno colormap (dark purple/black to yellow/orange)
    colors = plt.cm.inferno(np.linspace(0.9, 0.1, 5))  # Sample 5 colors from the inferno colormap
    
    # Plot each configuration
    for i, hidden_size in enumerate(hidden_sizes):
        for j, input_size in enumerate(input_sizes):
            ax = axes[i, j]
            
            # Check if we have data for this configuration
            if hidden_size in rnn_pca_dict and input_size in rnn_pca_dict[hidden_size]:
                rnn, pca = rnn_pca_dict[hidden_size][input_size]
                
                # Extract weights
                ih = rnn.rnn.weight_ih_l0.data
                hh = rnn.rnn.weight_hh_l0.data
                device = ih.device
                
                # Generate hidden states for different variances
                h = torch.zeros((len(variances), steps, rnn.hidden_size)).to(device)
                for var_idx, var in enumerate(variances):
                    for k in range(1, steps):
                        x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
                        h[var_idx, k] = torch.relu(x @ ih.T + h[var_idx, k-1] @ hh.T)
                
                # Plot contours for each variance
                for var_idx in range(len(variances)):
                    h_pca = pca.transform(h[var_idx].cpu().numpy())
                    # Add filled contours (shaded areas)
                    sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.01, 0.2], color=colors[var_idx], 
                               alpha=0.2, bw_adjust=3.0, fill=True, ax=ax, label=None)
                    # Add contour lines
                    sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.05], color=colors[var_idx], 
                               ax=ax, bw_adjust=3.0, linestyles='solid')
            
            # Set title for each subplot
            ax.set_title(f"Input: {input_size}, Hidden: {hidden_size}", fontsize=10)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set labels for the leftmost plots
            if j == 0:
                ax.set_ylabel('PC2', fontsize=16)
            
            # Set labels for the bottom plots
            if i == 2:
                ax.set_xlabel('PC1', fontsize=16)
    
    # Create legend
    handles = [plt.Line2D([0], [0], color=colors[i], linestyle='solid') for i in range(len(variances))]
    labels = [f'Variance={variances[i]} (95% Contour)' for i in range(len(variances))]
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=16)
    
    # Set overall title
    fig.suptitle(f"{model_type} Variance Contours", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    save_path = os.path.join(output_folder, f"{model_type}_variance_contour_grid.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def create_special_variance_contour_plot(model_types, rnn_pca_dict, output_folder):
    """Create a special 1x4 grid of variance contour plots for the case of RNN with 150 units and input 100."""
    # Create figure with 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    # Define variance levels to test
    variances = [0.1, 1.0, 2.0, 3.0, 4.0]
    steps = 5000  # Reduced for grid plot
    
    # Define colors from the inferno colormap (dark purple/black to yellow/orange)
    colors = plt.cm.inferno(np.linspace(0.9, 0.1, 5))  # Sample 5 colors from the inferno colormap
    
    # Plot each model type
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        
        # Check if we have data for this model type
        if model_type in rnn_pca_dict:
            rnn, pca = rnn_pca_dict[model_type]
            
            # Extract weights
            ih = rnn.rnn.weight_ih_l0.data
            hh = rnn.rnn.weight_hh_l0.data
            device = ih.device
            
            # Generate hidden states for different variances
            h = torch.zeros((len(variances), steps, rnn.hidden_size)).to(device)
            for var_idx, var in enumerate(variances):
                for k in range(1, steps):
                    x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
                    h[var_idx, k] = torch.relu(x @ ih.T + h[var_idx, k-1] @ hh.T)
            
            # Plot contours for each variance
            for var_idx in range(len(variances)):
                h_pca = pca.transform(h[var_idx].cpu().numpy())
                # Add filled contours (shaded areas)
                sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.01, 0.2], color=colors[var_idx], 
                           alpha=0.2, bw_adjust=3.0, fill=True, ax=ax, label=None)
                # Add contour lines
                sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.05], color=colors[var_idx], 
                           ax=ax, bw_adjust=3.0, linestyles='solid')
        
        # Set title for each subplot
        ax.set_title(f"{model_type}", fontsize=12)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set labels with explained variance
        if model_type in rnn_pca_dict:
            _, pca = rnn_pca_dict[model_type]
            # Get explained variance ratios and format as percentages
            var_ratio_1 = pca.explained_variance_ratio_[0] * 100
            var_ratio_2 = pca.explained_variance_ratio_[1] * 100
            ax.set_xlabel(f'PC1 ({var_ratio_1:.1f}%)', fontsize=16)
            ax.set_ylabel(f'PC2 ({var_ratio_2:.1f}%)', fontsize=16)
        else:
            ax.set_xlabel('PC1', fontsize=16)
            ax.set_ylabel('PC2', fontsize=16)
    
    # Create legend
    handles = [plt.Line2D([0], [0], color=colors[i], linestyle='solid') for i in range(len(variances))]
    labels = [f'Variance={variances[i]} (95% Contour)' for i in range(len(variances))]
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=16)
    
    # No overall title as requested
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    save_path = os.path.join(output_folder, f"special_variance_contour_grid.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def create_special_plot(model_types, trajectories_dict, color_labels_dict, fixed_points_dict, output_folder, with_input=False, rnn_pca_dict=None):
    """Create a special 1x4 grid of plots for the case of RNN with 150 units and input 100."""
    # Create figure with 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Plot each model type
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        
        # Check if we have data for this model type
        if model_type in trajectories_dict:
            trajectories = trajectories_dict[model_type]
            color_labels = color_labels_dict[model_type]
            fixed_point = fixed_points_dict.get(model_type, None)
            
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
            
            # Plot the fixed point as a black X if available
            if fixed_point is not None:
                ax.scatter(fixed_point[0], fixed_point[1], c='black', marker='x', s=80, linewidth=2, zorder=10)
        
        # Set title for each subplot
        ax.set_title(f"{model_type}", fontsize=12)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set labels with explained variance
        if rnn_pca_dict and model_type in rnn_pca_dict:
            _, pca = rnn_pca_dict[model_type]
            # Get explained variance ratios and format as percentages
            var_ratio_1 = pca.explained_variance_ratio_[0] * 100
            var_ratio_2 = pca.explained_variance_ratio_[1] * 100
            ax.set_xlabel(f'PC1 ({var_ratio_1:.1f}%)', fontsize=16)
            ax.set_ylabel(f'PC2 ({var_ratio_2:.1f}%)', fontsize=16)
        else:
            ax.set_xlabel('PC1', fontsize=16)
            ax.set_ylabel('PC2', fontsize=16)
    
    # Create legend patches
    green_patch = Patch(color='darkgreen', label='Logit 0')
    blue_patch = Patch(color='royalblue', label='Logit 1')
    red_patch = Patch(color='darkred', label='Logit 2')
    black_x = plt.Line2D([0], [0], marker='x', color='black', markersize=10, label='Fixed Point')
    
    # Add the legend to the figure
    legend_handles = [green_patch, blue_patch, red_patch]
    # Add fixed point to legend only if at least one subplot has a fixed point
    has_fixed_point = any(fixed_points_dict.get(model_type) is not None for model_type in trajectories_dict)
    if has_fixed_point:
        legend_handles.append(black_x)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=16)
    
    # No overall title as requested
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    mode = "traj_in" if with_input else "traj_no_in"
    save_path = os.path.join(output_folder, f"special_{mode}_grid.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def create_combined_grid_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, special_fixed_points_no_in, 
                             special_trajectories_in, special_color_labels_in, special_rnn_pca_dict, output_folder):
    """Create a 3x4 grid plot where rows are different plot types and columns are different models."""
    # Create figure with 3x4 grid (rows are plot types, columns are models)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # Increased width to accommodate legends on the right
    
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Define variance levels for contour plots
    variances = [0.1, 1.0, 2.0, 3.0, 4.0]
    steps = 5000  # Reduced for grid plot
    
    # Define colors from the inferno colormap for contour plots
    contour_colors = plt.cm.inferno(np.linspace(0.9, 0.1, 5))
    
    # Add column titles
    column_titles = ["2 States", "3 States", "4 States", "5 States"]
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Plot each model type (columns) and plot type (rows)
    for j, model_type in enumerate(model_types):
        # First row: traj_no_in plots
        ax_no_in = axes[0, j]
        
        if model_type in special_trajectories_no_in:
            trajectories = special_trajectories_no_in[model_type]
            color_labels = special_color_labels_no_in[model_type]
            fixed_point = special_fixed_points_no_in.get(model_type, None)
            
            # Plot trajectories without input
            for k in range(trajectories.shape[0]):
                traj = trajectories[k]
                color_label = color_labels[k]
                
                for l in range(len(traj) - 1):
                    start = traj[l]
                    end = traj[l + 1]
                    color = colors[color_label[l]]
                    
                    # Plot the segment with thicker line
                    ax_no_in.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=1.0)
                    
                    # Add an arrow with thicker head
                    ax_no_in.arrow(
                        start[0], start[1],
                        end[0] - start[0],
                        end[1] - start[1],
                        color=color,
                        head_width=0.08,
                        head_length=0.08,
                        length_includes_head=True,
                        lw=0.5
                    )
            
            # Plot the fixed point as a black X if available
            if fixed_point is not None:
                ax_no_in.scatter(fixed_point[0], fixed_point[1], c='black', marker='x', s=80, linewidth=2, zorder=10)
        
        # Second row: traj_in plots
        ax_in = axes[1, j]
        
        if model_type in special_trajectories_in:
            trajectories = special_trajectories_in[model_type]
            color_labels = special_color_labels_in[model_type]
            
            # Plot trajectories with input
            for k in range(trajectories.shape[0]):
                traj = trajectories[k]
                color_label = color_labels[k]
                
                for l in range(len(traj) - 1):
                    start = traj[l]
                    end = traj[l + 1]
                    color = colors[color_label[l]]
                    
                    # Plot the segment with thicker line
                    ax_in.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=1.0)
                    
                    # Add an arrow with thicker head
                    ax_in.arrow(
                        start[0], start[1],
                        end[0] - start[0],
                        end[1] - start[1],
                        color=color,
                        head_width=0.08,
                        head_length=0.08,
                        length_includes_head=True,
                        lw=0.5
                    )
        
        # Third row: contour plots
        ax_contour = axes[2, j]
        
        if model_type in special_rnn_pca_dict:
            rnn, pca = special_rnn_pca_dict[model_type]
            
            # Extract weights
            ih = rnn.rnn.weight_ih_l0.data
            hh = rnn.rnn.weight_hh_l0.data
            device = ih.device
            
            # Generate hidden states for different variances
            h = torch.zeros((len(variances), steps, rnn.hidden_size)).to(device)
            for var_idx, var in enumerate(variances):
                for k in range(1, steps):
                    x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
                    h[var_idx, k] = torch.relu(x @ ih.T + h[var_idx, k-1] @ hh.T)
            
            # Plot contours for each variance
            for var_idx in range(len(variances)):
                h_pca = pca.transform(h[var_idx].cpu().numpy())
                # Add filled contours (shaded areas)
                sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.01, 0.2], color=contour_colors[var_idx], 
                           alpha=0.2, bw_adjust=3.0, fill=True, ax=ax_contour, label=None)
                # Add contour lines
                sns.kdeplot(x=h_pca[:, 0], y=h_pca[:, 1], levels=[0.05], color=contour_colors[var_idx], 
                           ax=ax_contour, bw_adjust=3.0, linestyles='solid')
        
        # Remove ticks for all plots
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        
        # Set axis labels with explained variance
        if model_type in special_rnn_pca_dict:
            _, pca = special_rnn_pca_dict[model_type]
            # Get explained variance ratios and format as percentages
            var_ratio_1 = pca.explained_variance_ratio_[0] * 100
            var_ratio_2 = pca.explained_variance_ratio_[1] * 100
            
            # Only add x-axis labels to the third row (contour plots)
            axes[2, j].set_xlabel(f'PC1 ({var_ratio_1:.1f}%)', fontsize=14)
            
            # Add PC2 with variance percentage to y-axis only for the first column
            if j == 0:
                for i in range(3):
                    axes[i, j].set_ylabel(f'PC2 ({var_ratio_2:.1f}%)', fontsize=14)
    
    # Create legend for trajectory plots
    green_patch = Patch(color='darkgreen', label='Logit 0')
    blue_patch = Patch(color='royalblue', label='Logit 1')
    red_patch = Patch(color='darkred', label='Logit 2')
    black_x = plt.Line2D([0], [0], marker='x', color='black', markersize=10, label='Fixed Point')
    
    # Create legend for contour plots
    contour_handles = [plt.Line2D([0], [0], color=contour_colors[i], linestyle='solid') 
                      for i in range(len(variances))]
    contour_labels = [f'Variance={variances[i]}' for i in range(len(variances))]
    
    # Create two separate legends
    # First legend (trajectory plots) - positioned to the right of the first two rows
    first_legend = fig.legend(
        handles=[green_patch, blue_patch, red_patch, black_x],
        labels=['Logit 0', 'Logit 1', 'Logit 2', 'Fixed Point'],
        loc='center right',
        bbox_to_anchor=(1.15, 0.65),  # Position between traj_no_in and traj_in rows
        fontsize=14
    )
    
    # Second legend (contour plots) - positioned to the right of the third row
    second_legend = fig.legend(
        handles=contour_handles,
        labels=contour_labels,
        loc='center right',
        bbox_to_anchor=(1.15, 0.25),  # Position aligned with contour plots
        fontsize=14
    )
    
    # Add both legends to the figure
    fig.add_artist(first_legend)
    fig.add_artist(second_legend)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Leave space for the legends on the right
    
    # Save the figure
    save_path = os.path.join(output_folder, "combined_grid_plot.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Combined grid plot saved to {save_path}")

def run_pipeline(traj_no_in=True, traj_in=True, variance_contour_plots=True):
    """Main function to run the pipeline."""
    # Extract all RNN models
    print("Extracting RNN models...")
    grouped_models = extract_rnn_models()
    
    # Create output folders
    traj_no_in_folder = "traj_no_in"
    traj_in_folder = "traj_in"
    variance_contour_folder = "variance_contours"
    
    if traj_no_in:
        os.makedirs(traj_no_in_folder, exist_ok=True)
    if traj_in:
        os.makedirs(traj_in_folder, exist_ok=True)
    if variance_contour_plots:
        os.makedirs(variance_contour_folder, exist_ok=True)
    
    # Process each model type
    model_types = ["Two", "Three", "Four", "Five"]
    
    # For the special plot (Hidden=150, Input=100)
    special_trajectories_no_in = {}
    special_color_labels_no_in = {}
    special_fixed_points_no_in = {}
    special_trajectories_in = {}
    special_color_labels_in = {}
    special_fixed_points_in = {}
    special_rnn_pca_dict = {}  # For variance contour plots
    
    # For variance contour plots
    rnn_pca_dicts = {}
    for model_type in model_types:
        rnn_pca_dicts[model_type] = {}
    
    for model_type in model_types:
        print(f"\nProcessing {model_type} models...")
        
        # Dictionaries to store trajectories, color labels, and fixed points for grid plots
        trajectories_no_in_dict = {}
        color_labels_no_in_dict = {}
        fixed_points_no_in_dict = {}
        trajectories_in_dict = {}
        color_labels_in_dict = {}
        fixed_points_in_dict = {}
        
        # Process each hidden size
        for hidden_size in [50, 150, 200]:
            if hidden_size not in grouped_models[model_type]:
                print(f"No models found for {model_type} with hidden_size={hidden_size}")
                continue
                
            trajectories_no_in_dict[hidden_size] = {}
            color_labels_no_in_dict[hidden_size] = {}
            fixed_points_no_in_dict[hidden_size] = {}
            trajectories_in_dict[hidden_size] = {}
            color_labels_in_dict[hidden_size] = {}
            fixed_points_in_dict[hidden_size] = {}
            
            # Process each input size
            for input_size in [1, 10, 100, 200]:
                if input_size not in grouped_models[model_type][hidden_size]:
                    print(f"No models found for {model_type} with hidden_size={hidden_size}, input_size={input_size}")
                    continue
                
                # Process each seed (we'll use seed 0 for simplicity)
                seed = 0
                if seed not in grouped_models[model_type][hidden_size][input_size]:
                    print(f"No models found for {model_type} with hidden_size={hidden_size}, input_size={input_size}, seed={seed}")
                    continue
                
                # Get the first model for this configuration
                model_path, config = grouped_models[model_type][hidden_size][input_size][seed][0]
                print(f"\nProcessing model: {os.path.basename(model_path)}")
                
                # Run model and compute PCA
                result = run_model_and_compute_pca(
                    model_path, input_size, hidden_size
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
                
                # Store trajectories, color labels, and fixed points for grid plots
                trajectories_no_in_dict[hidden_size][input_size] = trajectories_no_in
                color_labels_no_in_dict[hidden_size][input_size] = color_labels_no_in
                fixed_points_no_in_dict[hidden_size][input_size] = fixed_point_no_in
                trajectories_in_dict[hidden_size][input_size] = trajectories_in
                color_labels_in_dict[hidden_size][input_size] = color_labels_in
                fixed_points_in_dict[hidden_size][input_size] = None  # No fixed point for with-input case
                
                # Store data for special plot (Hidden=150, Input=100)
                if hidden_size == 150 and input_size == 100:
                    special_trajectories_no_in[model_type] = trajectories_no_in
                    special_color_labels_no_in[model_type] = color_labels_no_in
                    special_fixed_points_no_in[model_type] = fixed_point_no_in
                    special_trajectories_in[model_type] = trajectories_in
                    special_color_labels_in[model_type] = color_labels_in
                    special_fixed_points_in[model_type] = None  # No fixed point for with-input case
                    special_rnn_pca_dict[model_type] = (rnn, pca)  # For variance contour plots
                
                # Store RNN and PCA for variance contour plots
                if variance_contour_plots:
                    if hidden_size not in rnn_pca_dicts[model_type]:
                        rnn_pca_dicts[model_type][hidden_size] = {}
                    rnn_pca_dicts[model_type][hidden_size][input_size] = (rnn, pca)
                
                # Save individual plots (with fixed point only for no-input case)
                if traj_no_in:
                    plot_title_no_in = f"{model_type} (Hidden={hidden_size}, Input={input_size}) - No Input"
                    save_path_no_in = os.path.join(traj_no_in_folder, f"{model_type}_h{hidden_size}_i{input_size}_no_in.png")
                    plot_trajectories_2d(trajectories_no_in, color_labels_no_in, plot_title_no_in, save_path_no_in, fixed_point=fixed_point_no_in)
                
                if traj_in:
                    plot_title_in = f"{model_type} (Hidden={hidden_size}, Input={input_size}) - With Input"
                    save_path_in = os.path.join(traj_in_folder, f"{model_type}_h{hidden_size}_i{input_size}_in.png")
                    plot_trajectories_2d(trajectories_in, color_labels_in, plot_title_in, save_path_in, fixed_point=None)  # No fixed point for with-input case
                
                # Generate and save variance contour plots
                if variance_contour_plots:
                    plot_title_var = f"{model_type} (Hidden={hidden_size}, Input={input_size}) - Variance Contours"
                    save_path_var = os.path.join(variance_contour_folder, f"{model_type}_h{hidden_size}_i{input_size}_var.png")
                    generate_variance_contour_plots(rnn, pca, plot_title_var, save_path_var)
        
        # Create grid plots for this model type with fixed points
        if traj_no_in:
            create_grid_plot(model_type, trajectories_no_in_dict, color_labels_no_in_dict, fixed_points_no_in_dict, traj_no_in_folder, with_input=False)
        if traj_in:
            create_grid_plot(model_type, trajectories_in_dict, color_labels_in_dict, fixed_points_in_dict, traj_in_folder, with_input=True)
        if variance_contour_plots:
            create_variance_contour_grid(model_type, rnn_pca_dicts[model_type], variance_contour_folder)
    
    # Create special plots with fixed points
    if traj_no_in:
        create_special_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, special_fixed_points_no_in, traj_no_in_folder, with_input=False, rnn_pca_dict=special_rnn_pca_dict)
    if traj_in:
        create_special_plot(model_types, special_trajectories_in, special_color_labels_in, special_fixed_points_in, traj_in_folder, with_input=True, rnn_pca_dict=special_rnn_pca_dict)
    if variance_contour_plots:
        create_special_variance_contour_plot(model_types, special_rnn_pca_dict, variance_contour_folder)
    
    # Create the combined grid plot
    create_combined_grid_plot(model_types, special_trajectories_no_in, special_color_labels_no_in, 
                             special_fixed_points_no_in, special_trajectories_in, special_color_labels_in, 
                             special_rnn_pca_dict, ".")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
