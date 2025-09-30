import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import torch
import os
import sys

# Add parent directory to path to import from scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.mechint import load_model


def load_and_setup_model(model_path, input_size, hidden_size):
    """
    Load a trained RNN model.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        
    Returns:
        rnn: Loaded RNN model object
    """
    print(f"Loading model from: {model_path}")
    rnn = load_model(model_path, input_size=input_size, hidden_size=hidden_size)
    print(f"Model loaded successfully (input_size={input_size}, hidden_size={hidden_size})")
    return rnn


def compute_pca_from_data(rnn, time_steps=30000, n_components=10):
    """
    Generate data with input and compute PCA on hidden states.
    
    Args:
        rnn: RNN model object
        time_steps: Number of timesteps to generate
        n_components: Number of PCA components to compute
        
    Returns:
        pca: Fitted PCA object
        hidden_states: Hidden states used for PCA (for IC sampling)
    """
    print(f"Generating {time_steps} timesteps with input for PCA computation...")
    rnn_data = rnn.gen_seq(time_steps=time_steps, dynamics_mode="full")
    hidden_states = rnn_data["h"]
    
    # Check for NaN values
    if np.isnan(hidden_states).any():
        raise ValueError("NaN values detected in hidden states. Cannot compute PCA.")
    
    # Compute PCA
    print(f"Computing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(hidden_states)
    
    print(f"PCA computed. Explained variance ratios: {pca.explained_variance_ratio_[:5]}")
    
    return pca, hidden_states


def generate_trajectories(rnn, initial_states, trajectory_length=500, with_input=False, input_std=1.0):
    """
    Generate trajectories with or without input from given initial conditions.
    
    Args:
        rnn: RNN model object
        initial_states: Array of initial hidden states (num_samples, hidden_size)
        trajectory_length: Length of each trajectory
        with_input: If True, generate trajectories with random input; if False, recurrence only
        input_std: Standard deviation of the Gaussian input (only used if with_input=True)
        
    Returns:
        trajectories: Raw trajectories in hidden state space (num_samples, trajectory_length, hidden_size)
        logit_labels: Argmax of logits at each timestep (num_samples, trajectory_length)
        sampled_outputs: Sampled outputs at each timestep (num_samples, trajectory_length)
        fixed_point_hidden: Fixed point in hidden state space (only for without input case, None otherwise)
    """
    num_samples = initial_states.shape[0]
    input_mode = "with input" if with_input else "without input"
    print(f"Generating {num_samples} trajectories {input_mode} (length={trajectory_length})...")
    
    # Initialize arrays
    trajectories = np.zeros((num_samples, trajectory_length, rnn.hidden_size))
    logit_labels = np.zeros((num_samples, trajectory_length), dtype=int)
    sampled_outputs = np.zeros((num_samples, trajectory_length), dtype=int)
    
    # Array to store final hidden states for fixed point calculation (only for without input)
    if not with_input:
        final_hidden_states = np.zeros((num_samples, rnn.hidden_size))
    
    # Generate trajectories
    for i in range(num_samples):
        h_current = torch.tensor(initial_states[i], dtype=torch.float32).to(rnn.device)
        
        for t in range(trajectory_length):
            if with_input:
                # Full dynamics: with input
                x = torch.normal(mean=0, std=input_std, size=(rnn.input_size,)).float().to(rnn.device)
                h_current = torch.relu(x @ rnn.rnn.weight_ih_l0.data.T + h_current @ rnn.rnn.weight_hh_l0.data.T)
            else:
                # Recurrence only (no input)
                h_current = torch.relu(h_current @ rnn.rnn.weight_hh_l0.data.T)
            
            trajectories[i, t] = h_current.cpu().detach().numpy()
            
            # Compute logits for argmax coloring
            logits = h_current @ rnn.fc.weight.data.T
            logit_labels[i, t] = torch.argmax(logits).item()
            
            # Sample output for sampling-based coloring
            probs = torch.softmax(logits, dim=0)
            sampled_outputs[i, t] = torch.multinomial(probs, num_samples=1).item()
        
        # Store the final hidden state for fixed point calculation (only for without input)
        if not with_input:
            final_hidden_states[i] = trajectories[i, -1]
    
    # Calculate the fixed point by averaging the final hidden states (only for without input)
    if not with_input:
        fixed_point_hidden = np.mean(final_hidden_states, axis=0)
        print(f"Fixed point calculated (averaged from {num_samples} final states)")
    else:
        fixed_point_hidden = None
    
    print("Trajectory generation complete.")
    return trajectories, logit_labels, sampled_outputs, fixed_point_hidden


def project_trajectories(trajectories, pca, pc_axes=(0, 1)):
    """
    Project trajectories onto specified PC axes.
    
    Args:
        trajectories: Raw trajectories in hidden state space (num_samples, trajectory_length, hidden_size)
        pca: Fitted PCA object
        pc_axes: Tuple of two PC indices to project onto (e.g., (0, 1) for PC1 vs PC2)
        
    Returns:
        projected_trajectories: Projected trajectories (num_samples, trajectory_length, 2)
    """
    num_samples, trajectory_length, hidden_size = trajectories.shape
    pc1_idx, pc2_idx = pc_axes
    
    print(f"Projecting trajectories onto PC{pc1_idx+1} and PC{pc2_idx+1}...")
    
    # Reshape for PCA transform
    trajectories_reshaped = trajectories.reshape(-1, hidden_size)
    
    # Project onto all PCs first
    projected_all = pca.transform(trajectories_reshaped)
    
    # Select desired PC axes
    projected_selected = projected_all[:, [pc1_idx, pc2_idx]]
    
    # Reshape back
    projected_trajectories = projected_selected.reshape(num_samples, trajectory_length, 2)
    
    print("Projection complete.")
    return projected_trajectories


def plot_trajectories(projected_trajectories, color_data, pc_axes, color_mode, 
                      save_path, title=None, pca=None, show_arrows=True, fixed_point_pca=None):
    """
    Plot projected trajectories with appropriate coloring.
    
    Args:
        projected_trajectories: Projected trajectories (num_samples, trajectory_length, 2)
        color_data: Color information (num_samples, trajectory_length)
        pc_axes: Tuple of PC indices used for projection
        color_mode: 'logits' or 'sampled'
        save_path: Path to save the figure
        title: Optional plot title
        pca: Optional PCA object to show explained variance
        show_arrows: Whether to show directional arrows on trajectories
        fixed_point_pca: Optional fixed point in PCA space (2D coordinates)
    """
    # Define colors for each output class
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    num_classes = len(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    num_samples = projected_trajectories.shape[0]
    
    # Plot each trajectory
    for i in range(num_samples):
        traj = projected_trajectories[i]
        color_labels = color_data[i]
        
        # Plot each segment with its corresponding color
        for j in range(len(traj) - 1):
            start = traj[j]
            end = traj[j + 1]
            
            # Get color based on class (handle any number of classes)
            color_idx = color_labels[j]
            if color_idx < num_classes:
                color = colors[color_idx]
            else:
                color = 'gray'  # Fallback for unexpected classes
            
            # Plot the segment
            ax.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=2.5, alpha=0.7)
            
            # Add arrows if requested
            if show_arrows:
                ax.arrow(
                    start[0], start[1],
                    end[0] - start[0],
                    end[1] - start[1],
                    color=color,
                    head_width=0.15,
                    head_length=0.15,
                    length_includes_head=True,
                    lw=0.8,
                    alpha=0.7
                )
    
    # Plot the fixed point as a black X if provided
    if fixed_point_pca is not None:
        ax.scatter(fixed_point_pca[0], fixed_point_pca[1], c='black', marker='x', 
                  s=200, linewidth=3, zorder=10, label='Fixed Point')
    
    # Set labels with explained variance if PCA provided
    pc1_idx, pc2_idx = pc_axes
    if pca is not None:
        var_ratio_1 = pca.explained_variance_ratio_[pc1_idx] * 100
        var_ratio_2 = pca.explained_variance_ratio_[pc2_idx] * 100
        ax.set_xlabel(f'PC{pc1_idx+1} ({var_ratio_1:.1f}%)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC{pc2_idx+1} ({var_ratio_2:.1f}%)', fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel(f'PC{pc1_idx+1}', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC{pc2_idx+1}', fontsize=16, fontweight='bold')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        color_label = 'Logit Argmax' if color_mode == 'logits' else 'Sampled Output'
        ax.set_title(f'Trajectories (Colored by {color_label})', fontsize=14, fontweight='bold')
    
    # Create legend
    legend_handles = []
    legend_labels = []
    for class_idx in range(num_classes):
        patch = Patch(color=colors[class_idx], label=f'Class {class_idx}')
        legend_handles.append(patch)
        legend_labels.append(f'Class {class_idx}')
    
    # Add fixed point to legend if it exists
    if fixed_point_pca is not None:
        fixed_point_marker = plt.Line2D([0], [0], marker='x', color='black', 
                                       markersize=12, linestyle='None', label='Fixed Point')
        legend_handles.append(fixed_point_marker)
        legend_labels.append('Fixed Point')
    
    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=14)
    
    # Save figure (determine format from extension)
    plt.savefig(save_path, format='svg' if save_path.endswith('.svg') else None, 
                dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def run_pipeline(model_path, input_size, hidden_size, 
                pc_axes=(0, 1), color_mode='logits',
                num_samples=100, trajectory_length=500,
                time_steps_pca=30000, n_components=10,
                save_path='trajectory_plot.png', title=None,
                show_arrows=True, random_seed=None,
                with_input=False, input_std=1.0):
    """
    Main pipeline to generate and plot trajectories with or without input.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        pc_axes: Tuple of two PC indices to plot (e.g., (0, 1) for PC1 vs PC2)
        color_mode: 'logits' for argmax of logits, 'sampled' for sampled outputs
        num_samples: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        time_steps_pca: Number of timesteps to generate for PCA computation
        n_components: Number of PCA components to compute
        save_path: Path to save the output figure
        title: Optional plot title
        show_arrows: Whether to show directional arrows
        random_seed: Optional random seed for reproducibility
        with_input: If True, generate trajectories with random input; if False, recurrence only
        input_std: Standard deviation of the Gaussian input (only used if with_input=True)
        
    Returns:
        results: Dictionary containing:
            - 'pca': Fitted PCA object
            - 'trajectories': Raw trajectories in hidden state space
            - 'projected_trajectories': Projected trajectories
            - 'logit_labels': Argmax of logits
            - 'sampled_outputs': Sampled outputs
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Validate color_mode
    if color_mode not in ['logits', 'sampled']:
        raise ValueError("color_mode must be 'logits' or 'sampled'")
    
    # Validate pc_axes
    if len(pc_axes) != 2:
        raise ValueError("pc_axes must be a tuple of two indices")
    
    print("=" * 60)
    print("Starting trajectory generation pipeline")
    print("=" * 60)
    
    # Step 1: Load model
    rnn = load_and_setup_model(model_path, input_size, hidden_size)
    
    # Step 2: Compute PCA from data with input
    pca, hidden_states = compute_pca_from_data(rnn, time_steps=time_steps_pca, 
                                               n_components=n_components)
    
    # Step 3: Sample initial conditions
    print(f"Sampling {num_samples} initial conditions...")
    indices = np.random.choice(len(hidden_states), num_samples, replace=False)
    initial_states = hidden_states[indices]
    
    # Step 4: Generate trajectories (with or without input)
    trajectories, logit_labels, sampled_outputs, fixed_point_hidden = generate_trajectories(
        rnn, initial_states, trajectory_length=trajectory_length,
        with_input=with_input, input_std=input_std
    )
    
    # Step 5: Project trajectories
    projected_trajectories = project_trajectories(trajectories, pca, pc_axes=pc_axes)
    
    # Step 6: Project fixed point if it exists (only for without input case)
    if fixed_point_hidden is not None:
        fixed_point_pca = pca.transform(fixed_point_hidden.reshape(1, -1))[0, list(pc_axes)]
        print(f"Fixed point in PCA space: {fixed_point_pca}")
    else:
        fixed_point_pca = None
    
    # Step 7: Plot trajectories
    color_data = logit_labels if color_mode == 'logits' else sampled_outputs
    plot_trajectories(projected_trajectories, color_data, pc_axes, color_mode,
                     save_path, title=title, pca=pca, show_arrows=show_arrows,
                     fixed_point_pca=fixed_point_pca)
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Return results for further analysis if needed
    results = {
        'pca': pca,
        'trajectories': trajectories,
        'projected_trajectories': projected_trajectories,
        'logit_labels': logit_labels,
        'sampled_outputs': sampled_outputs,
        'fixed_point_hidden': fixed_point_hidden,
        'fixed_point_pca': fixed_point_pca
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    model_path = "/home/elia/Documents/rnnrep/TrainedModels/Fully_Connected/Seed 0/models/3HMM_3Outputs_triangular_30kData_0.001lr_1.9Loss.pth"
    input_size = 100
    hidden_size = 150
    
    # Run with default parameters (PC1 vs PC2, colored by logits)
    results = run_pipeline(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        pc_axes=(0, 1),  # PC1 vs PC2
        color_mode='logits',  # or 'sampled'
        num_samples=1,
        trajectory_length=700,
        save_path='plots/trajectories/trajectories_cyclic_prova.svg',  # SVG format
        title='RNN Trajectories',
        random_seed=42,
        with_input=True 
    )
