import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import os
import sys

# Add parent directory to path to import from scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.mechint import load_model


def load_and_setup_model(model_path, input_size, hidden_size, output_size):
    """
    Load a trained RNN model.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        output_size: Output dimension of the model
        
    Returns:
        rnn: Loaded RNN model object
    """
    print(f"Loading model from: {model_path}")
    rnn = load_model(model_path, input_size=input_size, hidden_size=hidden_size)
    print(f"Model loaded successfully (input_size={input_size}, hidden_size={hidden_size}, output_size={output_size})")
    return rnn


def generate_hidden_states_and_outputs(rnn, time_steps=100000, input_std=1.0):
    """
    Generate hidden states and outputs by running the model with random inputs.
    
    Args:
        rnn: RNN model object
        time_steps: Number of timesteps to generate
        input_std: Standard deviation of input noise
        
    Returns:
        hidden_states: Array of hidden states (time_steps, hidden_size)
        outputs: Array of output labels (time_steps,)
    """
    print(f"Generating {time_steps} timesteps of hidden states and outputs...")
    
    # Initialize
    h = torch.normal(0, 1, size=(rnn.hidden_size,), device=rnn.device)
    hidden_states = []
    outputs = []
    
    # Generate data
    with torch.no_grad():
        for t in range(time_steps):
            # Random input
            x = torch.normal(0, input_std, size=(rnn.input_size,), device=rnn.device)
            
            # Update hidden state
            h = torch.relu(h @ rnn.rnn.weight_hh_l0.data.T + x @ rnn.rnn.weight_ih_l0.data.T)
            
            # Compute output
            logits = h @ rnn.fc.weight.data.T
            probs = F.softmax(logits, dim=0)
            output = torch.argmax(probs).cpu().numpy()
            
            hidden_states.append(h.cpu().numpy())
            outputs.append(output)
    
    hidden_states = np.array(hidden_states)
    outputs = np.array(outputs)
    
    print(f"Generated {len(hidden_states)} hidden states")
    print(f"Output distribution: {np.bincount(outputs)}")
    
    return hidden_states, outputs


def compute_global_pca(hidden_states, n_components=4):
    """
    Compute PCA on all hidden states (first stage).
    
    Args:
        hidden_states: Array of hidden states (time_steps, hidden_size)
        n_components: Number of PCA components to compute
        
    Returns:
        pca: Fitted PCA object
        hidden_states_pca: Transformed hidden states (time_steps, n_components)
    """
    print(f"Computing global PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    hidden_states_pca = pca.fit_transform(hidden_states)
    print(f"Global PCA computed. Explained variance ratios: {pca.explained_variance_ratio_}")
    return pca, hidden_states_pca


def filter_by_output_combination(outputs, hidden_states_pca, output_set, min_length=200):
    """
    Filter trajectories by output combinations and detect continuous sequences.
    
    Args:
        outputs: Array of output labels (time_steps,)
        hidden_states_pca: PCA-transformed hidden states (time_steps, n_components)
        output_set: Set or list of output values to include (e.g., [0, 1])
        min_length: Minimum length of continuous sequences to keep
        
    Returns:
        sequences: List of continuous trajectory segments (each is array of shape (seq_len, 3))
        sequence_outputs: List of output labels for each segment (each is array of shape (seq_len,))
        pca_subspace: PCA object fitted on the filtered subspace
    """
    # Create mask for desired outputs
    mask = np.isin(outputs, output_set)
    
    # Filter data
    filtered_hiddens = hidden_states_pca[mask]
    filtered_outputs = outputs[mask]
    
    print(f"Outputs {output_set}: {len(filtered_hiddens)} points after filtering")
    
    # Apply second-stage PCA (to 3D) on filtered data
    pca_subspace = PCA(n_components=3)
    hiddens_3d = pca_subspace.fit_transform(filtered_hiddens)
    print(f"Subspace PCA explained variance: {pca_subspace.explained_variance_ratio_}")
    
    # Detect continuous sequences
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    long_sequences = [(s, e) for s, e in zip(starts, ends) if (e - s) > min_length]
    
    print(f"Found {len(long_sequences)} continuous sequences (min_length={min_length})")
    
    if not long_sequences:
        return [], [], pca_subspace
    
    # Extract sequences
    sequences = []
    sequence_outputs = []
    filtered_indices = np.where(mask)[0]
    
    for seq_start, seq_end in long_sequences:
        # Map back to filtered indices
        start_idx = np.where(filtered_indices == seq_start)[0][0]
        end_idx = np.where(filtered_indices == seq_end - 1)[0][0] + 1
        
        sequences.append(hiddens_3d[start_idx:end_idx])
        sequence_outputs.append(filtered_outputs[start_idx:end_idx])
    
    return sequences, sequence_outputs, pca_subspace


def plot_subspace_trajectories(sequences, sequence_outputs, pc_dims=(0, 1), 
                               num_trajectories=None, title="", save_path=None,
                               output_colors=None, show_arrows=True, pca_subspace=None):
    """
    Plot trajectories in a subspace defined by output combinations.
    
    Args:
        sequences: List of trajectory segments (each is array of shape (seq_len, 3))
        sequence_outputs: List of output labels for each segment
        pc_dims: Tuple of two PC indices to plot (e.g., (0, 1))
        num_trajectories: Number of longest trajectories to plot (None = all)
        title: Plot title
        save_path: Path to save figure (None = display only)
        output_colors: Dictionary mapping output labels to colors (optional)
        show_arrows: Whether to show directional arrows on trajectories
        pca_subspace: Optional PCA object to show explained variance in labels
    """
    if len(sequences) == 0:
        print("No sequences to plot!")
        return None, None
    
    # Default colors
    if output_colors is None:
        output_colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Select trajectories (prioritize longest ones)
    if num_trajectories is None:
        num_trajectories = len(sequences)
    else:
        num_trajectories = min(num_trajectories, len(sequences))
    
    seq_lengths = [len(seq) for seq in sequences]
    sorted_indices = np.argsort(seq_lengths)[::-1]
    selected_indices = sorted_indices[:num_trajectories]
    
    print(f"Plotting {num_trajectories} trajectories in PC{pc_dims[0]+1} vs PC{pc_dims[1]+1}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each selected trajectory
    for plot_idx, seq_idx in enumerate(selected_indices):
        hiddens = sequences[seq_idx]
        outputs = sequence_outputs[seq_idx]
        
        # Alpha decreases for shorter trajectories
        alpha = max(0.8 - (plot_idx * 0.1), 0.3)
        
        # Plot trajectory segments and arrows
        for i in range(len(hiddens) - 1):
            start = hiddens[i, list(pc_dims)]
            end = hiddens[i + 1, list(pc_dims)]
            
            color = output_colors.get(outputs[i], 'gray')
            
            # Plot the line segment
            #ax.plot([start[0], end[0]], [start[1], end[1]], c=color, lw=2.5, alpha=alpha)
            
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
                    lw=2.5,
                    alpha=alpha
                )
    
    # Set labels with explained variance if PCA provided
    pc1_idx, pc2_idx = pc_dims
    if pca_subspace is not None:
        var_ratio_1 = pca_subspace.explained_variance_ratio_[pc1_idx] * 100
        var_ratio_2 = pca_subspace.explained_variance_ratio_[pc2_idx] * 100
        ax.set_xlabel(f'PC{pc1_idx+1} ({var_ratio_1:.1f}%)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC{pc2_idx+1} ({var_ratio_2:.1f}%)', fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel(f'PC{pc1_idx+1}', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'PC{pc2_idx+1}', fontsize=16, fontweight='bold')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Create legend for output classes present in the data
    unique_outputs = sorted(set(np.concatenate(sequence_outputs)))
    legend_handles = [Patch(color=output_colors.get(out, 'gray'), label=f'Class {out}') 
                     for out in unique_outputs]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=14)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, format='svg' if save_path.endswith('.svg') else None,
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    return fig, ax


def run_pipeline(model_path, input_size, hidden_size, output_size,
                output_combinations=None, pc_dims=(0, 1),
                time_steps=100000, min_sequence_length=200,
                n_components_global=4, num_trajectories=2,
                save_dir=None, random_seed=None, output_colors=None):
    """
    Main pipeline to generate and plot output-based subspace trajectories.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        output_size: Output dimension of the model
        output_combinations: List of output combinations to analyze (e.g., [[0,1], [1,2], [0,2]])
                           If None, uses all pairwise combinations
        pc_dims: Tuple of two PC indices to plot
        time_steps: Number of timesteps to generate
        min_sequence_length: Minimum length of continuous sequences
        n_components_global: Number of components for global PCA
        num_trajectories: Number of longest trajectories to plot per subspace
        save_dir: Directory to save plots (None = display only)
        random_seed: Random seed for reproducibility
        output_colors: Dictionary mapping output labels to colors
        
    Returns:
        results: Dictionary containing all computed data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Default output combinations (all pairwise)
    if output_combinations is None:
        output_combinations = [[i, j] for i in range(output_size) 
                              for j in range(i+1, output_size)]
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Starting output subspace trajectory pipeline")
    print("=" * 60)
    
    # Step 1: Load model
    rnn = load_and_setup_model(model_path, input_size, hidden_size, output_size)
    
    # Step 2: Generate hidden states and outputs
    hidden_states, outputs = generate_hidden_states_and_outputs(rnn, time_steps=time_steps)
    
    # Step 3: Compute global PCA
    pca_global, hidden_states_pca = compute_global_pca(hidden_states, 
                                                       n_components=n_components_global)
    
    # Step 4: Process each output combination
    results = {
        'pca_global': pca_global,
        'hidden_states': hidden_states,
        'hidden_states_pca': hidden_states_pca,
        'outputs': outputs,
        'subspaces': {}
    }
    
    for output_set in output_combinations:
        print("\n" + "=" * 60)
        print(f"Processing subspace for outputs {output_set}")
        print("=" * 60)
        
        # Filter and compute subspace PCA
        sequences, sequence_outputs, pca_subspace = filter_by_output_combination(
            outputs, hidden_states_pca, output_set, min_length=min_sequence_length
        )
        
        # Store results
        subspace_name = ''.join(map(str, output_set))
        results['subspaces'][subspace_name] = {
            'output_set': output_set,
            'sequences': sequences,
            'sequence_outputs': sequence_outputs,
            'pca_subspace': pca_subspace
        }
        
        # Plot if sequences exist
        if len(sequences) > 0:
            title = f"Outputs {' & '.join(map(str, output_set))} Subspace"
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f'subspace_{subspace_name}_PC{pc_dims[0]+1}_PC{pc_dims[1]+1}.svg')
            
            plot_subspace_trajectories(
                sequences, sequence_outputs,
                pc_dims=pc_dims,
                num_trajectories=num_trajectories,
                title=title,
                save_path=save_path,
                output_colors=output_colors,
                show_arrows=True,
                pca_subspace=pca_subspace
            )
        else:
            print(f"No sequences found for outputs {output_set}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    model_path = "TrainedModels/Cyclic/Seed 0/models/4HMM_3Outputs_linear_30kData_0.001lr_5.5Loss.pth"
    input_size = 100
    hidden_size = 150
    output_size = 3
    
    # Define custom colors (optional)
    output_colors = {0: 'darkred', 1: 'royalblue', 2: 'darkgreen'}
    
    # Run pipeline
    results = run_pipeline(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        output_combinations=[[0, 1], [1, 2], [0, 2]],
        pc_dims=(0, 2),  # Plot PC1 vs PC3
        time_steps=100000,
        min_sequence_length=200,
        n_components_global=4,
        num_trajectories=1,
        save_dir='plots/subspaces/subspaces_cyclic',
        random_seed=3,
        output_colors=output_colors
    )