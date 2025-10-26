import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def compute_subspace_pca(outputs, hidden_states_pca, output_set):
    """
    Compute PCA on filtered subspace for specific output combination.
    
    Args:
        outputs: Array of output labels (time_steps,)
        hidden_states_pca: PCA-transformed hidden states (time_steps, n_components)
        output_set: Set or list of output values to include (e.g., [0, 1])
        
    Returns:
        pca_subspace: PCA object fitted on the filtered subspace (3D)
        mask: Boolean mask for filtering
    """
    # Create mask for desired outputs
    mask = np.isin(outputs, output_set)
    
    # Filter data
    filtered_hiddens = hidden_states_pca[mask]
    
    print(f"Outputs {output_set}: {len(filtered_hiddens)} points after filtering")
    
    # Apply second-stage PCA (to 3D) on filtered data
    pca_subspace = PCA(n_components=3)
    pca_subspace.fit(filtered_hiddens)
    print(f"Subspace PCA explained variance: {pca_subspace.explained_variance_ratio_}")
    
    return pca_subspace, mask


def generate_variance_trajectories_for_subspace(rnn, pca_global, pca_subspace, 
                                                output_set, variances, steps):
    """
    Generate hidden states for different variances and filter by output subspace.
    
    Args:
        rnn: RNN model object
        pca_global: Global PCA object (first stage)
        pca_subspace: Subspace PCA object (second stage)
        output_set: Set of output values to filter for
        variances: List of variance levels to test
        steps: Number of timesteps to generate for each variance
        
    Returns:
        filtered_data: List of arrays, one per variance level (variable length, 3)
    """
    print(f"Generating variance trajectories for outputs {output_set}...")
    
    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc_weight = rnn.fc.weight.data
    device = ih.device
    
    filtered_data = []
    
    for var_idx, var in enumerate(variances):
        print(f"  Variance={var}...")
        
        # Generate trajectories
        h = torch.zeros(rnn.hidden_size).to(device)
        hidden_states = []
        outputs = []
        
        for i in range(steps):
            x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
            h = torch.relu(x @ ih.T + h @ hh.T)
            
            # Compute output
            logits = h @ fc_weight.T
            probs = F.softmax(logits, dim=0)
            output = torch.argmax(probs).cpu().numpy()
            
            hidden_states.append(h.cpu().numpy())
            outputs.append(output)
        
        hidden_states = np.array(hidden_states)
        outputs = np.array(outputs)
        
        # Filter by output set
        mask = np.isin(outputs, output_set)
        filtered_hiddens = hidden_states[mask]
        
        print(f"    Filtered: {len(filtered_hiddens)}/{len(hidden_states)} points")
        
        if len(filtered_hiddens) > 0:
            # Apply two-stage PCA
            hiddens_global_pca = pca_global.transform(filtered_hiddens)
            hiddens_subspace_pca = pca_subspace.transform(hiddens_global_pca)
            filtered_data.append(hiddens_subspace_pca)
        else:
            filtered_data.append(np.array([]).reshape(0, 3))
    
    return filtered_data


def plot_subspace_variance_contours(filtered_data, variances, pca_subspace,
                                    output_set, pc_axes=(0, 1),
                                    save_path=None, bw_adjust=3.0):
    """
    Plot variance contour plots for a specific output subspace.
    
    Args:
        filtered_data: List of arrays (one per variance), each shape (n_points, 3)
        variances: List of variance levels
        pca_subspace: Subspace PCA object for variance labels
        output_set: Output combination being plotted
        pc_axes: Tuple of two PC indices to plot
        save_path: Path to save figure
        bw_adjust: Bandwidth adjustment for KDE plots
    """
    pc1_idx, pc2_idx = pc_axes
    
    # Define colors from the inferno colormap
    colors = plt.cm.inferno(np.linspace(0.9, 0.1, len(variances)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    handles = []
    labels = []
    
    # Plot contours for each variance
    for i, (var, data) in enumerate(zip(variances, filtered_data)):
        if len(data) < 10:  # Skip if too few points
            print(f"Skipping variance={var} (only {len(data)} points)")
            continue
        
        # Extract the specified PC axes
        x_data = data[:, pc1_idx]
        y_data = data[:, pc2_idx]
        
        try:
            # Plot 95% contour (outer boundary)
            sns.kdeplot(x=x_data, y=y_data, levels=[0.05], color=colors[i],
                       bw_adjust=bw_adjust, linestyles='solid', ax=ax)
            
            # Plot filled contours for additional context
            sns.kdeplot(x=x_data, y=y_data, levels=[0.01, 0.2], color=colors[i],
                       alpha=0.2, bw_adjust=bw_adjust, fill=True, ax=ax)
            
            # Create legend handle
            handles.append(plt.Line2D([0], [0], color=colors[i], linestyle='solid', linewidth=2))
            labels.append(f'Variance={var} (95% Contour)')
        except Exception as e:
            print(f"Could not plot contour for variance={var}: {e}")
    
    # Set labels with explained variance
    var_ratio_1 = pca_subspace.explained_variance_ratio_[pc1_idx] * 100
    var_ratio_2 = pca_subspace.explained_variance_ratio_[pc2_idx] * 100
    ax.set_xlabel(f'PC{pc1_idx+1} ({var_ratio_1:.1f}%)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'PC{pc2_idx+1} ({var_ratio_2:.1f}%)', fontsize=16, fontweight='bold')
    
    # Set title
    title = f"Outputs {' & '.join(map(str, output_set))} Subspace (95% Confidence)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    if handles:
        ax.legend(handles, labels, fontsize=12, loc='best')
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, format='svg' if save_path.endswith('.svg') else None,
                   dpi=600, bbox_inches='tight')
        print(f"Subspace variance contour plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_subspace_variance_pipeline(model_path, input_size, hidden_size, output_size,
                                   output_combinations=None,
                                   variances=[0.1, 1.0, 2.0, 3.0, 4.0],
                                   steps=10000, time_steps_pca=100000,
                                   n_components_global=4, pc_axes=(0, 1),
                                   save_dir=None, bw_adjust=3.0, random_seed=None):
    """
    Main pipeline to generate variance contour plots for output subspaces.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        output_size: Output dimension of the model
        output_combinations: List of output combinations to analyze
        variances: List of variance levels to test
        steps: Number of timesteps to generate for each variance
        time_steps_pca: Number of timesteps to generate for PCA computation
        n_components_global: Number of components for global PCA
        pc_axes: Tuple of two PC indices to plot
        save_dir: Directory to save plots
        bw_adjust: Bandwidth adjustment for KDE plots
        random_seed: Optional random seed for reproducibility
        
    Returns:
        results: Dictionary containing computed data
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
    
    # Validate pc_axes
    if len(pc_axes) != 2:
        raise ValueError("pc_axes must be a tuple of two indices")
    
    print("=" * 60)
    print("Starting subspace variance contour pipeline")
    print("=" * 60)
    
    # Step 1: Load model
    rnn = load_and_setup_model(model_path, input_size, hidden_size, output_size)
    
    # Step 2: Generate hidden states and outputs for PCA computation
    hidden_states, outputs = generate_hidden_states_and_outputs(
        rnn, time_steps=time_steps_pca, input_std=1.0
    )
    
    # Step 3: Compute global PCA
    pca_global, hidden_states_pca = compute_global_pca(
        hidden_states, n_components=n_components_global
    )
    
    # Step 4: Process each output combination
    results = {
        'pca_global': pca_global,
        'variances': variances,
        'subspaces': {}
    }
    
    for output_set in output_combinations:
        print("\n" + "=" * 60)
        print(f"Processing subspace for outputs {output_set}")
        print("=" * 60)
        
        # Compute subspace PCA
        pca_subspace, mask = compute_subspace_pca(outputs, hidden_states_pca, output_set)
        
        # Generate variance trajectories for this subspace
        filtered_data = generate_variance_trajectories_for_subspace(
            rnn, pca_global, pca_subspace, output_set, variances, steps
        )
        
        # Store results
        subspace_name = ''.join(map(str, output_set))
        results['subspaces'][subspace_name] = {
            'output_set': output_set,
            'pca_subspace': pca_subspace,
            'filtered_data': filtered_data
        }
        
        # Plot variance contours
        save_path = None
        if save_dir:
            save_path = os.path.join(
                save_dir,
                f'subspace_variance_{subspace_name}_PC{pc_axes[0]+1}_PC{pc_axes[1]+1}.svg'
            )
        
        plot_subspace_variance_contours(
            filtered_data, variances, pca_subspace, output_set,
            pc_axes=pc_axes, save_path=save_path, bw_adjust=bw_adjust
        )
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    model_path = "TrainedModels/Fully_Connected/Seed 0/models/3HMM_3Outputs_triangular_30kData_0.001lr_1.9Loss.pth"
    input_size = 100
    hidden_size = 150
    output_size = 3
    
    # Run pipeline
    results = run_subspace_variance_pipeline(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        output_combinations=[[0, 1], [1, 2], [0, 2]],
        variances=[0.1, 1.0, 2.0, 3.0, 4.0],
        steps=10000,
        time_steps_pca=100000,
        n_components_global=4,
        pc_axes=(0, 2),
        save_dir='plots/variance/fully',
        bw_adjust=3.0,
        random_seed=42
    )