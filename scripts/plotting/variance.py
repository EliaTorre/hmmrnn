import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import sys
from sklearn.decomposition import PCA

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
    
    return pca


def generate_variance_contour_plots(rnn, pca, plot_title, save_path, 
                                   variances=[0.1, 1.0, 2.0, 3.0, 4.0],
                                   steps=10000, pc_axes=(0, 1),
                                   bw_adjust=3.0):
    """
    Generate contour plots showing where 95% of the variance of the trajectory 
    resides for different input variances.
    
    Args:
        rnn: RNN model object
        pca: Fitted PCA object
        plot_title: Title for the plot
        save_path: Path to save the figure
        variances: List of variance levels to test
        steps: Number of timesteps to generate for each variance
        pc_axes: Tuple of two PC indices to plot (e.g., (0, 1) for PC1 vs PC2)
        bw_adjust: Bandwidth adjustment for KDE plots
    """
    print(f"Generating variance contour plots for variances: {variances}")
    print(f"Generating {steps} timesteps for each variance level...")
    
    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    device = ih.device
    
    pc1_idx, pc2_idx = pc_axes
    
    # Generate hidden states for different variances
    h = torch.zeros((len(variances), steps, rnn.hidden_size)).to(device)
    for var_idx, var in enumerate(variances):
        print(f"Generating trajectories for variance={var}...")
        for i in range(1, steps):
            x = torch.normal(mean=0, std=var, size=(rnn.input_size,)).float().to(device)
            h[var_idx, i] = torch.relu(x @ ih.T + h[var_idx, i-1] @ hh.T)
    
    print("Projecting hidden states onto PCA space...")
    
    # Define colors from the inferno colormap (dark purple/black to yellow/orange)
    colors = plt.cm.inferno(np.linspace(0.9, 0.1, len(variances)))
    handles = []
    labels = []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours for each variance
    for i in range(len(variances)):
        h_pca = pca.transform(h[i].cpu().numpy())
        
        # Extract the specified PC axes
        x_data = h_pca[:, pc1_idx]
        y_data = h_pca[:, pc2_idx]
        
        # Plot 95% contour (outer boundary)
        sns.kdeplot(x=x_data, y=y_data, levels=[0.05], color=colors[i], 
                   label=f'Variance={variances[i]} (95% Contour)', 
                   bw_adjust=bw_adjust, linestyles='solid', ax=ax)
        
        # Plot filled contours for additional context
        sns.kdeplot(x=x_data, y=y_data, levels=[0.01, 0.2], color=colors[i], 
                   alpha=0.2, bw_adjust=bw_adjust, fill=True, label=None, ax=ax)
        
        # Create legend handle
        handles.append(plt.Line2D([0], [0], color=colors[i], linestyle='solid', linewidth=2))
        labels.append(f'Variance={variances[i]} (95% Contour)')
    
    # Set labels with explained variance
    var_ratio_1 = pca.explained_variance_ratio_[pc1_idx] * 100
    var_ratio_2 = pca.explained_variance_ratio_[pc2_idx] * 100
    ax.set_xlabel(f'PC{pc1_idx+1} ({var_ratio_1:.1f}%)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'PC{pc2_idx+1} ({var_ratio_2:.1f}%)', fontsize=16, fontweight='bold')
    
    # Set title
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(handles, labels, fontsize=12, loc='best')
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure (determine format from extension)
    plt.savefig(save_path, format='svg' if save_path.endswith('.svg') else None,
                dpi=600, bbox_inches='tight')
    print(f"Variance contour plot saved to: {save_path}")
    plt.close()


def run_variance_pipeline(model_path, input_size, hidden_size,
                         variances=[0.1, 1.0, 2.0, 3.0, 4.0],
                         steps=10000, time_steps_pca=30000,
                         n_components=10, pc_axes=(0, 1),
                         save_path='variance_contour_plot.svg',
                         title='Variance Contour Plot',
                         bw_adjust=3.0, random_seed=None):
    """
    Main pipeline to generate variance contour plots.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension of the model
        hidden_size: Hidden state dimension of the model
        variances: List of variance levels to test
        steps: Number of timesteps to generate for each variance
        time_steps_pca: Number of timesteps to generate for PCA computation
        n_components: Number of PCA components to compute
        pc_axes: Tuple of two PC indices to plot (e.g., (0, 1) for PC1 vs PC2)
        save_path: Path to save the output figure
        title: Plot title
        bw_adjust: Bandwidth adjustment for KDE plots
        random_seed: Optional random seed for reproducibility
        
    Returns:
        results: Dictionary containing:
            - 'pca': Fitted PCA object
            - 'variances': List of variances used
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Validate pc_axes
    if len(pc_axes) != 2:
        raise ValueError("pc_axes must be a tuple of two indices")
    
    print("=" * 60)
    print("Starting variance contour plot generation pipeline")
    print("=" * 60)
    
    # Step 1: Load model
    rnn = load_and_setup_model(model_path, input_size, hidden_size)
    
    # Step 2: Compute PCA from data with input
    pca = compute_pca_from_data(rnn, time_steps=time_steps_pca, 
                               n_components=n_components)
    
    # Step 3: Generate variance contour plots
    generate_variance_contour_plots(rnn, pca, title, save_path,
                                   variances=variances, steps=steps,
                                   pc_axes=pc_axes, bw_adjust=bw_adjust)
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Return results for further analysis if needed
    results = {
        'pca': pca,
        'variances': variances
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    model_path = "/home/elia/Documents/rnnrep/TrainedModels/ReverseEngineeredModel/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth"
    input_size = 100
    hidden_size = 150
    
    # Run with default parameters
    results = run_variance_pipeline(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        variances=[0.1, 1.0, 2.0, 3.0, 4.0],
        steps=10000,
        time_steps_pca=30000,
        n_components=10,
        pc_axes=(0, 1),  # PC1 vs PC2
        save_path='plots/variance/variance_2.svg', 
        title='RNN Variance Contour Plot (95% Confidence)',
        bw_adjust=3.0,
        random_seed=42
    )