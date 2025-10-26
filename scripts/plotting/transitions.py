import os
import sys
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add parent directory to path to import from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.hmm import HMM
from scripts.mechint import load_model


def get_model_paths_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    """
    Extract model paths from a folder containing models organized by seeds.
    
    Args:
        folder_path: Path to the folder containing seed subfolders
        
    Returns:
        List of tuples (model_path, seed_name)
    """
    model_paths = []
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {folder_path} does not exist.")
        return model_paths
    
    # Look for seed folders (case-insensitive)
    for seed_folder in os.listdir(folder_path):
        seed_path = os.path.join(folder_path, seed_folder)
        
        # Check if it's a directory and contains "seed" in the name (case-insensitive)
        if os.path.isdir(seed_path) and "seed" in seed_folder.lower():
            models_dir = os.path.join(seed_path, "models")
            
            # Check if models directory exists
            if os.path.exists(models_dir):
                # Find all .pth files in the models directory
                for model_file in glob.glob(os.path.join(models_dir, "*.pth")):
                    # Skip files in evolution subfolder
                    if "evolution" not in model_file:
                        model_paths.append((model_file, seed_folder))
    
    return model_paths


def load_config_from_folder(folder_path: str) -> Dict:
    """
    Load configuration from the first available config.json file in seed folders.
    
    Args:
        folder_path: Path to the folder containing seed subfolders
        
    Returns:
        Configuration dictionary or None if not found
    """
    for seed_folder in os.listdir(folder_path):
        seed_path = os.path.join(folder_path, seed_folder)
        
        if os.path.isdir(seed_path) and "seed" in seed_folder.lower():
            config_path = os.path.join(seed_path, "config.json")
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load config file {config_path}: {e}")
    
    return None


def calculate_transition_matrix(sequences, num_outputs):
    """Calculate transition matrix between states."""
    # Convert sequences to numpy if they're torch tensors
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    
    # Initialize transition matrix
    trans_matrix = np.zeros((num_outputs, num_outputs))
    
    # Get the most probable output at each time step
    seq_max = np.argmax(sequences, axis=2)
    
    # Count transitions
    for i in range(sequences.shape[0]):  # For each sequence
        for j in range(1, sequences.shape[1]):  # For each time step (except the first)
            prev_state = seq_max[i, j-1]
            curr_state = seq_max[i, j]
            trans_matrix[prev_state, curr_state] += 1
    
    # Normalize by row sums to get probabilities
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    trans_matrix = trans_matrix / row_sums
    
    return trans_matrix


def calculate_transition_matrix_differences(
    folder_path: str,
    num_seq: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Calculate squared differences between ground truth and RNN transition matrices.
    
    Args:
        folder_path: Path to the folder containing models organized by seeds
        num_seq: Number of sequences to generate
        
    Returns:
        Dictionary with average and std of squared difference matrices
    """
    # Get model paths
    model_paths = get_model_paths_from_folder(folder_path)
    
    if not model_paths:
        raise ValueError(f"No models found in {folder_path}")
    
    print(f"Found {len(model_paths)} models across {len(set(seed for _, seed in model_paths))} seeds")
    
    # Load configuration
    config = load_config_from_folder(folder_path)
    
    if config is None:
        raise ValueError(f"No config.json found in {folder_path}")
    
    print(f"Loaded configuration: {config}")
    
    # Extract parameters from config
    states = config.get('states', 2)
    outputs = config.get('outputs', 3)
    stay_prob = config.get('stay_prob', 0.95)
    target_prob = config.get('target_prob', 0.05)
    transition_method = config.get('transition_method', 'target_prob')
    emission_method = config.get('emission_method', 'linear')
    input_size_val = config.get('input_size', 1)
    hidden_size_val = config.get('hidden_size', 50)
    seq_len = config.get('seq_len', 100)
    
    # Get custom matrices if they exist
    custom_transition_matrix = config.get('custom_transition_matrix', None)
    custom_emission_matrix = config.get('custom_emission_matrix', None)
    
    # Convert to numpy arrays if they exist
    if custom_transition_matrix is not None:
        custom_transition_matrix = np.array(custom_transition_matrix)
        print(f"Using custom transition matrix from config")
    
    if custom_emission_matrix is not None:
        custom_emission_matrix = np.array(custom_emission_matrix)
        print(f"Using custom emission matrix from config")
    
    print(f"Using sequence length from config: {seq_len}")
    
    # Initialize HMM with custom matrices if provided
    hmm = HMM(
        states=states,
        outputs=outputs,
        stay_prob=stay_prob,
        target_prob=target_prob,
        transition_method=transition_method,
        emission_method=emission_method,
        custom_transition_matrix=custom_transition_matrix,
        custom_emission_matrix=custom_emission_matrix
    )
    
    # Generate ground truth transition matrix from HMM sequences
    print("\nGenerating ground truth transition matrix from HMM...")
    hmm_data, _ = hmm.gen_seq(num_seq, seq_len)
    ground_truth_tm = calculate_transition_matrix(hmm_data, outputs)
    
    print(f"Ground truth transition matrix:")
    print(ground_truth_tm)
    
    # Device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Lists to store squared difference matrices across seeds
    squared_diff_matrices = []
    
    # Process each model
    for model_path, seed_name in model_paths:
        print(f"\nProcessing {seed_name}: {os.path.basename(model_path)}")
        
        # Load RNN model
        rnn = load_model(
            model_path=model_path,
            input_size=input_size_val,
            hidden_size=hidden_size_val,
            output_size=outputs
        )
        
        print(f"  Using num_seq={num_seq}, seq_len={seq_len}")
        
        # Generate sequences for RNN
        print("  Generating RNN test sequences...")
        rnn_outputs = rnn.gen_seq(
            dynamics_mode="full",
            batch_mode=True,
            num_seq=num_seq,
            seq_len=seq_len
        )
        rnn_data = rnn_outputs["outs"]
        
        # Convert RNN outputs to numpy
        if torch.is_tensor(rnn_data):
            rnn_data = rnn_data.cpu().detach().numpy()
        
        # Calculate RNN transition matrix
        rnn_tm = calculate_transition_matrix(rnn_data, outputs)
        
        print(f"  RNN transition matrix:")
        print(rnn_tm)
        
        # Calculate squared difference
        squared_diff = (ground_truth_tm - rnn_tm) ** 2
        squared_diff_matrices.append(squared_diff)
        
        print(f"  Squared difference matrix:")
        print(squared_diff)
    
    # Calculate mean and std of squared differences across seeds
    avg_squared_diff = np.mean(squared_diff_matrices, axis=0)
    std_squared_diff = np.std(squared_diff_matrices, axis=0)
    
    results = {
        "avg_matrix": avg_squared_diff,
        "std_matrix": std_squared_diff,
        "ground_truth_tm": ground_truth_tm
    }
    
    return results


def plot_squared_difference_matrix(results: Dict[str, np.ndarray], output_path: str = "transition_matrix_squared_diff.svg"):
    """
    Create a plot showing the averaged squared difference matrix.
    
    Args:
        results: Dictionary with avg_matrix, std_matrix, and ground_truth_tm
        output_path: Path to save the output figure
    """
    avg_matrix = results["avg_matrix"]
    std_matrix = results["std_matrix"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Set fixed color scale limits to match the colorbar range
    vmin = 0.001
    vmax = 0.020
    
    # Plot the matrix with fixed color scale
    im = ax.matshow(avg_matrix, cmap="Blues", alpha=0.7, vmin=vmin, vmax=vmax)
    
    # Add text annotations with mean ± std
    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            ax.text(j, i, f"{avg_matrix[i, j]:.2f}\n±{std_matrix[i, j]:.2f}", 
                    ha="center", va="center", fontsize=10)
    
    # Set title and labels
    ax.set_title("Transition Matrix Squared Differences\n(Ground Truth HMM - RNN)", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("To Output", fontsize=12)
    ax.set_ylabel("From Output", fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(avg_matrix.shape[1]))
    ax.set_yticks(np.arange(avg_matrix.shape[0]))
    ax.set_xticklabels(np.arange(1, avg_matrix.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, avg_matrix.shape[0] + 1))
    
    # Add colorbar with specific ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Squared Difference", fontsize=12)
    cbar.set_ticks([0.001, 0.010, 0.015, 0.020])
    
    # Use tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main(folder_path: str):
    """
    Main function to calculate and plot transition matrix squared differences.
    
    Args:
        folder_path: Path to the folder containing models organized by seeds
    """
    print(f"Processing models in: {folder_path}")
    print("=" * 70)
    
    # Calculate squared differences (seq_len will be read from config.json)
    results = calculate_transition_matrix_differences(
        folder_path=folder_path,
        num_seq=5000
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Ground Truth Transition Matrix:")
    print(results["ground_truth_tm"])
    print("\nAverage Squared Difference Matrix:")
    print(results["avg_matrix"])
    print("\nStd of Squared Difference Matrix:")
    print(results["std_matrix"])
    print("=" * 70)
    
    # Create plot
    plot_squared_difference_matrix(results, "plots/tran_diff_cyclic.svg")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transitions.py <folder_path>")
        print("\nNote: Sequence length will be automatically read from config.json")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    main(folder_path)