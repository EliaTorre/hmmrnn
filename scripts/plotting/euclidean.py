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
from scripts.sinkhorn import SinkhornSolver


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


def calculate_sinkhorn_distances(
    folder_path: str,
    num_seq: int = 5000
) -> Dict[str, float]:
    """
    Calculate Sinkhorn-aligned Euclidean distances for models in the given folder.
    
    Args:
        folder_path: Path to the folder containing models organized by seeds
        num_seq: Number of sequences to generate
        
    Returns:
        Dictionary with means and standard deviations for HMM and RNN distances
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
    seq_len = config.get('seq_len', 100)  # Get seq_len from config
    
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
    
    # Device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Sinkhorn solver
    sinkhorn = SinkhornSolver(epsilon=0.1, iterations=1000)
    
    # Lists to store distances
    hmm_means_list = []
    hmm_stds_list = []
    rnn_means_list = []
    rnn_stds_list = []
    
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
        
        # Generate sequences for HMM baseline
        print("  Generating HMM test sequences...")
        hmm_data2, _ = hmm.gen_seq(num_seq, seq_len)
        hmm_data3, _ = hmm.gen_seq(num_seq, seq_len)
        
        # Generate sequences for HMM-RNN comparison
        hmm_data, _ = hmm.gen_seq(num_seq, seq_len)
        
        print("  Generating RNN test sequences...")
        rnn_outputs = rnn.gen_seq(
            dynamics_mode="full",
            batch_mode=True,
            num_seq=num_seq,
            seq_len=seq_len
        )
        rnn_data = rnn_outputs["outs"]
        
        # Match sequences using Sinkhorn transport
        print("  Matching sequences...")
        
        # Prepare HMM baseline data
        hmm_data2_flat = hmm_data2.reshape(hmm_data2.shape[0], -1)
        hmm_data2_flat = torch.tensor(hmm_data2_flat).float().to(device)
        
        hmm_data3_flat = hmm_data3.reshape(hmm_data3.shape[0], -1)
        hmm_data3_flat = torch.tensor(hmm_data3_flat).float().to(device)
        
        # Calculate transport plan for HMM-HMM baseline
        tp_hmm = sinkhorn(hmm_data2_flat, hmm_data3_flat)
        
        # Match sequences based on transport plan
        hmm_matched1 = hmm_data2_flat[tp_hmm[1].argmax(0)].cpu().detach().numpy()
        hmm_matched2 = hmm_data3_flat.cpu().detach().numpy()
        
        # Prepare HMM-RNN comparison data
        hmm_data_flat = hmm_data.reshape(hmm_data.shape[0], -1)
        hmm_data_flat = torch.tensor(hmm_data_flat).float().to(device)
        
        rnn_data_flat = rnn_data.reshape(rnn_data.shape[0], -1)
        rnn_data_flat = torch.tensor(rnn_data_flat).float().to(device)
        
        # Calculate transport plan for HMM-RNN
        tp_rnn = sinkhorn(hmm_data_flat, rnn_data_flat)
        
        # Match sequences based on transport plan
        hmm_matched_rnn = hmm_data_flat[tp_rnn[1].argmax(0)].cpu().detach().numpy()
        rnn_matched_hmm = rnn_data_flat.cpu().detach().numpy()
        
        # Calculate distances
        print("  Calculating Euclidean distances...")
        hmm_distances = np.linalg.norm(hmm_matched1 - hmm_matched2, axis=1)
        rnn_distances = np.linalg.norm(hmm_matched_rnn - rnn_matched_hmm, axis=1)
        
        # Store both mean and std for this seed
        hmm_means_list.append(np.mean(hmm_distances))
        hmm_stds_list.append(np.std(hmm_distances))
        rnn_means_list.append(np.mean(rnn_distances))
        rnn_stds_list.append(np.std(rnn_distances))
        
        print(f"  HMM distance: {np.mean(hmm_distances):.4f} ± {np.std(hmm_distances):.4f}")
        print(f"  RNN distance: {np.mean(rnn_distances):.4f} ± {np.std(rnn_distances):.4f}")
    
    # Calculate statistics: mean of means and mean of stds
    results = {
        "hmm_mean": np.mean(hmm_means_list),
        "hmm_std": np.mean(hmm_stds_list),  # Mean of individual stds
        "rnn_mean": np.mean(rnn_means_list),
        "rnn_std": np.mean(rnn_stds_list)   # Mean of individual stds
    }
    
    return results


def plot_distances(results: Dict[str, float], output_path: str = "sinkhorn_distances.svg"):
    """
    Create a bar plot comparing HMM and RNN distances using the Test class style.
    
    Args:
        results: Dictionary with means and standard deviations
        output_path: Path to save the output figure
    """
    # Prepare data for plotting
    x_positions = [0, 0.5]
    hmm_mean = results['hmm_mean']
    hmm_std = results['hmm_std']
    rnn_mean = results['rnn_mean']
    rnn_std = results['rnn_std']
    
    # Create figure with same size as Test class
    plt.figure(figsize=(5, 5))
    
    # Plot error bars with same colors as Test class
    plt.errorbar(x_positions[0], hmm_mean, yerr=hmm_std, 
                 fmt='o', color='darkgreen', capsize=3)
    plt.errorbar(x_positions[1], rnn_mean, yerr=rnn_std, 
                 fmt='o', color='royalblue', capsize=3)
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, ["HMM", "RNN"], fontsize=10)
    
    # Add value labels above error bars
    for i, (mean, std) in enumerate(zip([hmm_mean, rnn_mean], [hmm_std, rnn_std])):
        plt.text(x_positions[i], mean + std + 0.1, f"{mean:.2f}", ha='center', fontsize=8)
    
    # Set y-axis limits to be +/- 1 of the overall mean
    overall_mean = (hmm_mean + rnn_mean) / 2
    max_std = max(hmm_std, rnn_std)
    plt.ylim(overall_mean - (max_std + 0.5) , overall_mean + (max_std + 0.5))
    
    # Set title and ylabel with same style as Test class
    plt.title("Mean Euclidean Distances", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Distance")
    
    # Use tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main(folder_path: str):
    """
    Main function to calculate and plot Sinkhorn-aligned Euclidean distances.
    
    Args:
        folder_path: Path to the folder containing models organized by seeds
    """
    print(f"Processing models in: {folder_path}")
    print("=" * 70)
    
    # Calculate distances (seq_len will be read from config.json)
    results = calculate_sinkhorn_distances(
        folder_path=folder_path,
        num_seq=5000
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"HMM Distance: {results['hmm_mean']:.4f} ± {results['hmm_std']:.4f}")
    print(f"RNN Distance: {results['rnn_mean']:.4f} ± {results['rnn_std']:.4f}")
    print("=" * 70)
    
    # Create plot
    plot_distances(results, "plots/sinkhorn_distances_fully.svg")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python euclidean.py <folder_path>")
        print("Example: python euclidean.py /home/elia/Documents/rnnrep/TrainedModels/Cyclic")
        print("\nNote: Sequence length will be automatically read from config.json")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    main(folder_path)