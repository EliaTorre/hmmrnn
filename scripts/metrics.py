import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from scripts.hmm import HMM
from scripts.mechint import load_model
from scripts.sinkhorn import SinkhornSolver

def get_model_paths(structured=False, with_config=False) -> Any:
    """
    Extract model paths from the grid search experiment directory.
    """
    base_dir = "TrainedModels"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist.")
        return [] if not structured else {}
    
    model_types = ["Two", "Three", "Four", "Five"]
    hidden_sizes = ["hidden_50", "hidden_150", "hidden_200"]
    input_sizes = ["input_1", "input_10", "input_100", "input_200"]
    seeds = ["seed_0", "seed_1", "seed_2"]
    
    if structured:
        # Create a nested dictionary to organize models
        structured_paths = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list if not with_config else dict)
                )
            )
        )
    
    all_model_paths = []
    
    # Traverse the directory structure
    for model_type in model_types:
        for hidden_size in hidden_sizes:
            for input_size in input_sizes:
                for seed in seeds:
                    # Construct the path to the experiment directory
                    exp_dir = os.path.join(base_dir, model_type, hidden_size, input_size, seed)
                    models_dir = os.path.join(exp_dir, "models")
                    
                    # Skip if the models directory doesn't exist
                    if not os.path.exists(models_dir):
                        continue
                    
                    # Try to load the config file
                    config_path = os.path.join(exp_dir, "config.json")
                    config_data = None
                    if with_config and os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                        except Exception as e:
                            print(f"Warning: Could not load config file {config_path}: {e}")
                    
                    # Find all .pth files in the models directory (excluding evolution subfolder)
                    model_paths = []
                    model_configs = []
                    
                    for model_file in glob.glob(os.path.join(models_dir, "*.pth")):
                        # Skip files in the evolution subfolder
                        if "evolution" not in model_file:
                            model_paths.append(model_file)
                            if with_config:
                                model_configs.append((model_file, config_data))
                    
                    # Add the model paths to the appropriate structure
                    if model_paths:
                        if structured:
                            if with_config:
                                # Create a dictionary mapping model paths to configs
                                model_config_dict = {model_path: config for model_path, config in model_configs}
                                structured_paths[model_type][hidden_size][input_size][seed] = model_config_dict
                            else:
                                structured_paths[model_type][hidden_size][input_size][seed] = model_paths
                        
                        if with_config:
                            all_model_paths.append(model_configs)
                        else:
                            all_model_paths.append(model_paths)
    
    return structured_paths if structured else all_model_paths

def get_flat_model_list(with_config=False) -> List[Any]:
    model_groups = get_model_paths(structured=False, with_config=with_config)
    
    if with_config:
        return [item for group in model_groups for item in group]
    else:
        return [model for group in model_groups for model in group]

def get_specific_models(model_type=None, hidden_size=None, input_size=None, seed=None, with_config=False) -> List[Any]:
    structured_paths = get_model_paths(structured=True, with_config=with_config)
    result = []
    
    # Filter by model type
    model_types = [model_type] if model_type else structured_paths.keys()
    
    for mt in model_types:
        if mt not in structured_paths:
            continue
            
        # Filter by hidden size
        hidden_sizes = [hidden_size] if hidden_size else structured_paths[mt].keys()
        
        for hs in hidden_sizes:
            if hs not in structured_paths[mt]:
                continue
                
            # Filter by input size
            input_sizes = [input_size] if input_size else structured_paths[mt][hs].keys()
            
            for is_ in input_sizes:
                if is_ not in structured_paths[mt][hs]:
                    continue
                    
                # Filter by seed
                seeds = [seed] if seed else structured_paths[mt][hs][is_].keys()
                
                for s in seeds:
                    if s not in structured_paths[mt][hs][is_]:
                        continue
                    
                    if with_config:
                        # Add tuples of (model_path, config) to the result
                        for model_path, config in structured_paths[mt][hs][is_][s].items():
                            result.append((model_path, config))
                    else:
                        # Add just the model paths to the result
                        result.extend(structured_paths[mt][hs][is_][s])
    
    return result



def get_models_with_configs() -> List[Tuple[str, Dict]]:
    return get_flat_model_list(with_config=True)

def compare_hmm_rnn_transition_matrices():
    """
    Compare HMM and RNN models by calculating transition matrices and their squared differences.
    """
    # Get all models with their configurations
    models_with_configs = get_models_with_configs()
    
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
        
        grouped_models[model_type][hidden_size][input_size][seed].append((model_path, config))
    
    # Initialize results dictionary
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )
    
    # Process each model
    for model_type, hidden_sizes in grouped_models.items():
        print(f"Processing {model_type} models...")
        
        for hidden_size, input_sizes in hidden_sizes.items():
            print(f"  Processing {hidden_size}...")
            
            for input_size, seeds in input_sizes.items():
                print(f"    Processing {input_size}...")
                
                # List to store difference matrices for this combination
                diff_matrices = []
                
                for seed, models in seeds.items():
                    print(f"      Processing {seed}...")
                    
                    for model_path, config in models:
                        print(f"        Processing model: {os.path.basename(model_path)}")
                        
                        # Extract parameters from config
                        states = config.get('states', 2)
                        outputs = config.get('outputs', 3)
                        stay_prob = config.get('stay_prob', 0.95)
                        target_prob = config.get('target_prob', 0.05)
                        transition_method = config.get('transition_method', 'target_prob')
                        emission_method = config.get('emission_method', 'linear')
                        input_size_val = config.get('input_size', 1)
                        hidden_size_val = config.get('hidden_size', 50)
                        
                        # Initialize HMM
                        hmm = HMM(
                            states=states,
                            outputs=outputs,
                            stay_prob=stay_prob,
                            target_prob=target_prob,
                            transition_method=transition_method,
                            emission_method=emission_method
                        )
                        
                        # Load RNN model
                        rnn = load_model(
                            model_path=model_path,
                            input_size=input_size_val,
                            hidden_size=hidden_size_val,
                            output_size=outputs
                        )
                        
                        # Set sequence parameters based on model type
                        num_seq = 5000  # 5k for all models
                        seq_len = 100
                            
                        print(f"        Using num_seq={num_seq}, seq_len={seq_len}")
                        
                        # Generate sequences
                        hmm_sequences, _ = hmm.gen_seq(num_seq, seq_len)
                        rnn_outputs = rnn.gen_seq(dynamics_mode="full", batch_mode=True, 
                                                num_seq=num_seq, seq_len=seq_len)
                        rnn_sequences = rnn_outputs["outs"]
                        
                        # Calculate transition matrices
                        hmm_trans_matrix = calculate_transition_matrix(hmm_sequences.numpy(), outputs)
                        rnn_trans_matrix = calculate_transition_matrix(rnn_sequences, outputs)
                        
                        # Calculate differences
                        diff_matrix = (hmm_trans_matrix - rnn_trans_matrix)**2
                        diff_matrices.append(diff_matrix)
                
                # Average squared difference matrices across seeds
                if diff_matrices:
                    avg_diff_matrix = np.mean(diff_matrices, axis=0)
                    std_diff_matrix = np.std(diff_matrices, axis=0)
                    results[model_type][hidden_size][input_size] = (avg_diff_matrix, std_diff_matrix)
    
    # Create plots
    plot_transition_matrix_differences(results)
    
    return results

def calculate_transition_matrix(sequences, num_outputs):
    """
    Calculate transition matrix between states.
    """
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

def plot_transition_matrix_differences(results):
    """
    Create a grid of plots showing the averaged squared difference matrices.
    """
    model_types = ["Two", "Three", "Four", "Five"]
    hidden_sizes = ["hidden_50", "hidden_150", "hidden_200"]
    input_sizes = ["input_1", "input_10", "input_100", "input_200"]
    
    # Create a figure for each model type
    for model_type in model_types:
        if model_type not in results:
            continue
            
        # Create a grid of subplots (3 rows for hidden sizes, 4 columns for input sizes)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
        fig.suptitle(f"Transition Matrix Squared Differences: {model_type}", fontsize=16)
        
        # Find global min and max for consistent colorbar
        vmin, vmax = float('inf'), float('-inf')
        for hidden_size in hidden_sizes:
            if hidden_size not in results[model_type]:
                continue
                
            for input_size in input_sizes:
                if input_size not in results[model_type][hidden_size]:
                    continue
                    
                avg_matrix, _ = results[model_type][hidden_size][input_size]
                vmin = min(vmin, np.min(avg_matrix))
                vmax = max(vmax, np.max(avg_matrix))
        
        # If no data was found, skip this model type
        if vmin == float('inf') or vmax == float('-inf'):
            plt.close(fig)
            continue
        
        # Plot each matrix
        for i, hidden_size in enumerate(hidden_sizes):
            if hidden_size not in results[model_type]:
                continue
                
            for j, input_size in enumerate(input_sizes):
                if input_size not in results[model_type][hidden_size]:
                    continue
                    
                ax = axes[i, j]
                avg_matrix, std_matrix = results[model_type][hidden_size][input_size]
                
                # Plot the matrix
                im = ax.matshow(avg_matrix, cmap="Blues", vmin=vmin, vmax=vmax, alpha=0.7)
                
                # Add text annotations with mean ± std
                for ii in range(avg_matrix.shape[0]):
                    for jj in range(avg_matrix.shape[1]):
                        ax.text(jj, ii, f"{avg_matrix[ii, jj]:.2f}\n±{std_matrix[ii, jj]:.2f}", 
                                ha="center", va="center", fontsize=10)
                
                # Set title and labels
                hidden_val = hidden_size.split('_')[1]
                input_val = input_size.split('_')[1]
                ax.set_title(f"Hidden: {hidden_val}, Input: {input_val}")
                ax.set_xlabel("To Output")
                ax.set_ylabel("From Output")
                
                # Set ticks
                ax.set_xticks(np.arange(avg_matrix.shape[1]))
                ax.set_yticks(np.arange(avg_matrix.shape[0]))
                ax.set_xticklabels(np.arange(1, avg_matrix.shape[1] + 1))
                ax.set_yticklabels(np.arange(1, avg_matrix.shape[0] + 1))
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label("Squared Difference")
        
        # Save the figure
        plt.savefig(f"transition_matrix_differences_{model_type}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

def compare_hmm_rnn_euclidean_distances():
    """
    Compare HMM and RNN models by calculating Euclidean distances between matched sequences.
    """
    # Get all models with their configurations
    models_with_configs = get_models_with_configs()
    
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
        
        grouped_models[model_type][hidden_size][input_size][seed].append((model_path, config))
    
    # Initialize results dictionary
    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )
    
    # Device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process each model
    for model_type, hidden_sizes in grouped_models.items():
        print(f"Processing {model_type} models...")
        
        for hidden_size, input_sizes in hidden_sizes.items():
            print(f"  Processing {hidden_size}...")
            
            for input_size, seeds in input_sizes.items():
                print(f"    Processing {input_size}...")
                
                # List to store distances for this combination
                hmm_distances_list = []
                rnn_distances_list = []
                
                for seed, models in seeds.items():
                    print(f"      Processing {seed}...")
                    
                    for model_path, config in models:
                        print(f"        Processing model: {os.path.basename(model_path)}")
                        
                        # Extract parameters from config
                        states = config.get('states', 2)
                        outputs = config.get('outputs', 3)
                        stay_prob = config.get('stay_prob', 0.95)
                        target_prob = config.get('target_prob', 0.05)
                        transition_method = config.get('transition_method', 'target_prob')
                        emission_method = config.get('emission_method', 'linear')
                        input_size_val = config.get('input_size', 1)
                        hidden_size_val = config.get('hidden_size', 50)
                        
                        # Initialize HMM
                        hmm = HMM(
                            states=states,
                            outputs=outputs,
                            stay_prob=stay_prob,
                            target_prob=target_prob,
                            transition_method=transition_method,
                            emission_method=emission_method
                        )
                        
                        # Load RNN model
                        rnn = load_model(
                            model_path=model_path,
                            input_size=input_size_val,
                            hidden_size=hidden_size_val,
                            output_size=outputs
                        )
                        
                        # Set sequence parameters based on model type
                        num_seq = 5000  # 5k for all models
                        seq_len = 100
                            
                        print(f"        Using num_seq={num_seq}, seq_len={seq_len}")
                        
                        # Generate sequences
                        print("        Generating HMM test sequences...")
                        hmm_data, _ = hmm.gen_seq(num_seq, seq_len)
                        hmm_data2, _ = hmm.gen_seq(num_seq, seq_len)
                        hmm_data3, _ = hmm.gen_seq(num_seq, seq_len)
                        
                        print("        Generating RNN test sequences...")
                        rnn_outputs = rnn.gen_seq(dynamics_mode="full", batch_mode=True, 
                                                num_seq=num_seq, seq_len=seq_len)
                        rnn_data = rnn_outputs["outs"]
                        
                        # Match sequences using Sinkhorn transport
                        print("        Matching sequences...")
                        sinkhorn = SinkhornSolver(epsilon=0.1, iterations=1000)
                        
                        # Match HMM with HMM for baseline
                        hmm_data2_flat = hmm_data2.reshape(hmm_data2.shape[0], -1)
                        hmm_data2_flat = torch.tensor(hmm_data2_flat).float().to(device)
                        
                        hmm_data3_flat = hmm_data3.reshape(hmm_data3.shape[0], -1)
                        hmm_data3_flat = torch.tensor(hmm_data3_flat).float().to(device)
                        
                        # Calculate transport plan for HMM-HMM
                        tp_hmm = sinkhorn(hmm_data2_flat, hmm_data3_flat)
                        
                        # Match sequences based on transport plan
                        hmm_matched1 = hmm_data2_flat[tp_hmm[1].argmax(0)].cpu().detach().numpy()
                        hmm_matched2 = hmm_data3_flat.cpu().detach().numpy()
                        
                        # Match HMM with RNN
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
                        print("        Calculating Euclidean distances...")
                        hmm_distances = np.linalg.norm(hmm_matched1 - hmm_matched2, axis=1)
                        rnn_distances = np.linalg.norm(hmm_matched_rnn - rnn_matched_hmm, axis=1)
                        
                        # Store distances
                        hmm_distances_list.append(np.mean(hmm_distances))
                        rnn_distances_list.append(np.mean(rnn_distances))
                
                # Store results for this combination
                if hmm_distances_list and rnn_distances_list:
                    results[model_type][hidden_size][input_size] = {
                        "hmm_mean": np.mean(hmm_distances_list),
                        "hmm_std": np.std(hmm_distances_list),
                        "rnn_mean": np.mean(rnn_distances_list),
                        "rnn_std": np.std(rnn_distances_list)
                    }
    
    # Create plots
    plot_euclidean_distances(results)
    
    return results

def plot_euclidean_distances(results):
    """
    Create plots showing the Euclidean distances for different configurations.
    
    Args:
        results: Dictionary containing the averaged Euclidean distances
    """
    model_types = ["Two", "Three", "Four", "Five"]
    hidden_sizes = ["hidden_50", "hidden_150", "hidden_200"]
    input_sizes = ["input_1", "input_10", "input_100", "input_200"]
    
    # Map input sizes to equidistant positions (0, 1, 2, 3)
    input_positions = {"input_1": 0, "input_10": 1, "input_100": 2, "input_200": 3}
    
    # Colors for different hidden sizes
    colors = {"hidden_50": "deepskyblue", "hidden_150": "royalblue", "hidden_200": "midnightblue"}
    
    # Offsets for different hidden sizes to avoid overlap
    offsets = {"hidden_50": -0.1, "hidden_150": 0.0, "hidden_200": 0.1}
    
    # Create a figure for each model type
    for model_type in model_types:
        if model_type not in results:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.title(f"Euclidean Distances: {model_type}", fontsize=16)
        # Plot HMM line as a true horizontal line spanning the entire x-axis
        hmm_mean = results[model_type]["hidden_50"]["input_1"]["hmm_mean"]
        hmm_std = results[model_type]['hidden_50']['input_1']['hmm_std']
        
        # Use axhline for a true horizontal line
        plt.axhline(y=hmm_mean, color='darkgreen', linestyle='-', label="HMM")
        
        # Fill between for the entire x-axis range
        plt.axhspan(hmm_mean - hmm_std, hmm_mean + hmm_std, color='darkgreen', alpha=0.2)
        
        # Plot each hidden size as a separate line
        for hidden_size in hidden_sizes:
            if hidden_size not in results[model_type]:
                continue
                
            # Collect data points for this hidden size
            x_values = []
            y_values = []
            y_errors = []
            x_positions = []  # For scatter points with offset
            
            for input_size in input_sizes:
                if input_size not in results[model_type][hidden_size]:
                    continue
                    
                x_values.append(input_positions[input_size])
                x_positions.append(input_positions[input_size] + offsets[hidden_size])
                y_values.append(results[model_type][hidden_size][input_size]["rnn_mean"])
                y_errors.append(results[model_type][hidden_size][input_size]["rnn_std"])
            
            # Plot the line and error bars
            if x_values and y_values:
                # Plot connecting lines using the offset x positions
                plt.plot(
                    x_positions, 
                    y_values, 
                    '-', 
                    color=colors[hidden_size],
                    alpha=0.5
                )
                
                # Plot error bars with offset x positions
                plt.errorbar(
                    x_positions, 
                    y_values, 
                    yerr=y_errors, 
                    fmt='o', 
                    color=colors[hidden_size], 
                    label=f"Hidden: {hidden_size.split('_')[1]}",
                    capsize=5
                )
        
        # Set axis labels and legend
        plt.xlabel("Input Dimension", fontsize=12)
        plt.ylabel("Euclidean Distance", fontsize=12)
        plt.xticks(range(4), ["1", "10", "100", "200"])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure
        plt.savefig(f"euclidean_distances_{model_type}.png", dpi=600, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    compare_hmm_rnn_transition_matrices()
    compare_hmm_rnn_euclidean_distances()
