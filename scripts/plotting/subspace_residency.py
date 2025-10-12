import os, torch, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from scripts.rnn import RNN
from sklearn.decomposition import PCA
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from datetime import datetime

def load_model_for_subspace(model_path, input_size=100, hidden_size=150, output_size=3):
    """Load RNN model and extract weights."""
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=output_size)
    rnn.load_state_dict(torch.load(model_path))
    rnn.eval()
    return rnn

def generate_hidden_states_with_outputs(rnn, time_steps=100000, sigma=1, t0=20):
    """Generate hidden states and outputs by running the model."""
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc = rnn.fc.weight.data
    device = ih.device
    
    # Warm-up
    h = torch.zeros(rnn.hidden_size).to(device)
    for _ in range(t0):
        x = torch.normal(mean=0, std=sigma, size=(rnn.input_size,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
    
    # Generate sequence
    h_sequence = []
    outputs = []
    for _ in range(time_steps):
        x = torch.normal(mean=0, std=sigma, size=(rnn.input_size,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
        h_sequence.append(h)
        
        # Compute output
        logits = h @ fc.T
        output = torch.argmax(logits).item()
        outputs.append(output)
    
    h_sequence = torch.stack(h_sequence)
    outputs = np.array(outputs)
    
    return h_sequence, outputs, ih, hh, fc, device

def compute_residency_metrics(h, ih, hh, fc, sigma, max_steps, n_samples, device):
    """Compute residency time and sign changes for a single hidden state."""
    logits = h @ fc.T
    probs = torch.softmax(logits, dim=0)
    most_probable = torch.argmax(probs).item()
    initial_logit = logits[most_probable].item()
    
    steps_list = []
    sign_changes_list = []
    
    for _ in range(n_samples):
        h_current = h.clone()
        prev_logit = initial_logit
        sign_changes = 0
        prev_delta_sign = None
        steps = 0
        while steps < max_steps:
            x = torch.normal(mean=0, std=sigma, size=(ih.shape[1],)).float().to(device)
            h_current = torch.relu(x @ ih.T + h_current @ hh.T)
            new_logits = h_current @ fc.T
            new_probs = torch.softmax(new_logits, dim=0)
            new_most_probable = torch.argmax(new_probs).item()
            if new_most_probable != most_probable:
                steps_list.append(steps + 1)
                sign_changes_list.append(sign_changes)
                break
            current_logit = new_logits[most_probable].item()
            delta_logit = current_logit - prev_logit
            current_sign = 1 if delta_logit > 0 else (-1 if delta_logit < 0 else 0)
            if prev_delta_sign is not None and current_sign != prev_delta_sign and current_sign != 0:
                sign_changes += 1
            prev_delta_sign = current_sign
            prev_logit = current_logit
            steps += 1
        else:
            steps_list.append(max_steps)
            sign_changes_list.append(sign_changes)
    
    avg_steps = np.mean(steps_list)
    avg_sign_changes = np.mean(sign_changes_list)
    return avg_steps, avg_sign_changes

def compute_unstable_eigenvalues(h, ih, hh, device):
    """Compute number of unstable eigenvalues after Mobius transform."""
    x = torch.normal(mean=0, std=0, size=(ih.shape[1],)).float().to(device)
    pre_act = x @ ih.T + h @ hh.T
    D = torch.diag((pre_act > 0).float())
    J = D @ hh
    eigvals = torch.linalg.eigvals(J)
    mobius = (eigvals - 1) / (eigvals + 1 + 1e-8)
    num_unstable = (mobius.real > 0).sum().item()
    return num_unstable

def compute_all_metrics(h_sequence, ih, hh, fc, sigma, max_steps, n_samples, device):
    """Compute residency metrics and unstable eigenvalues for ALL hidden states."""
    print("Computing metrics for all hidden states...")
    avg_steps_list = []
    avg_sign_changes_list = []
    num_unstable_list = []
    
    for h in tqdm(h_sequence, desc="Computing metrics"):
        # Residency metrics
        avg_steps, avg_sign_changes = compute_residency_metrics(
            h, ih, hh, fc, sigma, max_steps, n_samples, device
        )
        avg_steps_list.append(avg_steps)
        avg_sign_changes_list.append(avg_sign_changes)
        
        # Unstable eigenvalues
        num_unstable = compute_unstable_eigenvalues(h, ih, hh, device)
        num_unstable_list.append(num_unstable)
    
    return np.array(avg_steps_list), np.array(avg_sign_changes_list), np.array(num_unstable_list)

def filter_by_output_combination(h_sequence, outputs, avg_steps, avg_sign_changes, 
                                 num_unstable_list, h_pca_global, output_set, min_length=200):
    """
    Filter sequences and metrics by output combination.
    Uses a more lenient approach: includes ALL points where output is in output_set,
    even if not in long continuous sequences.
    """
    mask = np.isin(outputs, output_set)
    
    # Count how many points match
    n_matching = np.sum(mask)
    print(f"  Found {n_matching} points matching outputs {output_set}")
    
    if n_matching < min_length:
        print(f"  Not enough points (need at least {min_length})")
        return None, None, None, None, None, None
    
    # Simply filter by mask - include all matching points
    filtered_indices = np.where(mask)[0]
    
    # Filter everything using the same indices
    filtered_h = h_sequence[filtered_indices]
    filtered_outputs = outputs[filtered_indices]
    filtered_avg_steps = avg_steps[filtered_indices]
    filtered_avg_sign_changes = avg_sign_changes[filtered_indices]
    filtered_num_unstable = num_unstable_list[filtered_indices]
    filtered_h_global_pca = h_pca_global[filtered_indices]
    
    print(f"  Kept {len(filtered_indices)} states after filtering")
    
    return (filtered_h, filtered_outputs, filtered_avg_steps, 
            filtered_avg_sign_changes, filtered_num_unstable, filtered_h_global_pca)

def plot_subspace_residency(h_pca, avg_steps, avg_sign_changes, num_unstable_list, 
                            fc, filtered_h, output_set, model_path, pc_dims=(0, 2)):
    """
    Create the 4-panel residency plot for a subspace.
    This uses the EXACT same logic as the original residency_plot function.
    
    Args:
        pc_dims: Tuple of PC indices to use for plotting (default: (0, 2) for PC1 vs PC3)
    """
    T = len(filtered_h)
    
    # Segment into groups (same as original)
    cutoffcol = 2
    logits = torch.softmax(filtered_h @ fc.T, dim=1)
    group1 = avg_steps < 2
    group2 = (avg_steps >= cutoffcol) & (avg_steps <= 4*cutoffcol)
    group3 = (avg_steps > 4*cutoffcol) & (torch.max(logits, dim=1)[0].cpu().numpy() > 0.8)
    
    # Compute average statistics per group
    avg_num_unstable_group1 = np.mean([num_unstable_list[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
    avg_num_unstable_group2 = np.mean([num_unstable_list[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
    avg_num_unstable_group3 = np.mean([num_unstable_list[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
    avg_residency_group1 = np.mean([avg_steps[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
    avg_residency_group2 = np.mean([avg_steps[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
    avg_residency_group3 = np.mean([avg_steps[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
    
    # Create subplots with four panels in 1x4 layout
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # Set up custom colormap with BoundaryNorm
    cmap_base = matplotlib.colormaps['inferno']
    color_map = cmap_base(np.linspace(0, 1, 15))
    colors = [
        color_map[10],
        color_map[9],
        np.array([0.6, 0.0, 0.0, 1.0]),
        np.array([0.5, 0.5, 0.5, 1.0]),
    ]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)], cmap.N)
    
    # Extract the specified PC dimensions
    pc_x, pc_y = pc_dims
    x_data = h_pca[:, pc_x]
    y_data = h_pca[:, pc_y]
    
    # First subplot: PCA colored by residency time
    sc1 = ax1.scatter(x_data, y_data, c=avg_steps, cmap=cmap, norm=norm, s=2, alpha=0.9)
    ax1.set_xlabel(f'PC{pc_x+1}', fontsize=14)
    ax1.set_ylabel(f'PC{pc_y+1}', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Second subplot: Histogram of sign changes colored by average residency time
    if max(avg_sign_changes) > 0:
        counts, bin_edges = np.histogram(avg_sign_changes, bins=int(max(avg_sign_changes)))
        bin_avg_steps = []
        for i in range(len(bin_edges) - 1):
            mask = (avg_sign_changes >= bin_edges[i]) & (avg_sign_changes < bin_edges[i+1])
            bin_avg_steps.append(np.mean(avg_steps[mask]) if np.any(mask) else 0)
        
        for i in range(len(counts)):
            norm_value = norm(bin_avg_steps[i])
            color = cmap(norm_value)
            ax2.bar((bin_edges[i] + bin_edges[i+1]) / 2, counts[i], 
                    width=bin_edges[i+1] - bin_edges[i], color=color, alpha=0.9)
    
    ax2.set_xlabel('Avg. Sign Changes in Logits Gradient', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    
    # Third subplot: Bar plot of average unstable eigenvalues per group
    groups = ['RT < 2\n(Transition)', '2 <= RT <= 8\n(Kick-Zone)', 'RT > 8\n(Cluster)']
    avg_num_unstable = [avg_num_unstable_group1, avg_num_unstable_group2, avg_num_unstable_group3]
    avg_residency = [avg_residency_group1, avg_residency_group2, avg_residency_group3]
    bar_colors = [cmap(norm(avg_residency[i])) for i in range(len(groups))]
    ax3.bar(groups, avg_num_unstable, color=bar_colors, alpha=0.9)
    ax3.set_ylabel('Avg. # Eigenvalues Re > 0\n(after Mobius Transform)', fontsize=14)
    ax3.tick_params(axis='x', labelsize=13) 

    # Set up second custom colormap for range 0-3
    colors2 = [
        np.array([0.5, 0.5, 0.5, 1.0]),
        np.array([0.6, 0.0, 0.0, 1.0]),
        color_map[10],
    ]
    cmap2 = mcolors.ListedColormap(colors2)
    norm2 = mcolors.BoundaryNorm([0, 1, 2, 3], cmap2.N)

    # Fourth subplot: PCA colored by number of unstable eigenvalues
    sc2 = ax4.scatter(x_data, y_data, c=num_unstable_list, cmap=cmap2, norm=norm2, s=2, alpha=0.9)
    ax4.set_xlabel(f'PC{pc_x+1}', fontsize=14)
    ax4.set_ylabel(f'PC{pc_y+1}', fontsize=14)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Adjust layout and add colorbars
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    
    # Colorbar for first subplot, spanning first three subplots
    cbar1 = fig.colorbar(sc1, ax=[ax1, ax2, ax3], orientation='horizontal', pad=0.15, aspect=60)
    cbar1.set_ticks([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)])
    cbar1.ax.set_xticklabels(['1', str(cutoffcol), str(2*cutoffcol), str(4*cutoffcol+1), f'{np.max(avg_steps).astype(int)}'], fontsize=14)
    cbar1.set_label('Residency Time (RT)', fontsize=14)
    
    # Colorbar for fourth subplot
    cbar2 = fig.colorbar(sc2, ax=ax4, orientation='horizontal', pad=0.15, aspect=20)
    cbar2.set_label('# Eigenvalues Re > 0', fontsize=14)
    cbar2.set_ticks([0.5, 1.5, 2.5])
    cbar2.ax.set_xticklabels(['0', '1', '2'], fontsize=14)
    
    # Add title
    subspace_name = ''.join(map(str, output_set))
    fig.suptitle(f'Residency Analysis - Outputs {" & ".join(map(str, output_set))} Subspace (PC{pc_x+1} vs PC{pc_y+1})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('scripts/plotting/plots/residency_subspace', exist_ok=True)
    save_path = f'scripts/plotting/plots/residency_subspace/residency_subspace_{subspace_name}_PC{pc_x+1}_PC{pc_y+1}_{datetime_str}.svg'
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.show()
    plt.close()

def subspace_residency_plot(model_path, 
                            output_combinations=None,
                            sigma=1, 
                            max_steps=50, 
                            n_samples=20, 
                            t0=20, 
                            time_steps=100000,
                            min_sequence_length=200,
                            n_components_global=4,
                            input_size=100,
                            hidden_size=150,
                            output_size=3,
                            pc_dims=(0, 2)):
    """
    Generate residency plots for each output subspace.
    
    This function:
    1. Generates hidden states for the entire trajectory
    2. Computes ALL metrics once (residency time, sign changes, unstable eigenvalues)
    3. Filters by output combination
    4. Plots using the exact same logic as residency_plot
    
    Args:
        model_path: Path to the saved model
        output_combinations: List of output combinations (e.g., [[0,1], [1,2], [0,2]])
        sigma: Standard deviation of input noise
        max_steps: Maximum steps for residency computation
        n_samples: Number of samples for residency computation
        t0: Warm-up steps
        time_steps: Total timesteps to generate
        min_sequence_length: Minimum total points needed in subspace
        n_components_global: Number of components for global PCA
        input_size: Input dimension
        hidden_size: Hidden state dimension
        output_size: Output dimension
        pc_dims: Tuple of PC indices to plot (default: (0, 2) for PC1 vs PC3)
    """
    if output_combinations is None:
        output_combinations = [[0, 1], [1, 2], [0, 2]]
    
    print("=" * 60)
    print("Starting subspace residency analysis")
    print("=" * 60)
    
    # Step 1: Load model
    print(f"\nLoading model from: {model_path}")
    rnn = load_model_for_subspace(model_path, input_size, hidden_size, output_size)
    
    # Step 2: Generate hidden states and outputs
    print(f"\nGenerating {time_steps} timesteps...")
    h_sequence, outputs, ih, hh, fc, device = generate_hidden_states_with_outputs(
        rnn, time_steps=time_steps, sigma=sigma, t0=t0
    )
    print(f"Generated {len(h_sequence)} hidden states")
    print(f"Output distribution: {np.bincount(outputs)}")
    
    # Step 3: Compute ALL metrics ONCE (before filtering)
    print(f"\nComputing metrics for all {len(h_sequence)} states...")
    avg_steps, avg_sign_changes, num_unstable_list = compute_all_metrics(
        h_sequence, ih, hh, fc, sigma, max_steps, n_samples, device
    )
    
    # Step 4: Compute global PCA
    print(f"\nComputing global PCA with {n_components_global} components...")
    h_np = h_sequence.cpu().numpy()
    pca_global = PCA(n_components=n_components_global)
    h_pca_global = pca_global.fit_transform(h_np)
    print(f"Global PCA explained variance: {pca_global.explained_variance_ratio_}")
    
    # Step 5: Process each output combination
    for output_set in output_combinations:
        print("\n" + "=" * 60)
        print(f"Processing subspace for outputs {output_set}")
        print("=" * 60)
        
        # Filter sequences and pre-computed metrics
        result = filter_by_output_combination(
            h_sequence, outputs, avg_steps, avg_sign_changes, 
            num_unstable_list, h_pca_global, output_set, min_length=min_sequence_length
        )
        
        if result[0] is None:
            print(f"No sufficient data found for outputs {output_set}. Skipping.")
            continue
        
        (filtered_h, filtered_outputs, filtered_avg_steps, 
         filtered_avg_sign_changes, filtered_num_unstable, filtered_h_global_pca) = result
        
        T = len(filtered_h)
        print(f"Using {T} states in subspace")
        
        # Apply subspace-specific PCA (need at least 3 components for PC1 vs PC3)
        n_pca_components = max(3, pc_dims[0] + 1, pc_dims[1] + 1)
        n_pca_components = min(n_pca_components, filtered_h_global_pca.shape[1], filtered_h_global_pca.shape[0])
        
        print(f"Computing subspace PCA with {n_pca_components} components...")
        pca_subspace = PCA(n_components=n_pca_components)
        h_pca = pca_subspace.fit_transform(filtered_h_global_pca)
        print(f"Subspace PCA explained variance: {pca_subspace.explained_variance_ratio_}")
        
        # Plot using the exact same logic as residency_plot
        print("Creating plot...")
        plot_subspace_residency(
            h_pca, filtered_avg_steps, filtered_avg_sign_changes, 
            filtered_num_unstable, fc, filtered_h, output_set, model_path, pc_dims=pc_dims
        )
    
    print("\n" + "=" * 60)
    print("Subspace residency analysis completed!")
    print("=" * 60)











def load_model_for_subspace(model_path, input_size=100, hidden_size=150, output_size=3):
    """Load RNN model and extract weights."""
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=output_size)
    rnn.load_state_dict(torch.load(model_path))
    rnn.eval()
    return rnn

def generate_hidden_states_with_outputs(rnn, time_steps=100000, sigma=1, t0=20):
    """Generate hidden states and outputs by running the model."""
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc = rnn.fc.weight.data
    device = ih.device
    
    # Warm-up
    h = torch.zeros(rnn.hidden_size).to(device)
    for _ in range(t0):
        x = torch.normal(mean=0, std=sigma, size=(rnn.input_size,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
    
    # Generate sequence
    h_sequence = []
    outputs = []
    for _ in range(time_steps):
        x = torch.normal(mean=0, std=sigma, size=(rnn.input_size,)).float().to(device)
        h = torch.relu(x @ ih.T + h @ hh.T)
        h_sequence.append(h)
        
        # Compute output
        logits = h @ fc.T
        output = torch.argmax(logits).item()
        outputs.append(output)
    
    h_sequence = torch.stack(h_sequence)
    outputs = np.array(outputs)
    
    return h_sequence, outputs, ih, hh, fc, device

def filter_sequences_by_output(h_sequence, outputs, output_set, min_length=200):
    """Filter continuous sequences by output combination."""
    mask = np.isin(outputs, output_set)
    
    # Detect continuous sequences
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    long_sequences = [(s, e) for s, e in zip(starts, ends) if (e - s) > min_length]
    
    if not long_sequences:
        return None, None, None
    
    # Extract all indices belonging to long sequences
    filtered_indices = []
    for start, end in long_sequences:
        filtered_indices.extend(range(start, end))
    filtered_indices = np.array(filtered_indices)
    
    # Filter hidden states and outputs
    filtered_h = h_sequence[filtered_indices]
    filtered_outputs = outputs[filtered_indices]
    
    return filtered_h, filtered_outputs, filtered_indices

def compute_residency_metrics(h, ih, hh, fc, sigma, max_steps, n_samples, device):
    """Compute residency time and sign changes for a single hidden state."""
    logits = h @ fc.T
    probs = torch.softmax(logits, dim=0)
    most_probable = torch.argmax(probs).item()
    initial_logit = logits[most_probable].item()
    
    steps_list = []
    sign_changes_list = []
    
    for _ in range(n_samples):
        h_current = h.clone()
        prev_logit = initial_logit
        sign_changes = 0
        prev_delta_sign = None
        steps = 0
        while steps < max_steps:
            x = torch.normal(mean=0, std=sigma, size=(ih.shape[1],)).float().to(device)
            h_current = torch.relu(x @ ih.T + h_current @ hh.T)
            new_logits = h_current @ fc.T
            new_probs = torch.softmax(new_logits, dim=0)
            new_most_probable = torch.argmax(new_probs).item()
            if new_most_probable != most_probable:
                steps_list.append(steps + 1)
                sign_changes_list.append(sign_changes)
                break
            current_logit = new_logits[most_probable].item()
            delta_logit = current_logit - prev_logit
            current_sign = 1 if delta_logit > 0 else (-1 if delta_logit < 0 else 0)
            if prev_delta_sign is not None and current_sign != prev_delta_sign and current_sign != 0:
                sign_changes += 1
            prev_delta_sign = current_sign
            prev_logit = current_logit
            steps += 1
        else:
            steps_list.append(max_steps)
            sign_changes_list.append(sign_changes)
    
    avg_steps = np.mean(steps_list)
    avg_sign_changes = np.mean(sign_changes_list)
    return avg_steps, avg_sign_changes

def compute_unstable_eigenvalues(h, ih, hh, device):
    """Compute number of unstable eigenvalues after Mobius transform."""
    x = torch.normal(mean=0, std=0, size=(ih.shape[1],)).float().to(device)
    pre_act = x @ ih.T + h @ hh.T
    D = torch.diag((pre_act > 0).float())
    J = D @ hh
    eigvals = torch.linalg.eigvals(J)
    mobius = (eigvals - 1) / (eigvals + 1 + 1e-8)
    num_unstable = (mobius.real > 0).sum().item()
    return num_unstable

def subspace_residency_plotv2(model_path, 
                            output_combinations=None,
                            sigma=1, 
                            max_steps=50, 
                            n_samples=20, 
                            t0=20, 
                            time_steps=100000,
                            min_sequence_length=200,
                            n_components_global=4,
                            input_size=100,
                            hidden_size=150,
                            output_size=3,
                            pc_dims=(0, 2)):
    """
    Generate residency plots for each output subspace.
    This version: Filter first, then compute metrics.
    
    Args:
        model_path: Path to the saved model
        output_combinations: List of output combinations (e.g., [[0,1], [1,2], [0,2]])
        sigma: Standard deviation of input noise
        max_steps: Maximum steps for residency computation
        n_samples: Number of samples for residency computation
        t0: Warm-up steps
        time_steps: Total timesteps to generate
        min_sequence_length: Minimum length of continuous sequences
        n_components_global: Number of components for global PCA
        input_size: Input dimension
        hidden_size: Hidden state dimension
        output_size: Output dimension
        pc_dims: Tuple of PC indices to plot (default: (0, 2) for PC1 vs PC3)
    """
    if output_combinations is None:
        output_combinations = [[0, 1], [1, 2], [0, 2]]
    
    print("=" * 60)
    print("Starting subspace residency analysis")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from: {model_path}")
    rnn = load_model_for_subspace(model_path, input_size, hidden_size, output_size)
    
    # Generate hidden states and outputs
    print(f"Generating {time_steps} timesteps...")
    h_sequence, outputs, ih, hh, fc, device = generate_hidden_states_with_outputs(
        rnn, time_steps=time_steps, sigma=sigma, t0=t0
    )
    print(f"Generated {len(h_sequence)} hidden states")
    print(f"Output distribution: {np.bincount(outputs)}")
    
    # Compute global PCA
    print(f"Computing global PCA with {n_components_global} components...")
    h_np = h_sequence.cpu().numpy()
    pca_global = PCA(n_components=n_components_global)
    h_pca_global = pca_global.fit_transform(h_np)
    print(f"Global PCA explained variance: {pca_global.explained_variance_ratio_}")
    
    # Extract PC indices
    pc_x, pc_y = pc_dims
    
    # Process each output combination
    for output_set in output_combinations:
        print("\n" + "=" * 60)
        print(f"Processing subspace for outputs {output_set}")
        print("=" * 60)
        
        # Filter sequences
        filtered_h, filtered_outputs, filtered_indices = filter_sequences_by_output(
            h_sequence, outputs, output_set, min_length=min_sequence_length
        )
        
        if filtered_h is None:
            print(f"No sequences found for outputs {output_set}. Skipping.")
            continue
        
        T = len(filtered_h)
        print(f"Filtered to {T} states in {len(output_set)}-output subspace")
        
        # Apply subspace-specific PCA (need enough components for the requested dims)
        print("Computing subspace PCA...")
        filtered_h_global_pca = h_pca_global[filtered_indices]
        
        # Determine number of PCA components needed
        n_pca_components = max(3, pc_x + 1, pc_y + 1)
        n_pca_components = min(n_pca_components, filtered_h_global_pca.shape[1], filtered_h_global_pca.shape[0])
        
        pca_subspace = PCA(n_components=n_pca_components)
        h_pca = pca_subspace.fit_transform(filtered_h_global_pca)
        
        print(f"Subspace PCA explained variance: {pca_subspace.explained_variance_ratio_}")
        
        # Compute residency metrics (AFTER filtering)
        print("Computing residency metrics...")
        avg_steps_list = []
        avg_sign_changes_list = []
        num_unstable_list = []
        
        for i, h in enumerate(tqdm(filtered_h, desc="Computing metrics")):
            avg_steps, avg_sign_changes = compute_residency_metrics(
                h, ih, hh, fc, sigma, max_steps, n_samples, device
            )
            avg_steps_list.append(avg_steps)
            avg_sign_changes_list.append(avg_sign_changes)
            
            num_unstable = compute_unstable_eigenvalues(h, ih, hh, device)
            num_unstable_list.append(num_unstable)
        
        avg_steps = np.array(avg_steps_list)
        avg_sign_changes = np.array(avg_sign_changes_list)
        
        # Segment into groups
        cutoffcol = 2
        logits = torch.softmax(filtered_h @ fc.T, dim=1)
        group1 = avg_steps < 2
        group2 = (avg_steps >= cutoffcol) & (avg_steps <= 4*cutoffcol)
        group3 = (avg_steps > 4*cutoffcol) & (torch.max(logits, dim=1)[0].cpu().numpy() > 0.8)
        
        # Compute average unstable eigenvalues per group
        avg_num_unstable_group1 = np.mean([num_unstable_list[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
        avg_num_unstable_group2 = np.mean([num_unstable_list[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
        avg_num_unstable_group3 = np.mean([num_unstable_list[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
        avg_residency_group1 = np.mean([avg_steps_list[i] for i in range(T) if group1[i]]) if np.any(group1) else 0
        avg_residency_group2 = np.mean([avg_steps_list[i] for i in range(T) if group2[i]]) if np.any(group2) else 0
        avg_residency_group3 = np.mean([avg_steps_list[i] for i in range(T) if group3[i]]) if np.any(group3) else 0
        
        # Create plot
        print("Creating plot...")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
        
        # Set up colormap
        cmap_base = matplotlib.colormaps['inferno']
        color_map = cmap_base(np.linspace(0, 1, 15))
        colors = [
            color_map[10],
            color_map[9],
            np.array([0.6, 0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.5, 1.0]),
        ]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)], cmap.N)
        
        # Extract the specified PC dimensions
        x_data = h_pca[:, pc_x]
        y_data = h_pca[:, pc_y]
        
        # Panel 1: PCA colored by residency time
        sc1 = ax1.scatter(x_data, y_data, c=avg_steps, cmap=cmap, norm=norm, s=2, alpha=0.9)
        ax1.set_xlabel(f'PC{pc_x+1}', fontsize=14)
        ax1.set_ylabel(f'PC{pc_y+1}', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Panel 2: Histogram of sign changes
        if max(avg_sign_changes) > 0:
            counts, bin_edges = np.histogram(avg_sign_changes, bins=int(max(avg_sign_changes)))
            bin_avg_steps = []
            for i in range(len(bin_edges) - 1):
                mask = (avg_sign_changes >= bin_edges[i]) & (avg_sign_changes < bin_edges[i+1])
                bin_avg_steps.append(np.mean(avg_steps[mask]) if np.any(mask) else 0)
            
            for i in range(len(counts)):
                norm_value = norm(bin_avg_steps[i])
                color = cmap(norm_value)
                ax2.bar((bin_edges[i] + bin_edges[i+1]) / 2, counts[i], 
                        width=bin_edges[i+1] - bin_edges[i], color=color, alpha=0.9)
        
        ax2.set_xlabel('Avg. Sign Changes in Logits Gradient', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)
        
        # Panel 3: Bar plot of unstable eigenvalues
        groups = ['RT < 2\n(Transition)', '2 <= RT <= 8\n(Kick-Zone)', 'RT > 8\n(Cluster)']
        avg_num_unstable = [avg_num_unstable_group1, avg_num_unstable_group2, avg_num_unstable_group3]
        avg_residency = [avg_residency_group1, avg_residency_group2, avg_residency_group3]
        bar_colors = [cmap(norm(avg_residency[i])) for i in range(len(groups))]
        ax3.bar(groups, avg_num_unstable, color=bar_colors, alpha=0.9)
        ax3.set_ylabel('Avg. # Eigenvalues Re > 0\n(after Mobius Transform)', fontsize=14)
        ax3.tick_params(axis='x', labelsize=13)
        
        # Panel 4: PCA colored by unstable eigenvalues
        colors2 = [
            np.array([0.5, 0.5, 0.5, 1.0]),
            np.array([0.6, 0.0, 0.0, 1.0]),
            color_map[10],
        ]
        cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm([0, 1, 2, 3], cmap2.N)
        
        sc2 = ax4.scatter(x_data, y_data, c=num_unstable_list, cmap=cmap2, norm=norm2, s=2, alpha=0.9)
        ax4.set_xlabel(f'PC{pc_x+1}', fontsize=14)
        ax4.set_ylabel(f'PC{pc_y+1}', fontsize=14)
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        # Adjust layout and add colorbars
        plt.subplots_adjust(bottom=0.25, wspace=0.3)
        
        # Colorbar 1
        cbar1 = fig.colorbar(sc1, ax=[ax1, ax2, ax3], orientation='horizontal', pad=0.15, aspect=60)
        cbar1.set_ticks([1, cutoffcol, 2*cutoffcol, 4*cutoffcol+1, np.max(avg_steps)])
        cbar1.ax.set_xticklabels(['1', str(cutoffcol), str(2*cutoffcol), str(4*cutoffcol+1), f'{np.max(avg_steps).astype(int)}'], fontsize=14)
        cbar1.set_label('Residency Time (RT)', fontsize=14)
        
        # Colorbar 2
        cbar2 = fig.colorbar(sc2, ax=ax4, orientation='horizontal', pad=0.15, aspect=20)
        cbar2.set_label('# Eigenvalues Re > 0', fontsize=14)
        cbar2.set_ticks([0.5, 1.5, 2.5])
        cbar2.ax.set_xticklabels(['0', '1', '2'], fontsize=14)
        
        # Add title with PC information
        subspace_name = ''.join(map(str, output_set))
        fig.suptitle(f'Residency Analysis - Outputs {" & ".join(map(str, output_set))} Subspace (PC{pc_x+1} vs PC{pc_y+1})', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save with PC information in filename
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('scripts/plotting/plots/residency_subspace', exist_ok=True)
        save_path = f'scripts/plotting/plots/residency_subspace/residency_subspace_{subspace_name}_PC{pc_x+1}_PC{pc_y+1}_{datetime_str}.svg'
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        plt.show()
        plt.close()
    
    print("\n" + "=" * 60)
    print("Subspace residency analysis completed!")
    print("=" * 60)
