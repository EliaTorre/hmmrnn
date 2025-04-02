
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from scripts.rnn import RNN
from scipy import linalg
import itertools
from tqdm import tqdm

def load_model(model_path, input_size=100, hidden_size=150, num_layers=1, output_size=3, biased=[False, False]):
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, biased=biased)
    rnn.load_model(model_path)
    return rnn

def generate_sequences(rnn, time_steps=30000):
    rnn_data = rnn.gen_seq(time_steps=time_steps, dynamics_mode="full")
    return rnn_data

def create_bounded_grid(pca_result, n_points=15, pc1=None, pc2=None):
    mins = np.min(pca_result[:, :2], axis=0)
    maxs = np.max(pca_result[:, :2], axis=0)
    if pc1 is not None and len(pc1) == 2:
        mins[0] = pc1[0]
        maxs[0] = pc1[1]
    if pc2 is not None and len(pc2) == 2:
        mins[1] = pc2[0]
        maxs[1] = pc2[1]
    x = np.linspace(mins[0], maxs[0], n_points)
    y = np.linspace(mins[1], maxs[1], n_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    return grid_points

def perform_pca(hidden_states, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(hidden_states)
    return pca, pca_result

def map_to_original_space(grid_points, pca):
    original_points = pca.inverse_transform(grid_points)
    return original_points

def project_to_pca(next_steps, pca):
    pca_next_steps = pca.transform(next_steps)
    return pca_next_steps

def project_directions_to_pca(directions, pca):
    projected_directions = np.zeros((2, directions.shape[1]))
    for i in range(directions.shape[1]):
        for j in range(2):
            projected_directions[j, i] = np.dot(directions[:, i], pca.components_[j])
    return projected_directions

def project_point_to_svd_space(points, svd_directions):
    projected_points = np.zeros((points.shape[0], 2))
    for i in range(points.shape[0]):
        for j in range(2):
            projected_points[i, j] = np.dot(points[i], svd_directions[:, j])
    return projected_points

def project_data_to_svd_space(data, svd_directions):
    projected_data = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        for j in range(2):
            projected_data[i, j] = np.dot(data[i], svd_directions[:, j])
    return projected_data

def map_from_svd_to_original(grid_points, svd_directions):
    original_points = np.zeros((grid_points.shape[0], svd_directions.shape[0]))
    for i in range(grid_points.shape[0]):
        for j in range(2):
            original_points[i] += grid_points[i, j] * svd_directions[:, j]
    return original_points

def find_fixed_points(rnn, original_points, pca):
    device = rnn.device
    hh = rnn.rnn.weight_hh_l0.data
    print(original_points.shape)
    h = torch.tensor(original_points, dtype=torch.float32).to(device)
    for _ in range(1000):
        h = torch.relu(h @ hh.T)
    fixed_points_original = h.cpu().detach().numpy()
    fixed_points_pca = pca.transform(fixed_points_original)
    return fixed_points_pca

def compute_both_dynamics(rnn, original_points, use_relu=True):
    device = rnn.device
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    h = torch.tensor(original_points, dtype=torch.float32).to(device)

    if use_relu:
        h_recurrence = torch.relu(h @ hh.T)
    else:
        h_recurrence = h @ hh.T

    identity_matrix = torch.eye(h.shape[1], device=device)
    h_input = torch.zeros_like(h)
    for _ in range(100):
        x = torch.normal(mean=0, std=1, size=(h.shape[0], rnn.input_size)).float().to(device)
        if use_relu:
            h_input += torch.relu(x @ ih.T + h @ identity_matrix)
        else:
            h_input += x @ ih.T + h @ identity_matrix
    h_input /= 100
    
    h_full = torch.zeros_like(h)
    for _ in range(10):
        x = torch.normal(mean=0, std=1, size=(h.shape[0], rnn.input_size)).float().to(device)
        if use_relu:
            h_full += torch.relu(x @ ih.T + h @ hh.T)
        else:
            h_full += x @ ih.T + h @ hh.T
    h_full /= 10 
    
    return (h_recurrence.cpu().detach().numpy(), h_input.cpu().detach().numpy(), h_full.cpu().detach().numpy())

def calculate_vector_alignment(original_points, recurrence_steps, input_steps, method="cosine"):
    recurrence_vectors = recurrence_steps - original_points
    input_vectors = input_steps - original_points

    if method == "dot":
        alignment = np.sum(recurrence_vectors * input_vectors, axis=1)
    elif method == "cosine":
        recurrence_norm = np.linalg.norm(recurrence_vectors, axis=1)
        input_norm = np.linalg.norm(input_vectors, axis=1)
        valid_indices = (recurrence_norm > 0) & (input_norm > 0)
        alignment = np.zeros(len(original_points))
        alignment[valid_indices] = np.sum(recurrence_vectors[valid_indices] * input_vectors[valid_indices], axis=1) / (recurrence_norm[valid_indices] * input_norm[valid_indices])
    else:
        raise ValueError("Method must be either 'dot' or 'cosine'")
    return alignment

def analyze_weight_matrices_svd(rnn):
    ih = rnn.rnn.weight_ih_l0.data.cpu().numpy()
    hh = rnn.rnn.weight_hh_l0.data.cpu().numpy()

    U_ih, S_ih, Vt_ih = np.linalg.svd(ih, full_matrices=False)
    U_hh, S_hh, Vt_hh = np.linalg.svd(hh, full_matrices=False)
    
    ih_directions = U_ih[:, :5]
    hh_directions = U_hh[:, :5]
    
    return ih_directions, hh_directions, S_ih, S_hh

def visualize_flow_field(model_path, use_relu=True, alignment_method="cosine", input_size=100, hidden_size=150, 
                         num_layers=1, output_size=3, biased=[False, False], pc1=None, pc2=None, color_by="alignment"):
    rnn = load_model(model_path, input_size, hidden_size, num_layers, output_size, biased)
    rnn_data = generate_sequences(rnn)
    pca, pca_result = perform_pca(rnn_data["h"])
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    grid_points = create_bounded_grid(pca_result, pc1=pc1, pc2=pc2)
    original_points = map_to_original_space(grid_points, pca)
    recurrence_steps, input_steps, full_steps = compute_both_dynamics(rnn, original_points, use_relu)
    alignment = calculate_vector_alignment(original_points, recurrence_steps, input_steps, alignment_method)
    fixed_points = find_fixed_points(rnn, original_points, pca)
    pca_recurrence = project_to_pca(recurrence_steps, pca)
    pca_input = project_to_pca(input_steps, pca)
    pca_full = project_to_pca(full_steps, pca)
    return plot_flow_field_2d(grid_points, pca_recurrence, pca_input, pca_full, fixed_points, alignment, pc1, pc2, color_by)

def visualize_flow_field_svd_dir(model_path, use_relu=True, alignment_method="cosine", input_size=100, hidden_size=150, 
                                num_layers=1, output_size=3, biased=[False, False], pc1=None, pc2=None, color_by="alignment"):
    rnn = load_model(model_path, input_size, hidden_size, num_layers, output_size, biased)
    ih_directions, hh_directions, ih_values, hh_values = analyze_weight_matrices_svd(rnn)
    rnn_data = generate_sequences(rnn)
    pca, pca_result = perform_pca(rnn_data["h"])
    pca_ih_directions = project_directions_to_pca(ih_directions, pca)
    pca_hh_directions = project_directions_to_pca(hh_directions, pca)
    grid_points = create_bounded_grid(pca_result, pc1=pc1, pc2=pc2)
    original_points = map_to_original_space(grid_points, pca)
    recurrence_steps, input_steps, full_steps = compute_both_dynamics(rnn, original_points, use_relu)
    alignment = calculate_vector_alignment(original_points, recurrence_steps, input_steps, alignment_method)
    fixed_points = find_fixed_points(rnn, original_points, pca)
    pca_recurrence = project_to_pca(recurrence_steps, pca)
    pca_input = project_to_pca(input_steps, pca)
    pca_full = project_to_pca(full_steps, pca)
    return plot_flow_field_with_svd_directions(grid_points, pca_recurrence, pca_input, pca_full, fixed_points, alignment, 
                                              pca_ih_directions, pca_hh_directions, ih_values, hh_values, pc1, pc2, color_by)

def visualize_flow_field_svd_dim(model_path, use_relu=True, alignment_method="cosine", input_size=100, hidden_size=150, 
                                num_layers=1, output_size=3, biased=[False, False], pc1=None, pc2=None, color_by="alignment"):
    rnn = load_model(model_path, input_size, hidden_size, num_layers, output_size, biased)
    ih_directions, hh_directions, ih_values, hh_values = analyze_weight_matrices_svd(rnn)
    rnn_data = generate_sequences(rnn)
    pca, pca_result = perform_pca(rnn_data["h"])
    hh_data = project_data_to_svd_space(rnn_data["h"], hh_directions)
    ih_data = project_data_to_svd_space(rnn_data["h"], ih_directions)
    grid_hh = create_bounded_grid(hh_data, pc1=pc1, pc2=pc2)
    grid_ih = create_bounded_grid(ih_data, pc1=pc1, pc2=pc2)
    grid_pca = create_bounded_grid(pca_result, pc1=pc1, pc2=pc2)
    original_points_hh = map_from_svd_to_original(grid_hh, hh_directions)
    original_points_ih = map_from_svd_to_original(grid_ih, ih_directions)
    original_points_pca = map_to_original_space(grid_pca, pca)
    hh_recurrence, ih_input_for_hh, _ = compute_both_dynamics(rnn, original_points_hh, use_relu)
    ih_recurrence, ih_input, _ = compute_both_dynamics(rnn, original_points_ih, use_relu)
    pca_recurrence, pca_input, pca_full = compute_both_dynamics(rnn, original_points_pca, use_relu)
    next_points_hh = project_point_to_svd_space(hh_recurrence, hh_directions)
    next_points_ih = project_point_to_svd_space(ih_input, ih_directions)
    next_points_pca = project_to_pca(pca_full, pca)
    alignment_hh = calculate_vector_alignment(original_points_hh, hh_recurrence, ih_input_for_hh, alignment_method)
    fixed_original = find_fixed_points(rnn, original_points_pca, pca)
    fixed_original_space = map_to_original_space(fixed_original, pca)
    fixed_points_hh = project_point_to_svd_space(fixed_original_space, hh_directions)
    fixed_points_ih = project_point_to_svd_space(fixed_original_space, ih_directions)
    return plot_flow_field_svd_dim(grid_hh, grid_ih, grid_pca, next_points_hh, next_points_ih, next_points_pca, 
                                  fixed_points_hh, fixed_points_ih, fixed_original, alignment_hh, hh_values, ih_values, pc1, pc2, color_by)

def visualize_singular_vector_flow_fields(model_path, use_relu=True, input_size=100, hidden_size=150, num_layers=1, output_size=3, biased=[False, False], pc1=None, pc2=None):
    rnn = load_model(model_path, input_size, hidden_size, num_layers, output_size, biased)
    rnn_data = generate_sequences(rnn)
    pca, pca_result = perform_pca(rnn_data["h"])
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    grid_points = create_bounded_grid(pca_result, n_points=15, pc1=pc1, pc2=pc2)
    original_grid_points = map_to_original_space(grid_points, pca)
    ih = rnn.rnn.weight_ih_l0.data.cpu().numpy()
    hh = rnn.rnn.weight_hh_l0.data.cpu().numpy()
    U_ih, S_ih, _ = np.linalg.svd(ih, full_matrices=False)
    U_hh, S_hh, _ = np.linalg.svd(hh, full_matrices=False)
    U_ih_top5 = U_ih[:, :5]
    U_hh_top5 = U_hh[:, :5]
    S_ih_top5 = S_ih[:5]
    S_hh_top5 = S_hh[:5]
    n_grid_points = original_grid_points.shape[0]
    ih_perturbed = np.zeros((5, n_grid_points, hidden_size))
    hh_perturbed = np.zeros((5, n_grid_points, hidden_size))

    for i in range(5):
        for j in range(n_grid_points):
            if use_relu:
                ih_perturbed[i, j] = relu(original_grid_points[j] + U_ih_top5[:, i])
                hh_perturbed[i, j] = relu(original_grid_points[j] + U_hh_top5[:, i])
            else:
                ih_perturbed[i, j] = original_grid_points[j] + U_ih_top5[:, i]
                hh_perturbed[i, j] = original_grid_points[j] + U_hh_top5[:, i]

    ih_perturbed_pca = np.zeros((5, n_grid_points, 2))
    hh_perturbed_pca = np.zeros((5, n_grid_points, 2))
    
    for i in range(5):
        ih_perturbed_pca[i] = pca.transform(ih_perturbed[i])
        hh_perturbed_pca[i] = pca.transform(hh_perturbed[i])

    return plot_singular_vector_flow_fields(grid_points, ih_perturbed_pca, hh_perturbed_pca, S_ih_top5, S_hh_top5, pc1, pc2)

def plot_singular_vector_flow_fields(grid_points, ih_perturbed_pca, hh_perturbed_pca, S_ih_top5, S_hh_top5, pc1=None, pc2=None):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for i in range(5):
        x_min = min(x_min, hh_perturbed_pca[i, :, 0].min(), ih_perturbed_pca[i, :, 0].min())
        x_max = max(x_max, hh_perturbed_pca[i, :, 0].max(), ih_perturbed_pca[i, :, 0].max())
        y_min = min(y_min, hh_perturbed_pca[i, :, 1].min(), ih_perturbed_pca[i, :, 1].min())
        y_max = max(y_max, hh_perturbed_pca[i, :, 1].max(), ih_perturbed_pca[i, :, 1].max())
    if pc1 is not None and len(pc1) == 2:
        x_min, x_max = pc1[0], pc1[1]
    if pc2 is not None and len(pc2) == 2:
        y_min, y_max = pc2[0], pc2[1]
    
    all_magnitudes = []
    for i in range(5):
        U_hh = hh_perturbed_pca[i, :, 0] - grid_points[:, 0]
        V_hh = hh_perturbed_pca[i, :, 1] - grid_points[:, 1]
        U_ih = ih_perturbed_pca[i, :, 0] - grid_points[:, 0]
        V_ih = ih_perturbed_pca[i, :, 1] - grid_points[:, 1]
        all_magnitudes.extend(np.sqrt(U_hh**2 + V_hh**2))
        all_magnitudes.extend(np.sqrt(U_ih**2 + V_ih**2)) 
    vmin, vmax = min(all_magnitudes), max(all_magnitudes)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(5):
        ax = axes[0, i]
        X = grid_points[:, 0] 
        Y = grid_points[:, 1]
        U = hh_perturbed_pca[i, :, 0] - grid_points[:, 0]
        V = hh_perturbed_pca[i, :, 1] - grid_points[:, 1]
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitude + 1e-8)
        V_norm = V / (magnitude + 1e-8)
        q = ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', norm=norm)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"HH-SV {i+1} (σ={S_hh_top5[i]:.2f})", fontsize=12)
        if i == 0:
            ax.set_ylabel("PC2", fontsize=12)
        ax.grid(True)
    
    for i in range(5):
        ax = axes[1, i]
        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        U = ih_perturbed_pca[i, :, 0] - grid_points[:, 0]
        V = ih_perturbed_pca[i, :, 1] - grid_points[:, 1]
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitude + 1e-8)
        V_norm = V / (magnitude + 1e-8)
        q = ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', norm=norm)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"IH-SV {i+1} (σ={S_ih_top5[i]:.2f})", fontsize=12)
        ax.set_xlabel("PC1", fontsize=12)
    
        if i == 0:
            ax.set_ylabel("PC2", fontsize=12)  
        ax.grid(True)

    cbar = fig.colorbar(q, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label("Vector Magnitude")
    fig.suptitle("Impact of SVD Singular Vectors on RNN Dynamics in PCA Space", fontsize=16)
    
    return fig, axes

def plot_flow_field_2d(grid_points, pca_recurrence, pca_input, pca_full, fixed_points, alignment, pc1=None, pc2=None, color_by="alignment"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    if pc1 is not None and len(pc1) == 2:
        x_min, x_max = pc1[0], pc1[1]
    else:
        x_min, x_max = grid_points[:, 0].min(), grid_points[:, 0].max()
    if pc2 is not None and len(pc2) == 2:
        y_min, y_max = pc2[0], pc2[1]
    else:
        y_min, y_max = grid_points[:, 1].min(), grid_points[:, 1].max()
        
    # Calculate vector magnitudes if needed
    if color_by == "magnitude":
        recurrence_magnitude = np.sqrt(np.sum((pca_recurrence - grid_points)**2, axis=1))
        input_magnitude = np.sqrt(np.sum((pca_input - grid_points)**2, axis=1))
        full_magnitude = np.sqrt(np.sum((pca_full - grid_points)**2, axis=1))
        magnitudes = [recurrence_magnitude, input_magnitude, full_magnitude]
        # Use viridis colormap for magnitude and set appropriate min/max
        cmap = 'viridis'
        vmin = min(np.min(recurrence_magnitude), np.min(input_magnitude), np.min(full_magnitude))
        vmax = max(np.max(recurrence_magnitude), np.max(input_magnitude), np.max(full_magnitude))
        color_label = 'Vector Magnitude'
    else:  # color_by == "alignment"
        # Use original alignment values
        magnitudes = [alignment, alignment, alignment]
        cmap = 'Reds'
        vmin = np.min(alignment)
        vmax = np.max(alignment)
        color_label = 'Vector Alignment'
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    dynamics_data = [("Recurrence Only", pca_recurrence, axes[0]), 
                    ("Input Only", pca_input, axes[1]), 
                    ("Full Dynamics", pca_full, axes[2])]
                    
    for i, (title, pca_next_steps, ax) in enumerate(dynamics_data):
        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        U = pca_next_steps[:, 0] - grid_points[:, 0]
        V = pca_next_steps[:, 1] - grid_points[:, 1]
        norm_vec = np.sqrt(U**2 + V**2)
        U = U / (norm_vec + 1e-8)
        V = V / (norm_vec + 1e-8)
        q = ax.quiver(X, Y, U, V, magnitudes[i], cmap=cmap, norm=norm)
        ax.scatter(fixed_points[:, 0], fixed_points[:, 1], marker='x', color='black', s=100, linewidth=2, label='Fixed Point')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.grid(True)
    
    cbar = fig.colorbar(q, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label(color_label)
    fig.suptitle(f"Comparison of RNN Dynamics in PCA Space (Colored by {color_label})", fontsize=16)
    return fig, axes

def plot_flow_field_with_svd_directions(grid_points, pca_recurrence, pca_input, pca_full, fixed_points, alignment, 
                                      pca_ih_directions, pca_hh_directions, ih_values, hh_values, pc1=None, pc2=None, color_by="alignment"):
    """
    Plot the flow field with SVD directions.
    
    Parameters similar to plot_flow_field_2d with additional SVD direction parameters
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    if pc1 is not None and len(pc1) == 2:
        x_min, x_max = pc1[0], pc1[1]
    else:
        x_min, x_max = grid_points[:, 0].min(), grid_points[:, 0].max()
    
    if pc2 is not None and len(pc2) == 2:
        y_min, y_max = pc2[0], pc2[1]
    else:
        y_min, y_max = grid_points[:, 1].min(), grid_points[:, 1].max()
    
    # Calculate vector magnitudes if needed
    if color_by == "magnitude":
        recurrence_magnitude = np.sqrt(np.sum((pca_recurrence - grid_points)**2, axis=1))
        input_magnitude = np.sqrt(np.sum((pca_input - grid_points)**2, axis=1))
        full_magnitude = np.sqrt(np.sum((pca_full - grid_points)**2, axis=1))
        magnitudes = [recurrence_magnitude, input_magnitude, full_magnitude]
        # Use viridis colormap for magnitude and set appropriate min/max
        cmap = 'viridis'
        vmin = min(np.min(recurrence_magnitude), np.min(input_magnitude), np.min(full_magnitude))
        vmax = max(np.max(recurrence_magnitude), np.max(input_magnitude), np.max(full_magnitude))
        color_label = 'Vector Magnitude'
    else:  # color_by == "alignment"
        # Use original alignment values
        magnitudes = [alignment, alignment, alignment]
        cmap = 'Reds'
        vmin = np.min(alignment)
        vmax = np.max(alignment)
        color_label = 'Vector Alignment'
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    dynamics_data = [
        ("Recurrence Only", pca_recurrence, axes[0], pca_hh_directions, hh_values), 
        ("Input Only", pca_input, axes[1], pca_ih_directions, ih_values), 
        ("Full Dynamics", pca_full, axes[2], None, None)
    ]
    
    svd_colors = ['magenta', 'orange', 'green', 'purple', 'cyan']
    
    for i, (title, pca_next_steps, ax, svd_directions, sv_values) in enumerate(dynamics_data):
        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        U = pca_next_steps[:, 0] - grid_points[:, 0]
        V = pca_next_steps[:, 1] - grid_points[:, 1]
        norm_vec = np.sqrt(U**2 + V**2)
        U = U / (norm_vec + 1e-8)
        V = V / (norm_vec + 1e-8)
        q = ax.quiver(X, Y, U, V, magnitudes[i], cmap=cmap, norm=norm)
        ax.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                  marker='x', color='black', s=100, linewidth=2, label='Fixed Point')
        
        if svd_directions is not None:
            origin = np.zeros(2)
            for j in range(5):
                direction = svd_directions[:, j]
                scale_factor = min(x_max - x_min, y_max - y_min) / 4
                scaled_direction = direction * scale_factor
                ax.arrow(origin[0], origin[1], 
                        scaled_direction[0], scaled_direction[1],
                        color=svd_colors[j], width=0.02, length_includes_head=True,
                        head_width=0.1, head_length=0.1,
                        label=f'SVD Dir {j+1} (σ={sv_values[j]:.2f})')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=9)
    
    cbar = fig.colorbar(q, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label(color_label)
    fig.suptitle(f"RNN Dynamics with Principal SVD Directions (Colored by {color_label})", fontsize=16)
    return fig, axes

def plot_flow_field_svd_dim(grid_points_hh, grid_points_ih, grid_points_pca, next_points_hh, next_points_ih, next_points_pca, 
                          fixed_points_hh, fixed_points_ih, fixed_points_pca, alignment, hh_values, ih_values, pc1=None, pc2=None, color_by="alignment"):
    """
    Plot the flow field in SVD dimension space.
    
    Parameters similar to previous plotting functions
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Calculate vector magnitudes if needed
    if color_by == "magnitude":
        hh_magnitude = np.sqrt(np.sum((next_points_hh - grid_points_hh)**2, axis=1))
        ih_magnitude = np.sqrt(np.sum((next_points_ih - grid_points_ih)**2, axis=1))
        pca_magnitude = np.sqrt(np.sum((next_points_pca - grid_points_pca)**2, axis=1))
        magnitudes = [hh_magnitude, ih_magnitude, pca_magnitude]
        # Use viridis colormap for magnitude and set appropriate min/max
        cmap = 'viridis'
        vmin = min(np.min(hh_magnitude), np.min(ih_magnitude), np.min(pca_magnitude))
        vmax = max(np.max(hh_magnitude), np.max(ih_magnitude), np.max(pca_magnitude))
        color_label = 'Vector Magnitude'
    else:  # color_by == "alignment"
        # Use original alignment values for all plots for consistency
        magnitudes = [alignment, alignment, alignment]
        cmap = 'Reds'
        vmin = np.min(alignment)
        vmax = np.max(alignment)
        color_label = 'Vector Alignment'
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    plot_data = [
        ("Recurrence in HH Space", grid_points_hh, next_points_hh, fixed_points_hh, axes[0], hh_values), 
        ("Input in IH Space", grid_points_ih, next_points_ih, fixed_points_ih, axes[1], ih_values),
        ("Full in PCA Space", grid_points_pca, next_points_pca, fixed_points_pca, axes[2], None)
    ]
    
    for i, (title, grid, next_points, fixed, ax, sv_values) in enumerate(plot_data):
        if pc1 is not None and len(pc1) == 2:
            x_min, x_max = pc1[0], pc1[1]
        else:
            x_min, x_max = grid[:, 0].min(), grid[:, 0].max()
        
        if pc2 is not None and len(pc2) == 2:
            y_min, y_max = pc2[0], pc2[1]
        else:
            y_min, y_max = grid[:, 1].min(), grid[:, 1].max()
            
        X = grid[:, 0]
        Y = grid[:, 1]
        U = next_points[:, 0] - grid[:, 0]
        V = next_points[:, 1] - grid[:, 1]
        norm_vec = np.sqrt(U**2 + V**2)
        U = U / (norm_vec + 1e-8)
        V = V / (norm_vec + 1e-8)
        
        q = ax.quiver(X, Y, U, V, magnitudes[i], cmap=cmap, norm=norm)
        ax.scatter(fixed[:, 0], fixed[:, 1], marker='x', color='black', s=100, linewidth=2, label='Fixed Point')
        
        if "HH Space" in title:
            ax.set_xlabel("HH Dir 1 (σ={:.2f})".format(sv_values[0]), fontsize=12)
            ax.set_ylabel("HH Dir 2 (σ={:.2f})".format(sv_values[1]), fontsize=12)
        elif "IH Space" in title:
            ax.set_xlabel("IH Dir 1 (σ={:.2f})".format(sv_values[0]), fontsize=12)
            ax.set_ylabel("IH Dir 2 (σ={:.2f})".format(sv_values[1]), fontsize=12)
        else:
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=14)
        ax.grid(True)
        
    cbar = fig.colorbar(q, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label(color_label)
    fig.suptitle(f"RNN Dynamics in Different Projection Spaces (Colored by {color_label})", fontsize=16)
    return fig, axes

def relu(x):
    result = x.copy()
    result[result < 0] = 0
    return result

def rnn_var(model_path, n_states=1000, n_samples=1000, use_temp=True):
    rnn = RNN(input_size=100, hidden_size=150, num_layers=1, output_size=3)
    rnn.load_state_dict(torch.load(model_path))
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data

    h = torch.zeros((n_states, 150)).to(ih.device)
    for i in range(n_states):
        x = torch.normal(mean=0, std=1, size=(100,)).float().to(ih.device)
        h[i] = torch.relu(x @ ih.T + (h[i-1] @ hh.T if i > 0 else 0))
    
    types = {
        'ih': torch.zeros((n_states, n_samples, 150)).to(ih.device),
        'relu': torch.zeros((n_states, n_samples, 150)).to(ih.device),
        'hh': torch.zeros((n_states, n_samples, 150)).to(ih.device),
        'hh_relu': torch.zeros((n_states, n_samples, 150)).to(ih.device),
        'ih_relu': torch.zeros((n_states, n_samples, 150)).to(ih.device),
        'ih_relu_hh': torch.zeros((n_states, n_samples, 150)).to(ih.device)
    }
    for i in tqdm(range(n_states), desc="Computing noise samples"):
        for j in range(n_samples):
            x = torch.normal(mean=0, std=1, size=(100,)).float().to(ih.device)
            x_ih = x @ ih.T
            types['ih'][i, j] = x_ih
            types['hh'][i, j] = (h[i] @ hh.T) - h[i]
            types['relu'][i, j] = torch.relu(x_ih)
            types['ih_relu'][i, j] = torch.relu(x_ih + h[i]) - h[i]
            types['hh_relu'][i, j] = torch.relu(h[i] @ hh.T) - h[i]
            types['ih_relu_hh'][i, j] = torch.relu(x_ih + h[i] @ hh.T) - h[i]
    
    results = {}
    n_components = 150
    for step_type, step_tensor in types.items():
        exp_var = np.zeros((n_states, n_components))
        for i in tqdm(range(n_states), desc=f"PCA per state ({step_type})"):
            step_samples = step_tensor[i].cpu().numpy()
            if use_temp:
                temp = np.vstack((step_samples, -step_samples))
                pca = PCA(n_components=n_components)
                pca.fit(temp)
            else:
                pca = PCA(n_components=n_components)
                pca.fit(step_samples)
            exp_var[i] = pca.explained_variance_ratio_
        
        cum_variance = np.cumsum(exp_var, axis=1)
        avg_cum_variance = np.mean(cum_variance, axis=0)
        
        noise_stacked = step_tensor.reshape(n_states * n_samples, n_components).cpu().numpy()
        if use_temp:
            temp_stacked = np.vstack((noise_stacked, -noise_stacked))
            pca_stacked = PCA(n_components=n_components)
            pca_stacked.fit(temp_stacked)
        else:
            pca_stacked = PCA(n_components=n_components)
            pca_stacked.fit(noise_stacked)
        cum_variance_stacked = np.cumsum(pca_stacked.explained_variance_ratio_)
        
        results[step_type] = {
            'individual': cum_variance,
            'average': avg_cum_variance,
            'stacked': cum_variance_stacked
        }
    
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    titles = {
        'ih': 'x @ ih.T',
        'hh': 'h @ hh.T - h',
        'relu': 'ReLU(x @ ih.T)',
        'ih_relu': 'ReLU(x @ ih.T + h) - h',
        'hh_relu': 'ReLU(h @ hh.T) - h',
        'ih_relu_hh': 'ReLU(x @ ih.T + h @ hh.T) - h'
    }
    for i, (step_type, title) in enumerate(titles.items()):
        ax = axes[i]
        for j in range(n_states):
            ax.plot(range(1, 151), results[step_type]['individual'][j], alpha=0.05, color='gray')
        ax.plot(range(1, 151), results[step_type]['average'], color='red', label='Avg over states', linewidth=2)
        ax.plot(range(1, 151), results[step_type]['stacked'], color='blue', label='All concatenated', linewidth=2)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.7)
        if i == 0:
            ax.text(120, 0.96, '95%', fontsize=9)
            ax.text(120, 0.81, '80%', fontsize=9)
            ax.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
            ax.text(4, 0.05, '3 PC', fontsize=9)
        else:
            ax.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Principal Components')
        if i == 0:
            ax.set_ylabel('Explained Variance')
        ax.set_title(f'{title}')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1.05)
        if i == 3:
            ax.legend(loc='lower right', fontsize=9)
        ax.grid(False)
    plt.tight_layout()
    plt.show()

def noise_cosine_similarity(model_path, n_states=1000, n_samples=1000):
    rnn = RNN(input_size=100, hidden_size=150, num_layers=1, output_size=3)
    rnn.load_state_dict(torch.load(model_path))
    ih = rnn.rnn.weight_ih_l0.data 
    hh = rnn.rnn.weight_hh_l0.data 

    h = torch.zeros((n_states, 150)).to(ih.device)
    for i in tqdm(range(n_states), desc="Generating hidden states"):
        x = torch.normal(mean=0, std=1, size=(100,)).float().to(ih.device)
        h[i] = torch.relu(x @ ih.T + (h[i-1] @ hh.T if i > 0 else 0))

    noise_ih = np.zeros((n_states, n_samples, 150))
    noise_relu = np.zeros((n_states, n_samples, 150))
    noise_ih_relu = np.zeros((n_states, n_samples, 150))
    noise_ih_relu_hh = np.zeros((n_states, n_samples, 150))

    for i in tqdm(range(n_states), desc="Computing noise samples"):
        for j in range(n_samples):
            x = torch.normal(mean=0, std=1, size=(100,)).float().to(ih.device)
            noise_ih[i, j] = (x @ ih.T).cpu().detach().numpy()
            noise_relu[i, j] = torch.relu(x @ ih.T).cpu().detach().numpy()
            noise_ih_relu[i, j] = (torch.relu(x @ ih.T + h[i]) - h[i]).cpu().detach().numpy()
            noise_ih_relu_hh[i, j] = (torch.relu(x @ ih.T + h[i] @ hh.T) - h[i]).cpu().detach().numpy()
    
    noise_means = {
        'ih': np.mean(noise_ih, axis=1),
        'relu': np.mean(noise_relu, axis=1),
        'ih_relu': np.mean(noise_ih_relu, axis=1),
        'ih_relu_hh': np.mean(noise_ih_relu_hh, axis=1)
    }
    
    cos_sim = {
        'ih': np.zeros((n_states, n_samples)),
        'relu': np.zeros((n_states, n_samples)),
        'ih_relu': np.zeros((n_states, n_samples)),
        'ih_relu_hh': np.zeros((n_states, n_samples))
    }
    
    for mod in ['ih', 'relu', 'ih_relu', 'ih_relu_hh']:
        noise_data = noise_ih if mod == 'ih' else noise_relu if mod == 'relu' else noise_ih_relu if mod == 'ih_relu' else noise_ih_relu_hh
        for i in tqdm(range(n_states), desc=f"Computing cosine similarity ({mod})"):
            for j in range(n_samples):
                noise_vec = noise_data[i, j]
                mean_vec = noise_means[mod][i]
                cos_sim[mod][i, j] = np.dot(noise_vec, mean_vec) / (np.linalg.norm(noise_vec) * np.linalg.norm(mean_vec))
    
    cos_sim_mean = {mod: np.mean(cos_sim[mod], axis=1) for mod in cos_sim}
    cos_sim_std = {mod: np.std(cos_sim[mod], axis=1) for mod in cos_sim}
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = {
        'ih': 'x @ ih.T',
        'relu': 'ReLU(x @ ih.T)',
        'ih_relu': 'ReLU(x @ ih.T + h) - h',
        'ih_relu_hh': 'ReLU(x @ ih.T + h @ hh.T) - h'
    }
    
    for i, mod in enumerate(['ih', 'relu', 'ih_relu', 'ih_relu_hh']):
        ax = axs[i]
        ax.errorbar(np.arange(n_states), cos_sim_mean[mod], yerr=cos_sim_std[mod], 
                    fmt='o', markersize=2, color='darkgreen', ecolor='lightgreen', capsize=2)
        ax.text(0.85, 0.25, f'Mean: {np.mean(cos_sim_mean[mod]):.3f}\nStd: {np.mean(cos_sim_std[mod]):.3f}', 
                ha='center', va='bottom', transform=ax.transAxes)
        ax.set_title(titles[mod], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlabel('# of Random Initial Conditions', fontsize=10)
        if i == 0:
            ax.set_ylabel('Cosine Similarity', fontsize=10)
        if i > 0:
            ax.set_yticks([])
    
    plt.suptitle('Cosine Similarity between Random Noise and Mean at ICs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()