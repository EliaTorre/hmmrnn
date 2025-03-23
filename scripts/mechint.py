import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

# Import necessary modules from the existing codebase
from scripts.rnn import RNN

def load_model(model_path, input_size=100, hidden_size=150, num_layers=1, output_size=3, biased=[False, False]):
    """
    Load a trained RNN model from path.
    
    Args:
        model_path (str): Path to the model file
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden layer
        num_layers (int): Number of RNN layers
        output_size (int): Number of output classes
        biased (list): Whether to use bias in RNN and linear layers
        
    Returns:
        RNN: Loaded RNN model
    """
    rnn = RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        biased=biased
    )
    
    rnn.load_model(model_path)
    return rnn

def generate_sequences(rnn, time_steps=30000):
    """
    Generate sequences using the RNN model with full dynamics.
    
    Args:
        rnn (RNN): Trained RNN model
        time_steps (int): Number of time steps to generate
        
    Returns:
        dict: Dictionary containing hidden states, outputs, etc.
    """
    rnn_data = rnn.gen_seq(time_steps=time_steps, dynamics_mode="full")
    return rnn_data

def perform_pca(hidden_states, n_components=3):
    """
    Perform PCA on hidden states.
    
    Args:
        hidden_states (numpy.ndarray): Hidden states from RNN
        n_components (int): Number of principal components to extract
        
    Returns:
        tuple: (PCA object, PCA-transformed hidden states)
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(hidden_states)
    
    return pca, pca_result

def create_grid(pca_result, n_points=20, n_dims=2):
    """
    Create a grid in PCA space.
    
    Args:
        pca_result (numpy.ndarray): PCA-transformed hidden states
        n_points (int): Number of grid points in each dimension
        n_dims (int): Number of dimensions (2 or 3)
        
    Returns:
        numpy.ndarray: Grid points in PCA space
    """
    # Find min and max values for each dimension
    mins = np.min(pca_result[:, :n_dims], axis=0)
    maxs = np.max(pca_result[:, :n_dims], axis=0)
    
    # Create grid in PCA space
    if n_dims == 2:
        # Create a 2D grid
        x = np.linspace(mins[0], maxs[0], n_points)
        y = np.linspace(mins[1], maxs[1], n_points)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.flatten(), Y.flatten()])
    elif n_dims == 3:
        # For 3D, create three planar grids for better visualization
        x = np.linspace(mins[0], maxs[0], n_points)
        y = np.linspace(mins[1], maxs[1], n_points)
        z = np.linspace(mins[2], maxs[2], n_points)
        
        mean_x = np.mean([mins[0], maxs[0]])
        mean_y = np.mean([mins[1], maxs[1]])
        mean_z = np.mean([mins[2], maxs[2]])
        
        # Plane 1: z = mean_z (xy-plane)
        X1, Y1 = np.meshgrid(x, y)
        Z1 = np.full_like(X1, mean_z)
        
        # Plane 2: y = mean_y (xz-plane)
        X2, Z2 = np.meshgrid(x, z)
        Y2 = np.full_like(X2, mean_y)
        
        # Plane 3: x = mean_x (yz-plane)
        Y3, Z3 = np.meshgrid(y, z)
        X3 = np.full_like(Y3, mean_x)
        
        grid_points1 = np.column_stack([X1.flatten(), Y1.flatten(), Z1.flatten()])
        grid_points2 = np.column_stack([X2.flatten(), Y2.flatten(), Z2.flatten()])
        grid_points3 = np.column_stack([X3.flatten(), Y3.flatten(), Z3.flatten()])
        
        grid_points = np.vstack([grid_points1, grid_points2, grid_points3])
    
    return grid_points

def map_to_original_space(grid_points, pca):
    """
    Map grid points from PCA space back to original space.
    
    Args:
        grid_points (numpy.ndarray): Grid points in PCA space
        pca (PCA): PCA object used for transformation
        
    Returns:
        numpy.ndarray: Grid points in original space
    """
    # Add zeros for remaining components if needed
    if grid_points.shape[1] < pca.n_components_:
        zeros = np.zeros((grid_points.shape[0], pca.n_components_ - grid_points.shape[1]))
        grid_points_full = np.hstack((grid_points, zeros))
    else:
        grid_points_full = grid_points
        
    original_points = pca.inverse_transform(grid_points_full)
    return original_points

def find_fixed_points(rnn, original_points, pca):
    """
    Find fixed points by running recurrence-only dynamics for 300 steps.
    
    Args:
        rnn (RNN): Trained RNN model
        original_points (numpy.ndarray): Grid points in original space
        pca (PCA): PCA object used for transformation
        
    Returns:
        numpy.ndarray: Fixed points in PCA space
    """
    device = rnn.device
    
    # Extract recurrent weights
    hh = rnn.rnn.weight_hh_l0.data
    
    # Convert points to tensor
    h = torch.tensor(original_points, dtype=torch.float32).to(device)
    
    # Make a copy of initial points
    h_initial = h.clone()
    
    # Run 300 steps of recurrence-only dynamics with ReLU
    for _ in range(300):
        h = torch.relu(h @ hh.T)
    
    # Convert back to numpy array
    fixed_points_original = h.cpu().detach().numpy()
    
    # Project to PCA space
    fixed_points_pca = pca.transform(fixed_points_original)
    
    return fixed_points_pca

def compute_both_dynamics(rnn, original_points, use_relu=True):
    """
    Compute both recurrence-only and input-only dynamics at each grid point.
    
    Args:
        rnn (RNN): Trained RNN model
        original_points (numpy.ndarray): Grid points in original space
        use_relu (bool): Whether to use ReLU activation
        
    Returns:
        tuple: (recurrence_only_steps, input_only_steps, full_dynamics_steps)
    """
    device = rnn.device
    
    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    
    # Convert points to tensor
    h = torch.tensor(original_points, dtype=torch.float32).to(device)
    
    # Recurrence-only dynamics
    if use_relu:
        h_recurrence = torch.relu(h @ hh.T)
    else:
        h_recurrence = h @ hh.T
    
    # Input-only dynamics (average over 100 steps)
    identity_matrix = torch.eye(h.shape[1], device=device)
    h_input = torch.zeros_like(h)
    
    for _ in range(100):
        x = torch.normal(mean=0, std=1, size=(h.shape[0], rnn.input_size)).float().to(device)
        if use_relu:
            h_input += torch.relu(x @ ih.T + h @ identity_matrix)
        else:
            h_input += x @ ih.T + h @ identity_matrix
    
    h_input /= 100  # Average
    
    # Full dynamics (average over 100 steps)
    h_full = torch.zeros_like(h)
    
    for _ in range(100):
        x = torch.normal(mean=0, std=1, size=(h.shape[0], rnn.input_size)).float().to(device)
        if use_relu:
            h_full += torch.relu(x @ ih.T + h @ hh.T)
        else:
            h_full += x @ ih.T + h @ hh.T
    
    h_full /= 100  # Average
    
    return (
        h_recurrence.cpu().detach().numpy(),
        h_input.cpu().detach().numpy(),
        h_full.cpu().detach().numpy()
    )

def calculate_vector_alignment(original_points, recurrence_steps, input_steps, method="cosine"):
    """
    Calculate alignment between recurrence-only and input-only vectors.
    
    Args:
        original_points (numpy.ndarray): Original grid points
        recurrence_steps (numpy.ndarray): Grid points after recurrence-only step
        input_steps (numpy.ndarray): Grid points after input-only step
        method (str): Method to calculate alignment ('dot' or 'cosine')
        
    Returns:
        numpy.ndarray: Alignment values
    """
    # Calculate vectors
    recurrence_vectors = recurrence_steps - original_points
    input_vectors = input_steps - original_points
    
    # Calculate alignment
    if method == "dot":
        # Dot product
        alignment = np.sum(recurrence_vectors * input_vectors, axis=1)
    elif method == "cosine":
        # Cosine similarity
        recurrence_norm = np.linalg.norm(recurrence_vectors, axis=1)
        input_norm = np.linalg.norm(input_vectors, axis=1)
        
        # Avoid division by zero
        valid_indices = (recurrence_norm > 0) & (input_norm > 0)
        alignment = np.zeros(len(original_points))
        
        alignment[valid_indices] = np.sum(
            recurrence_vectors[valid_indices] * input_vectors[valid_indices], 
            axis=1
        ) / (recurrence_norm[valid_indices] * input_norm[valid_indices])
    else:
        raise ValueError("Method must be either 'dot' or 'cosine'")
    
    return alignment

def project_to_pca(next_steps, pca):
    """
    Project next steps from original space to PCA space.
    
    Args:
        next_steps (numpy.ndarray): Next steps in original space
        pca (PCA): PCA object used for transformation
        
    Returns:
        numpy.ndarray: Next steps in PCA space
    """
    pca_next_steps = pca.transform(next_steps)
    return pca_next_steps

def plot_flow_field_2d(grid_points, pca_next_steps, fixed_points, alignment, save_path=None):
    """
    Plot flow field in 2D using matplotlib with vector alignment coloring and fixed points.
    
    Args:
        grid_points (numpy.ndarray): Grid points in PCA space
        pca_next_steps (numpy.ndarray): Next steps in PCA space
        fixed_points (numpy.ndarray): Fixed points in PCA space
        alignment (numpy.ndarray): Vector alignment values
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Compute vector field
    X = grid_points[:, 0]
    Y = grid_points[:, 1]
    U = pca_next_steps[:, 0] - grid_points[:, 0]
    V = pca_next_steps[:, 1] - grid_points[:, 1]
    
    # Normalize vector lengths for better visualization
    norm = np.sqrt(U**2 + V**2)
    U = U / (norm + 1e-8)  # Avoid division by zero
    V = V / (norm + 1e-8)
    
    # Normalize alignment values for coloring
    alignment_min = np.min(alignment)
    alignment_max = np.max(alignment)
    norm = Normalize(vmin=alignment_min, vmax=alignment_max)
    
    # Plot vector field with color based on alignment
    q = ax.quiver(X, Y, U, V, alignment, cmap='Reds', norm=norm)
    fig.colorbar(q, ax=ax, label='Vector Alignment')
    
    # Add fixed points as black X markers
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], 
               marker='x', color='black', s=100, linewidth=2, label='Fixed Points')
    
    ax.set_title("2D Flow Field in PCA Space", fontsize=14)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.grid(True)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D flow field plot saved to {save_path}")
    
    return fig, ax

def plot_flow_field_3d(grid_points, pca_next_steps, fixed_points, alignment, save_path=None):
    """
    Plot flow field in 3D using Plotly with vector alignment coloring and fixed points.
    
    Args:
        grid_points (numpy.ndarray): Grid points in PCA space
        pca_next_steps (numpy.ndarray): Next steps in PCA space
        fixed_points (numpy.ndarray): Fixed points in PCA space
        alignment (numpy.ndarray): Vector alignment values
        save_path (str, optional): Path to save the plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Compute vector field
    X = grid_points[:, 0]
    Y = grid_points[:, 1]
    Z = grid_points[:, 2]
    U = pca_next_steps[:, 0] - grid_points[:, 0]
    V = pca_next_steps[:, 1] - grid_points[:, 1]
    W = pca_next_steps[:, 2] - grid_points[:, 2]
    
    # Normalize vector lengths for better visualization
    norm = np.sqrt(U**2 + V**2 + W**2)
    U = U / (norm + 1e-8)  # Avoid division by zero
    V = V / (norm + 1e-8)
    W = W / (norm + 1e-8)
    
    # Normalize alignment values for coloring
    alignment_min = np.min(alignment)
    alignment_max = np.max(alignment)
    
    # Add cones for vector field with color based on alignment
    fig.add_trace(go.Cone(
        x=X,
        y=Y,
        z=Z,
        u=U,
        v=V,
        w=W,
        colorscale='Reds',
        showscale=True,
        sizemode="absolute",
        sizeref=0.5,
        colorbar=dict(
            title='Vector Alignment'
        ),
        customdata=alignment,
        cmin=alignment_min,
        cmax=alignment_max
    ))
    
    # Add fixed points as black markers
    fig.add_trace(go.Scatter3d(
        x=fixed_points[:, 0],
        y=fixed_points[:, 1],
        z=fixed_points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            symbol='x',
            line=dict(
                width=2,
                color='black'
            )
        ),
        name='Fixed Points'
    ))
    
    # Add points to visualize the grid locations
    fig.add_trace(go.Scatter3d(
        x=X, 
        y=Y, 
        z=Z,
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.3
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title="3D Flow Field in PCA Space",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            aspectmode='cube'
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"3D flow field plot saved to {save_path}")
    
    return fig

def plot_combined_flow_field_2d(grid_points, pca_recurrence, pca_input, pca_full, fixed_points, alignment, save_path=None):
    """
    Plot a combined flow field in 2D with three subplots for different dynamics modes.
    
    Args:
        grid_points (numpy.ndarray): Grid points in PCA space
        pca_recurrence (numpy.ndarray): Recurrence-only next steps in PCA space
        pca_input (numpy.ndarray): Input-only next steps in PCA space
        pca_full (numpy.ndarray): Full dynamics next steps in PCA space
        fixed_points (numpy.ndarray): Fixed points in PCA space
        alignment (numpy.ndarray): Vector alignment values
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: (fig, axes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Compute common limits for all subplots
    x_min, x_max = grid_points[:, 0].min(), grid_points[:, 0].max()
    y_min, y_max = grid_points[:, 1].min(), grid_points[:, 1].max()
    
    # Normalize alignment values for coloring
    alignment_min = np.min(alignment)
    alignment_max = np.max(alignment)
    norm = Normalize(vmin=alignment_min, vmax=alignment_max)
    
    # Create a list of dynamics modes, corresponding data, and titles
    dynamics_data = [
        ("Recurrence Only", pca_recurrence, axes[0]),
        ("Input Only", pca_input, axes[1]),
        ("Full Dynamics", pca_full, axes[2])
    ]
    
    # Plot each dynamics mode
    for title, pca_next_steps, ax in dynamics_data:
        # Extract coordinates
        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        U = pca_next_steps[:, 0] - grid_points[:, 0]
        V = pca_next_steps[:, 1] - grid_points[:, 1]
        
        # Normalize vector lengths
        norm_vec = np.sqrt(U**2 + V**2)
        U = U / (norm_vec + 1e-8)
        V = V / (norm_vec + 1e-8)
        
        # Plot vector field
        q = ax.quiver(X, Y, U, V, alignment, cmap='Reds', norm=norm)
        
        # Add fixed points
        ax.scatter(fixed_points[:, 0], fixed_points[:, 1], 
                   marker='x', color='black', s=100, linewidth=2, label='Fixed Points')
        
        # Set limits and labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.grid(True)
    
    # Add a colorbar
    cbar = fig.colorbar(q, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label('Vector Alignment')
    
    fig.suptitle("Comparison of RNN Dynamics in PCA Space", fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined 2D flow field plot saved to {save_path}")
    
    return fig, axes

def visualize_flow_field(model_path, is_2d=True, dynamics_mode="recurrence_only", use_relu=True, 
                         alignment_method="cosine", save_path=None, input_size=100, hidden_size=150, 
                         num_layers=1, output_size=3, biased=[False, False], combined_plot=False):
    """
    Generate and visualize the flow field of an RNN model in PCA space with vector alignment coloring and fixed points.
    
    Args:
        model_path (str): Path to the model file
        is_2d (bool): Whether to generate a 2D (True) or 3D (False) plot
        dynamics_mode (str): Type of dynamics to visualize ("recurrence_only", "input_only", or "full")
        use_relu (bool): Whether to use ReLU activation
        alignment_method (str): Method to calculate vector alignment ('dot' or 'cosine')
        save_path (str, optional): Path to save the plot
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden layer
        num_layers (int): Number of RNN layers
        output_size (int): Number of output classes
        biased (list): Whether to use bias in RNN and linear layers
        combined_plot (bool): Whether to generate a combined plot with all three dynamics modes
        
    Returns:
        tuple: Plot figure and axes (for 2D) or Plotly figure (for 3D)
    """
    # Check if combined_plot is compatible with is_2d
    if combined_plot and not is_2d:
        print("Warning: Combined plot is only available for 2D visualization. Ignoring combined_plot.")
        combined_plot = False
    
    # Determine number of dimensions
    n_dims = 2 if is_2d else 3
    
    # 1. Load the model
    rnn = load_model(model_path, input_size, hidden_size, num_layers, output_size, biased)
    
    # 2. Generate sequences for PCA
    rnn_data = generate_sequences(rnn)
    
    # 3. Perform PCA
    pca, pca_result = perform_pca(rnn_data["h"], n_components=n_dims)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # 4. Create grid in PCA space
    n_points = 15 if is_2d else 10  # Fewer points for 3D to avoid clutter
    print(f"Creating {n_points}x{n_points} grid in PCA space...")
    grid_points = create_grid(pca_result, n_points=n_points, n_dims=n_dims)
    
    # 5. Map grid points to original space
    original_points = map_to_original_space(grid_points, pca)
    
    # 6. Compute dynamics for all three modes
    recurrence_steps, input_steps, full_steps = compute_both_dynamics(rnn, original_points, use_relu)
    
    # 7. Calculate vector alignment
    alignment = calculate_vector_alignment(original_points, recurrence_steps, input_steps, alignment_method)
    
    # 8. Find fixed points
    fixed_points = find_fixed_points(rnn, original_points, pca)
    
    # 9. Handle visualization based on combined_plot flag
    if combined_plot:
        # Project all dynamics to PCA space
        pca_recurrence = project_to_pca(recurrence_steps, pca)
        pca_input = project_to_pca(input_steps, pca)
        pca_full = project_to_pca(full_steps, pca)
        
        # Generate combined plot
        return plot_combined_flow_field_2d(
            grid_points, 
            pca_recurrence,
            pca_input,
            pca_full,
            fixed_points, 
            alignment, 
            save_path
        )
    else:
        # Original functionality for a single dynamics mode
        if dynamics_mode == "recurrence_only":
            next_steps = recurrence_steps
        elif dynamics_mode == "input_only":
            next_steps = input_steps
        elif dynamics_mode == "full":
            next_steps = full_steps
        else:
            raise ValueError("dynamics_mode must be 'recurrence_only', 'input_only', or 'full'")
        
        # Project next steps to PCA space
        pca_next_steps = project_to_pca(next_steps, pca)
        
        # Visualize flow field
        if is_2d:
            return plot_flow_field_2d(grid_points, pca_next_steps, fixed_points, alignment, save_path)
        else:
            return plot_flow_field_3d(grid_points, pca_next_steps, fixed_points, alignment, save_path)