import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
from scripts.rnn import RNN

def neuron_activity_analysis(n_points=1000, neuron_index=0, model_path=None, device=None):
    """
    Analyze neuron activity in PCA space, similar to neuron_activities function.
    
    Parameters:
    - n_points: Number of time steps to simulate
    - neuron_index: Index of neuron to analyze (0 to hidden_size-1)
    - model_path: Path to model file (default uses standard path)
    - device: Device for computation
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    input_size = 100
    hidden_size = 150
    output_size = 3
    
    # Default model path
    if model_path is None:
        #model_path = "Experiments/20250616_190051/HMMThreeTriangularFully/models/3HMM_3Outputs_triangular_30kData_0.001lr_1.8Loss.pth"
        model_path = "/home/elia/Documents/rnnrep/TrainedModels/Five/hidden_150/input_100/seed_0/models/5HMM_3Outputs_linear_30kData_0.001lr_7.8Loss.pth"
    
    # Load RNN model
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    rnn.load_state_dict(torch.load(model_path, map_location=device))
    rnn.eval()
    
    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    
    # Initialize hidden state
    h = torch.normal(0, 1, size=(hidden_size,), device=device)
    
    # Simulate RNN dynamics
    hidden_states = []
    neuron_activities = []
    
    with torch.no_grad():
        for t in range(n_points):
            x = torch.normal(0, 1, size=(input_size,), device=device)
            pre_act = h @ hh.T + x @ ih.T
            h = torch.relu(pre_act)
            
            hidden_states.append(h.cpu().numpy())
            neuron_activities.append(pre_act[neuron_index].cpu().item())
    
    # Convert to numpy arrays
    hidden_states = np.array(hidden_states)
    neuron_activities = np.array(neuron_activities)
    
    # Apply PCA to hidden states
    pca = PCA(n_components=2)
    hidden_states_pca = pca.fit_transform(hidden_states)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create trajectory using LineCollection for colored segments
    points = hidden_states_pca.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection with neuron activity coloring
    lc = LineCollection(segments, cmap='seismic', linewidths=1.5, alpha=0.8)
    lc.set_array(neuron_activities[:-1])  # Use activity values for coloring
    
    # Add the trajectory to the plot
    line = ax.add_collection(lc)
    
    # Add directional arrows along the trajectory
    # Sample every N points to avoid overcrowding
    arrow_spacing = max(1, n_points // 20)  # Show ~20 arrows total
    for i in range(arrow_spacing, n_points - arrow_spacing, arrow_spacing):
        # Calculate direction vector
        dx = hidden_states_pca[i+1, 0] - hidden_states_pca[i, 0]
        dy = hidden_states_pca[i+1, 1] - hidden_states_pca[i, 1]
        
        # Add arrow
        ax.annotate('', xy=(hidden_states_pca[i, 0] + dx*0.5, hidden_states_pca[i, 1] + dy*0.5),
                    xytext=(hidden_states_pca[i, 0] - dx*0.5, hidden_states_pca[i, 1] - dy*0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.7),
                    zorder=4)
    
    # Mark the starting point
    ax.plot(hidden_states_pca[0, 0], hidden_states_pca[0, 1], 'go', markersize=8, 
            label='Start', zorder=5)
    
    # Mark the ending point
    ax.plot(hidden_states_pca[-1, 0], hidden_states_pca[-1, 1], 'ro', markersize=8, 
            label='End', zorder=5)
    
    # Set axis limits
    ax.set_xlim(hidden_states_pca[:, 0].min() - 0.5, hidden_states_pca[:, 0].max() + 0.5)
    ax.set_ylim(hidden_states_pca[:, 1].min() - 0.5, hidden_states_pca[:, 1].max() + 0.5)
    
    # Labels and formatting
    ax.set_xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.3f})', fontsize=14)
    ax.set_ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.3f})', fontsize=14)
    ax.set_title(f'Hidden State Trajectory in PCA Space - Neuron {neuron_index} Activity', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend for start/end points
    ax.legend(loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label(f'Neuron {neuron_index} Activity', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, hidden_states_pca, pca, hidden_states, neuron_activities

if __name__ == "__main__":
    # Example usage - analyze different neurons
    for i in range(150):
        neuron_activity_analysis(n_points=1500, neuron_index=i)
