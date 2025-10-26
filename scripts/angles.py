# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from rnn import RNN

def angles(model_path=None, timesteps=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size=100 
    hidden_size=150 
    output_size=3

    # Load RNN model
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    rnn.load_state_dict(torch.load(model_path, map_location=device))
    rnn.eval()

    # Extract weights
    ih = rnn.rnn.weight_ih_l0.data
    hh = rnn.rnn.weight_hh_l0.data
    fc = rnn.fc.weight.data

    # Initialize hidden state
    h = torch.normal(0, 1, size=(hidden_size,), device=device)
    hidden_states = []

    with torch.no_grad():
        for t in range(timesteps):
            x = torch.normal(0, 1, size=(input_size,), device=device)
            h = torch.relu(h @ hh.T + x @ ih.T)
            hidden_states.append(h.cpu().numpy())

    hidden_states = np.array(hidden_states)

    # Get the first two principal components (they span the plane)
    pca = PCA(n_components=2)
    pca.fit(hidden_states)
    pc_axes = pca.components_
    pc1 = pc_axes[0]
    pc2 = pc_axes[1]
    
    # Define the plane spanned by PC1 and PC2
    # PC1 and PC2 are already orthonormal (from PCA), so they form a basis for the plane
    
    # Get readout layer axes (rows of fc correspond to output dimensions)
    readout_axes = fc.cpu().numpy()  # Shape: (output_size, hidden_size)
    
    # Calculate angles between each readout axis and the PC plane
    angles_with_plane = []
    projections_on_plane = []
    
    for i in range(output_size):
        readout_axis = readout_axes[i]
        readout_axis = readout_axis / np.linalg.norm(readout_axis)  # Normalize
        
        # Project the readout axis onto the PC plane
        # The projection onto the plane spanned by pc1 and pc2 is:
        # proj = (axis · pc1) * pc1 + (axis · pc2) * pc2
        proj_pc1 = np.dot(readout_axis, pc1)
        proj_pc2 = np.dot(readout_axis, pc2)
        projection = proj_pc1 * pc1 + proj_pc2 * pc2
        
        # The magnitude of the projection
        projection_magnitude = np.sqrt(proj_pc1**2 + proj_pc2**2)
        projections_on_plane.append(projection_magnitude)
        
        # The angle between the vector and the plane is:
        # arcsin(|component perpendicular to plane| / |vector|)
        # Since the vector is normalized, this simplifies to:
        # arcsin(sqrt(1 - projection_magnitude^2))
        # Or equivalently: arccos(projection_magnitude)
        angle_with_plane = np.arccos(np.clip(projection_magnitude, 0, 1))
        angles_with_plane.append(np.degrees(angle_with_plane))
    
    # Also calculate the variance captured by projecting readout axes onto PC plane
    readout_projections_2d = []
    for i in range(output_size):
        readout_axis = readout_axes[i]
        proj_pc1 = np.dot(readout_axis, pc1)
        proj_pc2 = np.dot(readout_axis, pc2)
        readout_projections_2d.append([proj_pc1, proj_pc2])
    
    # Print results
    print(f"Explained variance by PC1 and PC2: {pca.explained_variance_ratio_[:2].sum():.3f}")
    print(f"\nAngles between readout axes and PC plane:")
    for i in range(output_size):
        print(f"  Output {i}: {angles_with_plane[i]:.2f}° (projection magnitude: {projections_on_plane[i]:.3f})")
    
    # Visualize the results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Bar chart of angles
    ax1.bar(range(output_size), angles_with_plane)
    ax1.set_xlabel('Output dimension')
    ax1.set_ylabel('Angle with PC plane (degrees)')
    ax1.set_title('Angles between readout axes and PC1-PC2 plane')
    ax1.set_xticks(range(output_size))
    ax1.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='45°')
    ax1.legend()
    
    # Plot 2: Bar chart of projections
    ax2.bar(range(output_size), projections_on_plane)
    ax2.set_xlabel('Output dimension')
    ax2.set_ylabel('Projection magnitude')
    ax2.set_title('Projection of readout axes onto PC1-PC2 plane')
    ax2.set_xticks(range(output_size))
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1/np.sqrt(2), color='r', linestyle='--', alpha=0.5, label='1/√2')
    ax2.legend()
    
    # Plot 3: 2D visualization of readout projections on PC plane
    ax3.scatter([0], [0], marker='o', s=100, c='black', label='Origin')
    for i in range(output_size):
        ax3.arrow(0, 0, readout_projections_2d[i][0], readout_projections_2d[i][1], 
                  head_width=0.05, head_length=0.05, fc=f'C{i}', ec=f'C{i}', 
                  label=f'Output {i}')
    ax3.set_xlabel('PC1 component')
    ax3.set_ylabel('PC2 component')
    ax3.set_title('Readout axes projected onto PC1-PC2 plane')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'pc1': pc1,
        'pc2': pc2,
        'angles_with_plane': angles_with_plane,
        'projections_on_plane': projections_on_plane,
        'explained_variance': pca.explained_variance_ratio_[:2].sum(),
        'readout_projections_2d': readout_projections_2d
    }

# %%
models=["TrainedModels/ReverseEngineeredModel/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth", # 2 States
        "TrainedModels/Two/hidden_150/input_100/seed_0/models/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth", # 2 States
        "TrainedModels/Two/hidden_150/input_100/seed_1/models/2HMM_3Outputs_linear_30kData_0.001lr_9.8Loss.pth", # 2 States
        "TrainedModels/Three/hidden_150/input_100/seed_0/models/3HMM_3Outputs_linear_30kData_0.001lr_4.1Loss.pth", # 3 States
        "TrainedModels/Three/hidden_150/input_100/seed_1/models/3HMM_3Outputs_linear_30kData_0.001lr_4.1Loss.pth", # 3 States
        "TrainedModels/Three/hidden_150/input_100/seed_2/models/3HMM_3Outputs_linear_30kData_0.001lr_4.1Loss.pth", # 3 States
        "TrainedModels/Four/hidden_150/input_100/seed_0/models/4HMM_3Outputs_linear_30kData_0.001lr_4.6Loss.pth", # 4 States
        "TrainedModels/Four/hidden_150/input_100/seed_1/models/4HMM_3Outputs_linear_30kData_0.001lr_4.6Loss.pth", # 4 States
        "TrainedModels/Four/hidden_150/input_100/seed_2/models/4HMM_3Outputs_linear_30kData_0.001lr_4.7Loss.pth", # 4 States
        "TrainedModels/Five/hidden_150/input_100/seed_0/models/5HMM_3Outputs_linear_30kData_0.001lr_7.8Loss.pth", # 5 States
        "TrainedModels/Five/hidden_150/input_100/seed_1/models/5HMM_3Outputs_linear_30kData_0.001lr_7.7Loss.pth", # 5 States
        "TrainedModels/Five/hidden_150/input_100/seed_2/models/5HMM_3Outputs_linear_30kData_0.001lr_7.6Loss.pth" # 5 States
        ]

projections = []
angles_list = []
for model_path in models:
    result = angles(model_path=model_path, timesteps=5000)
    projections.append(result['projections_on_plane'])
    angles_list.append(result['angles_with_plane'])

projections = np.array(projections)
angles_list = np.array(angles_list)
# %%
model_groups = {
    '2 States': [0, 1, 2],      # First 2 models
    '3 States': [3, 4, 5],   # Next 3 models  
    '4 States': [6, 7, 8],   # Next 3 models
    '5 States': [9, 10, 11]   # Last 3 models
}

# Calculate averaged projections (average of columns 0 and 2, disregard column 1)
averaged_projections = (projections[:, 0] + projections[:, 2]) / 2

# Calculate means and standard deviations for each group
states = ['2 States', '3 States', '4 States', '5 States']
projection_means = []
projection_stds = []

for state in states:
    indices = model_groups[state]
    
    # For projections: average columns 0 and 2, then calculate stats across seeds
    proj_values = averaged_projections[indices]
    projection_means.append(np.mean(proj_values))
    projection_stds.append(np.std(proj_values))

# Create the plot
fig, ax = plt.subplots(figsize=(7, 5))

# Plot projections
x_positions = np.arange(len(states))
ax.errorbar(x_positions, projection_means, yerr=projection_stds, 
            marker='o', color='blue', linewidth=2, capsize=5, 
            label='Projections (avg. of readout axes 0 & 2)', markersize=8)

ax.set_xlabel('Number of HMM States', fontsize=12)
ax.set_ylabel('Projections on Plane', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(states)
ax.grid(True, alpha=0.3)
ax.legend()

plt.title('RNN Readouts Projections on PCA plane vs # States', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('angles_readout_pc_plane.svg', format='svg')
plt.show()
# %%
