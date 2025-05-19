import glob
import re
from pathlib import Path
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scripts.rnn import RNN

def get_model_files(directory):
    model_files = glob.glob(str(Path(directory) / "model_epoch_*.pth"))
    model_files.sort(key=lambda x: int(re.search(r'model_epoch_(\d+).pth', x).group(1)))
    return model_files

def model_evolution(evolution_dir, best_model_path, num_steps_best=30000, num_steps_other=50, best_steps_to_plot=0, title_prefix=None, input_size=100, hidden_size=150, num_states=None):
    evolution_dir = Path(evolution_dir)
    best_model_path = Path(best_model_path)
    rnn_best = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=3, biased=[False, False])
    rnn_best.load_model(str(best_model_path))
    ih = rnn_best.rnn.weight_ih_l0.data
    hh = rnn_best.rnn.weight_hh_l0.data
    h_best = torch.zeros((num_steps_best, hidden_size)).to(ih.device)
    for i in range(1, num_steps_best):
        x = torch.normal(mean=0, std=1, size=(input_size,)).float().to(ih.device)
        h_best[i] = torch.relu(x @ ih.T + h_best[i-1] @ hh.T)
    h_best_np = h_best.cpu().numpy()

    if np.any(np.isnan(h_best_np)):
        raise ValueError("Best model trajectory contains NaN values.")

    pca = PCA(n_components=2, random_state=0)
    h_best_pca = pca.fit_transform(h_best_np)
    pc1_max = 50
    pc1_min = -50
    pc2_max = 50
    pc2_min = -50

    model_files = get_model_files(evolution_dir)
    trajectories_pca = []
    epochs_list = []
    
    # Define the desired epochs to plot (every 5th epoch from 1 to 200)
    desired_epochs = range(1, 201, 5)
    
    for model_file in model_files:
        epoch = int(re.search(r'model_epoch_(\d+).pth', model_file).group(1))
        if epoch not in desired_epochs:
            continue  # Skip models not in the desired epochs
        rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=3, biased=[False, False])
        rnn.load_model(model_file)
        ih = rnn.rnn.weight_ih_l0.data
        hh = rnn.rnn.weight_hh_l0.data
        h_other = torch.zeros((num_steps_other, hidden_size)).to(ih.device)
        for i in range(1, num_steps_other):
            x = torch.normal(mean=0, std=1, size=(input_size,)).float().to(ih.device)
            h_other[i] = torch.relu(x @ ih.T + h_other[i-1] @ hh.T)
            if torch.norm(h_other[i]) > 1000:
                h_other[i] = h_other[i]/(torch.norm(h_other[i]) * 1000)
        h_other_np = h_other.cpu().numpy()

        if np.any(np.isnan(h_other_np)):
            print(f"Warning: Trajectory for epoch {epoch} contains NaN. Skipping.")
            continue
    
        h_other_pca = pca.transform(h_other_np)
        trajectories_pca.append(h_other_pca)
        epochs_list.append(epoch)

    # Increase figure size and add more top margin for the title
    fig = plt.figure(figsize=(14, 12))
    
    # Use suptitle for the title_prefix to avoid overlap with the plot title
    if title_prefix:
        plt.suptitle(title_prefix, fontsize=14, y=0.98)
    
    # Add more top margin to the subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color maps for logit 0 (green) and logit 2 (red)
    green_cmap = plt.cm.Greens
    red_cmap = plt.cm.Reds
    norm = plt.Normalize(min(epochs_list), max(epochs_list))

    for traj, epoch in zip(trajectories_pca, epochs_list):
        if np.all((traj[:, 0] >= pc1_min) & (traj[:, 0] <= pc1_max)) and \
           np.all((traj[:, 1] >= pc2_min) & (traj[:, 1] <= pc2_max)):
            pc1 = traj[:, 0]
            pc2 = traj[:, 1]
            z = np.full(pc1.shape, epoch)
            
            # Calculate logits by projecting hidden states onto the linear layer
            rnn_model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=3, biased=[False, False])
            rnn_model.load_model(str(evolution_dir / f"model_epoch_{epoch}.pth"))
            fc_weights = rnn_model.fc.weight.data
            
            # Get the hidden states from the trajectory
            h_other_np = pca.inverse_transform(traj)
            h_other_tensor = torch.tensor(h_other_np).float().to(fc_weights.device)
            
            # Project hidden states onto the linear layer to get logits
            logits = h_other_tensor @ fc_weights.T
            
            # Determine the dominant logit for each point
            dominant_logits = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Create arrows showing the motion (use every nth point to avoid overcrowding)
            n = 5  # Use every 5th point
            for i in range(0, len(pc1)-n, n):
                # Get the dominant logit for this segment
                dom_logit = dominant_logits[i]
                
                # Choose color based on dominant logit and epoch (darker as epoch increases)
                if dom_logit == 0:  # Green for logit 0
                    color = green_cmap(norm(epoch))
                elif dom_logit == 2:  # Red for logit 2
                    color = red_cmap(norm(epoch))
                else:  # Gray for logit 1
                    color = (0.5, 0.5, 0.5, 0.5)  # Gray with some transparency
                
                # Draw arrow
                ax.quiver(pc1[i], pc2[i], z[i], 
                          pc1[i+n] - pc1[i], pc2[i+n] - pc2[i], z[i+n] - z[i],
                          color=color, alpha=0.7, arrow_length_ratio=0.3, linewidth=1.5)
        else:
            print(f"Warning: Trajectory for epoch {epoch} exceeds best model's PC1/PC2 bounds. Skipping.")
            # Limit the trajectory to the bounds
            pc1 = np.clip(traj[:, 0], pc1_min, pc1_max)
            pc2 = np.clip(traj[:, 1], pc2_min, pc2_max)
            z = np.full(pc1.shape, epoch)
            
            # Calculate logits by projecting hidden states onto the linear layer
            rnn_model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=3, biased=[False, False])
            rnn_model.load_model(str(evolution_dir / f"model_epoch_{epoch}.pth"))
            fc_weights = rnn_model.fc.weight.data
            
            # Get the hidden states from the trajectory
            h_other_np = pca.inverse_transform(traj)
            h_other_tensor = torch.tensor(h_other_np).float().to(fc_weights.device)
            
            # Project hidden states onto the linear layer to get logits
            logits = h_other_tensor @ fc_weights.T
            
            # Determine the dominant logit for each point
            dominant_logits = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Create arrows showing the motion (use every nth point to avoid overcrowding)
            n = 5  # Use every 5th point
            for i in range(0, len(pc1)-n, n):
                # Get the dominant logit for this segment
                dom_logit = dominant_logits[i]
                
                # Choose color based on dominant logit and epoch (darker as epoch increases)
                if dom_logit == 0:  # Green for logit 0
                    color = green_cmap(norm(epoch))
                elif dom_logit == 2:  # Red for logit 2
                    color = red_cmap(norm(epoch))
                else:  # Gray for logit 1
                    color = (0.5, 0.5, 0.5, 0.5)  # Gray with some transparency
                
                # Draw arrow with reduced alpha for out-of-bounds trajectories
                ax.quiver(pc1[i], pc2[i], z[i], 
                          pc1[i+n] - pc1[i], pc2[i+n] - pc2[i], z[i+n] - z[i],
                          color=color, alpha=0.3, arrow_length_ratio=0.3, linewidth=1.5)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('Epoch', fontsize=12)
    ax.view_init(elev=20, azim=80)
    
    # Use a shorter title for the plot itself
    ax.set_title("3D Trajectories with Dominant Logit Coloring", pad=20)

    # Create custom legend for logit colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Logit 0 (Green)'),
        Line2D([0], [0], color='gray', lw=2, label='Logit 1 (Gray)'),
        Line2D([0], [0], color='red', lw=2, label='Logit 2 (Red)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    if best_steps_to_plot > 0:
        ax.legend()
    
    # Add padding to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
