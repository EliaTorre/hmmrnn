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
    pc1_max = 100
    pc1_min = -100
    pc2_max = 100
    pc2_min = -100

    model_files = get_model_files(evolution_dir)
    trajectories_pca = []
    epochs_list = []
    
    # Define the desired epochs to plot (interpreting range(0-20) as 1 to 20)
    desired_epochs = range(1, 201)
    
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
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(epochs_list), max(epochs_list))

    for traj, epoch in zip(trajectories_pca, epochs_list):
        if np.all((traj[:, 0] >= pc1_min) & (traj[:, 0] <= pc1_max)) and \
           np.all((traj[:, 1] >= pc2_min) & (traj[:, 1] <= pc2_max)):
            pc1 = traj[:, 0]
            pc2 = traj[:, 1]
            z = np.full(pc1.shape, epoch)
            color = cmap(norm(epoch))
            ax.scatter(pc1, pc2, z, alpha=0.5, s=2, color=color)
        else:
            print(f"Warning: Trajectory for epoch {epoch} exceeds best model's PC1/PC2 bounds. Skipping.")
            # Limit the trajectory to the bounds
            pc1 = np.clip(traj[:, 0], pc1_min, pc1_max)
            pc2 = np.clip(traj[:, 1], pc2_min, pc2_max)
            z = np.full(pc1.shape, epoch)
            color = cmap(norm(epoch))
            ax.scatter(pc1, pc2, z, alpha=0.1, s=2, color=color)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('Epoch', fontsize=12)
    ax.view_init(elev=20, azim=80)
    
    # Use a shorter title for the plot itself
    ax.set_title("3D Trajectories within Best Model's PCA Bounds", pad=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Epoch', rotation=270, labelpad=15)

    if best_steps_to_plot > 0:
        ax.legend()
    
    # Add padding to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
