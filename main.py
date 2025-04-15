#%%

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from pathlib import Path
from scripts.manager import Manager
from scripts.rnn import RNN
from scripts.mechint import visualize_flow_field, visualize_flow_field_svd_dir, visualize_flow_field_svd_dim, visualize_singular_vector_flow_fields, rnn_noise_var, noise_cosine_similarity
from scripts.ReLU import visualize_flow

#%%


# Initialize a list of sequence lengths to experiment with
#seq_lengths_2 = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
#seq_lengths_rest = [30, 35, 40]
seq_lengths = [30, 30, 30, 30, 40, 40, 40, 40]
configs = ["HMMThree"]
for name in configs:
    print(f"\n--- Running experiment with config = {name} ---\n")
        
    for seq_len in seq_lengths:
        print(f"\n--- Running experiment with seq_len = {seq_len} ---\n")

        manager = Manager(config_name=name)
        manager.config['seq_len'] = seq_len
        
        # Print the updated configuration
        #print("Configuration:")
        #for key, value in manager.config.items(): 
            #print(f"  {key}: {value}")
        
        # Run the experiment with verbose output
        results = manager.run_experiment(verbose=False)
    
print("\n--- All experiments completed ---")


# Show the experiment directory structure
import os

def print_directory_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

print_directory_tree("Experiments")


# Create a new manager
multi_manager = Manager()

# Run multiple experiments
config_names = ["HMMTwo", "HMMThree", "HMMFour", "HMMFive"]
all_results = multi_manager.run_multiple_experiments(config_names, verbose=False)


manager = Manager(config_name="HMMThree_RGB")

# Point to the experiment directory where your model is saved
manager.config_dir = Path("Experiments/20250304_162931/HMMThree_RGB")
manager.models_path = manager.config_dir / "models"
manager.figs_path = manager.config_dir / "figs"
manager.data_path = manager.config_dir / "data"


all_fps, unique_fps = manager.find_fixed_points(num_initial_states=1000)
print(f"Found {len(all_fps)} total fixed points, {len(unique_fps)} unique fixed points")


#old_models
#/home/elia/Documents/rnnrep/old_models/2HMM_3Outputs_30kData_0.001lr_19.4Loss.pth
#/home/elia/Documents/rnnrep/old_models/3HMM_3Outputs_30kData_0.001lr_31.9Loss.pth
#/home/elia/Documents/rnnrep/old_models/4HMM_3Outputs_30kData_0.001lr_33.2Loss.pth
#/home/elia/Documents/rnnrep/old_models/5HMM_3Outputs_30kData_0.001lr_37.3Loss.pth
manager = Manager()
all_fps, unique_fps = manager.find_fixed_points(num_initial_states=1000, num_traj=10, plot_unique=True, model_path="/home/elia/Documents/rnnrep/old_models/5HMM_3Outputs_30kData_0.001lr_37.3Loss.pth")
print(f"Found {len(all_fps)} total fixed points, {len(unique_fps)} unique fixed points")


visualize_flow_field("/home/elia/Documents/rnnrep/Experiments/2/2_100_1/HMMTwo/models/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth", alignment_method='cosine', use_relu=True, color_by='magnitude')
visualize_flow_field("/home/elia/Documents/rnnrep/Experiments/2/2_100_1/HMMTwo/models/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth", alignment_method='cosine', use_relu=False, color_by='magnitude')
visualize_flow_field("/home/elia/Documents/rnnrep/Experiments/2/2_100_0/HMMTwo/models/2HMM_3Outputs_linear_30kData_0.001lr_9.8Loss.pth", alignment_method='cosine', use_relu=True, color_by='magnitude')
visualize_flow_field("/home/elia/Documents/rnnrep/Experiments/2/2_100_0/HMMTwo/models/2HMM_3Outputs_linear_30kData_0.001lr_9.8Loss.pth", alignment_method='cosine', use_relu=False, color_by='magnitude')




rnn = RNN(input_size=100, hidden_size=150, num_layers=1, output_size=3)
rnn.load_state_dict(torch.load("/home/elia/Documents/rnnrep/Experiments/2/2_100_1/HMMTwo/models/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth"))
ih = rnn.rnn.weight_ih_l0.data.cpu().numpy()
hh = rnn.rnn.weight_hh_l0.data.cpu().numpy()

def extract_top_svd_directions(matrix1, matrix2, n_components=5):
    U1, s1, V1 = np.linalg.svd(matrix1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(matrix2, full_matrices=False)
    top_directions1 = U1[:, :n_components].T
    top_directions2 = U2[:, :n_components].T
    return top_directions1, top_directions2

def relu(x):
    result = x.copy()
    result[result < 0] = 0
    return result

ih_dir, hh_dir = extract_top_svd_directions(ih, hh, n_components=5)
ih_dir_relu, hh_dir_relu = relu(ih_dir), relu(hh_dir)  

stacked_directions = np.vstack([ih_dir, hh_dir, ih_dir_relu, hh_dir_relu])
alignments = np.zeros((20, 20))

for dir1, i in zip(stacked_directions, range(len(stacked_directions))):
    for dir2, j in zip(stacked_directions, range(len(stacked_directions))):
        alignments[i, j] = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
        #alignments[i, j] = np.dot(dir1, dir2)
plt.figure(figsize=(8, 8))
plt.imshow(alignments, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Alignment Value')
plt.title('20x20 Alignments Matrix')
plt.xlabel('Direction Index')
plt.ylabel('Direction Index')
plt.xticks(range(20), ['ih_SV1', 'ih_SV2', 'ih_SV3', 'ih_SV4', 'ih_SV5', 
                       'hh_SV1', 'hh_SV2', 'hh_SV3', 'hh_SV4', 'hh_SV5', 
                       'ih_SV1_relu', 'ih_SV2_relu', 'ih_SV3_relu', 'ih_SV4_relu', 'ih_SV5_relu', 
                       'hh_SV1_relu', 'hh_SV2_relu', 'hh_SV3_relu', 'hh_SV4_relu', 'hh_SV5_relu'], rotation=90)
plt.yticks(range(20), ['ih_SV1', 'ih_SV2', 'ih_SV3', 'ih_SV4', 'ih_SV5', 
                       'hh_SV1', 'hh_SV2', 'hh_SV3', 'hh_SV4', 'hh_SV5', 
                       'ih_SV1_relu', 'ih_SV2_relu', 'ih_SV3_relu', 'ih_SV4_relu', 'ih_SV5_relu', 
                       'hh_SV1_relu', 'hh_SV2_relu', 'hh_SV3_relu', 'hh_SV4_relu', 'hh_SV5_relu'])
plt.yticks(range(20))
plt.show()