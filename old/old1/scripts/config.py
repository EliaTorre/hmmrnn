import numpy as np

# HMM Hyperparams 
#start_probabilities = np.array([1/3, 1/3, 1/3])
#transition_matrix = np.array([[0.95, 0.05, 0], [0.025, 0.95, 0.025], [0, 0.05, 0.95]])
#emission_probabilities = np.array([[0.99, 0.01, 0], [0, 0.01, 0.99], [0.5, 0, 0.5]])
num_seq = 30000
seq_len = 150
#states = 3
#outputs = 3

# RNN Training Hyperparams
grad_clip = 0.9
batch_size = 4096
tau = 1.0
input_size = 100
hidden_size = 150
num_layers = 1
epochs = 1000
lr = 0.001
lrs = [0.005, 0.001, 0.0001]
init = True

# Generate RNN Sequences
#model_path = "models/3HMM_3Outputs_30kData_0.001lr_27.9Loss_RGRG.pth"
time_steps = num_seq*seq_len

# Paths
figs_path = "figs/"
data_path = "data/"
save_model_path = "models/"
