import numpy as np
import torch
from scripts.hmm import generate_hmm_data, split_hmm_data
from scripts.train import train, plot_losses
from scripts.tests import tests
from scripts.pca import projection
from scripts.config import *


def main(start_probabilities, transition_matrix, emission_probabilities, outputs, states, path):
    print("Generating Initial HMM Data...")
	# Generate HMM sequences
    one_hot_sequences, sampled_states = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )
    
    train_seq, val_seq, test_seq, train_states, val_states, test_states = split_hmm_data(
        one_hot_sequences, sampled_states
    )

    print("Training RNN Models...")
    best_loss, best_model = float('inf'), None
    best_train_losses, best_val_losses, best_model_name = None, None, None

    for lr in lrs: 
    	# Train the model and get the loss history
        train_losses, val_losses, model, model_name, min_loss = train(
            batch_size=batch_size,
            lr=lr,
            tau=tau,
            epochs=epochs,
            train_seq=train_seq,
            val_seq=val_seq,
            outputs=outputs, 
            num_states=states,
            init=init
        )

        if min_loss < best_loss: 
            best_loss, best_model = min_loss, model
            best_train_losses, best_val_losses, best_model_name = train_losses, val_losses, model_name

    # Save trained model
    print("Saving Best Model...")
    model_file = path + "models/" + best_model_name
    torch.save(best_model, model_file)
    print(f'Model saved to {model_file}')

    # Call the function to plot training and validation loss curves
    plot_losses(best_train_losses, best_val_losses, path=path)

    print("Evaluating RNN Fit Quality...")
    tests(start_probabilities, transition_matrix, emission_probabilities, outputs, states, model_file, path)

    print("Projecting RNN Hidden Trajectories in PCA Space...")
    projection(outputs, model_file, path)
