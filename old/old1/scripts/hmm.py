import numpy as np
import torch
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM
import pickle
from scripts.config import *

def generate_starting_probabilities(states):
    return np.full(states, 1/states)

def generate_transition_matrix(states, stay_prob=0.95):
    # Create an empty transition matrix
    transition_matrix = np.zeros((states, states))
    change_prob = 1 - stay_prob

    # Set the probabilities for the first and last states
    transition_matrix[0, 0] = stay_prob
    transition_matrix[0, 1] = change_prob
    transition_matrix[states-1, states-1] = stay_prob
    transition_matrix[states-1, states-2] = change_prob

    # Set the probabilities for the middle states
    for i in range(1, states-1):
        transition_matrix[i, i] = stay_prob
        transition_matrix[i, i-1] = change_prob/2
        transition_matrix[i, i+1] = change_prob/2

    # Normalize the rows of the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    return transition_matrix

def generate_emission_probabilities(states, outputs):
    emission_probabilities = np.zeros((states, outputs))

    # Calculate the center of the distribution for each state
    centers = np.linspace(0, outputs - 1, num=states)

    for i in range(states):
        center = centers[i]
        for j in range(outputs):
            distance = np.abs(j - center)

            # Use a Gaussian-like distribution to assign probabilities
            emission_probabilities[i, j] = np.exp(-0.2 * (distance / (0.1 * (outputs - 1)))**2)

    # Normalize probabilities for each state to sum to 1
    emission_probabilities /= emission_probabilities.sum(axis=1, keepdims=True)

    return emission_probabilities

def generate_hmm_data(start_probabilities, transition_matrix, emission_probabilities, num_seq, seq_len, outputs):
    """
    Generates sequences and hidden states from an HMM and formats them for machine learning tasks.

    Parameters:
        start_probabilities (array): Initial state probabilities.
        transition_matrix (array): State transition probabilities.
        emission_probabilities (array): Observation emission probabilities.
        num_seq (int): Number of sequences to generate.
        seq_len (int): Length of each sequence.
        outputs (int): Number of possible output symbols (for one-hot encoding).

    Returns:
        torch.Tensor: One-hot encoded observation sequences. (Shape: num_seq, seq_len, outputs)
        np.ndarray: Hidden state sequences.
    """
    # Initialize HMM model
    model = CategoricalHMM(n_components=len(start_probabilities))
    model.startprob_ = start_probabilities
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_probabilities

    # Generate sequences
    sampled_sequences = np.zeros((num_seq, seq_len))
    sampled_states = np.zeros((num_seq, seq_len))
    for i in range(num_seq):
        observations, hidden_states = model.sample(seq_len)
        sampled_sequences[i] = observations.reshape((seq_len))
        sampled_states[i] = hidden_states

    # One-hot encode the observation sequences
    one_hot_sequences = F.one_hot(torch.tensor(sampled_sequences).long(), num_classes=outputs)

    return one_hot_sequences, sampled_states

def split_hmm_data(one_hot_sequences, sampled_states):
    """
    Splits the HMM data into training, validation, and test sets.

    Parameters:
        one_hot_sequences (torch.Tensor): One-hot encoded observation sequences.
        sampled_states (np.ndarray): Hidden state sequences.
        train_frac (float): Fraction of data to use for training.
        val_frac (float): Fraction of data to use for validation.

    Returns:
        tuple: Splits for training, validation, and test data (sequences and states).
    """
    num_seq = one_hot_sequences.shape[0]

    # Compute split indices
    train_end = num_seq // 3
    val_end = 2 * num_seq // 3

    # Split sequences
    train_seq = one_hot_sequences[:train_end]
    val_seq = one_hot_sequences[train_end:val_end]
    test_seq = one_hot_sequences[val_end:]

    # Split states
    train_states = sampled_states[:train_end]
    val_states = sampled_states[train_end:val_end]
    test_states = sampled_states[val_end:]

    return train_seq, val_seq, test_seq, train_states, val_states, test_states

def main():
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

    # Save the data to a pickle file
    output_data = {
        "train_seq": train_seq.numpy(),
        "test_seq": test_seq.numpy(),
        "val_seq": val_seq.numpy(),
        "train_states": train_states,
        "test_states": test_states,
        "val_states": val_states
    }
    output_hmm_file = "data/hmm_sequences.pkl"
    with open(output_hmm_file, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Generated sequences saved to {output_hmm_file}")

if __name__ == "__main__":
    main()