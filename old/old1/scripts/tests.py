import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.sinkhorn import SinkhornSolver
from scripts.hmm import generate_hmm_data
from scripts.run import run
from scripts.config import *

# Function to match similar sequences
def match(hmm_data, rnn_data):
    sh = SinkhornSolver(epsilon=0.1, iterations=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hmm_flattened = hmm_data.reshape(hmm_data.shape[0], -1)
    hmm_flattened = torch.tensor(hmm_flattened).float().to(device)

    rnn_flattened = rnn_data.reshape(rnn_data.shape[0], -1)
    rnn_flattened = torch.tensor(rnn_flattened).float().to(device)

    tp = sh(hmm_flattened, rnn_flattened)
    matched_hmm = hmm_flattened[tp[1].argmax(0)].cpu().detach().numpy()
    matched_rnn = rnn_flattened.cpu().detach().numpy()

    return matched_hmm, matched_rnn


# Function to calculate Euclidean Distance
def dist(matched_hmm, matched_rnn):
    distances = np.linalg.norm(matched_hmm - matched_rnn, axis=1)
    return distances


# Function to calculate observation volatilities
def diffs(seq):
    seq_max = np.argmax(seq, axis=2)
    diff = np.diff(seq_max, axis=1)
    changes = np.count_nonzero(diff, axis=1)
    return np.mean(changes), np.std(changes)


# Function to calculate observation frequencies
def frequencies(seq):
    frequencies = np.zeros((seq.shape[0], seq.shape[2]))
    seq = np.argmax(seq, axis=2)
    for i in range(num_seq):
        unique, counts = np.unique(seq[i], return_counts=True)
        frequencies[i, unique] = counts / sum(counts)
    return np.mean(frequencies, axis=0), np.std(frequencies, axis=0)

# Function to calculate transition matrices
def trans(seq, outputs):
    trans = np.zeros((seq.shape[0], outputs, outputs))
    seq_max = np.argmax(seq, axis=2)
    for i in range(seq.shape[0]):
        mat = np.zeros((outputs, outputs))
        for j in range(1, seq_len):
            mat[seq_max[j-1], seq_max[j]] += 1
        trans[i] = mat / mat.sum(axis=1, keepdims=True)
    return np.mean(trans, axis=0)


# Function to generate and save plots
def generate_plots(hmm_distances, rnn_distances, hmm_volatilities, rnn_volatilities, hmm_freqs, rnn_freqs, hmm_trans, rnn_trans, outputs, states, path=''):
    # 1. Euclidean distance plot
    plt.figure(figsize=(4, 4))
    x_positions = [0, 0.5]
    plt.errorbar(x_positions[0], np.mean(hmm_distances),yerr=np.std(hmm_distances), fmt='o', color='darkgreen', capsize=3)
    plt.errorbar(x_positions[1], np.mean(rnn_distances),yerr=np.std(rnn_distances), fmt='o', color='darkred', capsize=3)
    plt.xticks(x_positions, ["HMM", "RNN"], fontsize=10)
    for i, (mean, std) in enumerate(zip([np.mean(hmm_distances), np.mean(rnn_distances)],
                                         [np.std(hmm_distances), np.std(rnn_distances)])):
        plt.text(x_positions[i], mean + std + 0.1, f"{mean:.2f}", ha='center', fontsize=8)
    plt.title("Mean Euclidean Distances", fontsize=16, fontweight="bold")
    plt.ylabel("Distance")
    plt.savefig(path+"figs/euclidean_distances.pdf")
    plt.close()

    # 2. Observation volatilities
    plt.figure(figsize=(4, 4))
    x_positions = [0, 0.5]
    plt.errorbar(x_positions[0], hmm_volatilities[0], yerr=hmm_volatilities[1], fmt='o', color='darkred', capsize=3)
    plt.errorbar(x_positions[1], rnn_volatilities[0], yerr=rnn_volatilities[1], fmt='o', color='darkgreen', capsize=3)
    plt.xticks(x_positions, ["HMM", "RNN"], fontsize=10)
    for i, (mean, std) in enumerate(zip([hmm_volatilities[0], rnn_volatilities[0]],[hmm_volatilities[1], rnn_volatilities[1]])):
        plt.text(x_positions[i], mean + std + 1, f"{mean:.2f}", ha='center', fontsize=8)
    plt.title("Observation Volatilities", fontsize=16, fontweight="bold")
    plt.ylabel("# of Changes in Outputs")
    plt.savefig(path+"figs/observation_volatilities.pdf")
    plt.close()

    # 3. Output frequencies
    bar_width = 0.4
    indices = np.arange(outputs)
    plt.figure(figsize=(8, 6))
    plt.errorbar(indices - bar_width / 2, hmm_freqs[0], yerr=hmm_freqs[1], fmt='o', label="HMM", color="darkgreen", capsize=3)
    plt.errorbar(indices + bar_width / 2, rnn_freqs[0], yerr=rnn_freqs[1], fmt='o', label="RNN", color="darkred", capsize=3)
    for i, (hmm_value, hmm_err, rnn_value, rnn_err) in enumerate(zip(hmm_freqs[0], hmm_freqs[1], rnn_freqs[0], rnn_freqs[1])):
        plt.text(i - bar_width / 2, hmm_value + hmm_err + 0.01, f"{hmm_value:.2f}", fontsize=10, ha="center", va="bottom")
        plt.text(i + bar_width / 2, rnn_value + rnn_err + 0.01, f"{rnn_value:.2f}", fontsize=10, ha="center", va="bottom")
    plt.xticks(indices, [f"Output {i + 1}" for i in range(outputs)])
    plt.title("Output Frequencies", fontsize=16, fontweight="bold")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(path+"figs/output_frequencies.pdf")
    plt.close()

    # 4. Transition matrices
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, matrix, title in zip(axs, [hmm_trans, rnn_trans], ["HMM", "RNN"]):
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(states))
        ax.set_yticks(np.arange(states))
        ax.set_xticklabels(np.arange(1, states + 1))
        ax.set_yticklabels(np.arange(1, states + 1))
        for i in range(states):
            for j in range(states):
                ax.text(j, i, f"{matrix[i, j]:.2f}", va="center", ha="center")
    fig.colorbar(cax, ax=axs.ravel().tolist(), shrink=0.95)
    plt.suptitle('Transition Matrices between Outputs', fontsize=16, fontweight='bold')
    plt.savefig(path+"figs/transition_matrices.pdf")
    plt.close()

def tests(start_probabilities, transition_matrix, emission_probabilities, outputs, states, model, path): 
    print("Generating HMM Data...")
    # Generate HMM sequences
    hmm_data, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    hmm_data2, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    hmm_data3, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    print("Generating RNN Data...")
    # Generate RNN sequences
    rnn_outputs = run(
        model_path=model,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=outputs,
        time_steps=time_steps,
        dynamics_mode="full"
    )
    rnn_data = rnn_outputs["outs"].reshape(num_seq, seq_len, outputs)

    print("Evaluating Euclidean Distances...")
    # Evaluate Euclidean distances
    hmmvhmm_matched, hmmvhmm_matched2 = match(hmm_data2, hmm_data3)
    hmmvrnn_matched, rnnvhmm_matched = match(hmm_data, rnn_data)
    hmm_distances = dist(hmmvhmm_matched, hmmvhmm_matched2)
    rnn_distances = dist(hmmvrnn_matched, rnnvhmm_matched)  # Use the same distances as a placeholder

    print("Evaluating Volatilities...")
    # Evaluate observation volatilities
    hmm_volatilities = diffs(hmm_data)
    rnn_volatilities = diffs(rnn_data)

    print("Evaluating Frequencies...")
    # Evaluate output frequencies
    hmm_freqs = frequencies(hmm_data)
    rnn_freqs = frequencies(rnn_data)

    print("Evaluating Transition Matrices...")
    # Evaluate transition matrices
    hmm_trans = trans(hmm_data, outputs)
    rnn_trans = trans(rnn_data, outputs)

    print("Generating Plots...")
    # Generate and save plots
    generate_plots(hmm_distances, rnn_distances, hmm_volatilities, rnn_volatilities, hmm_freqs, rnn_freqs, hmm_trans, rnn_trans, outputs, states, path)

def main():
    print("Generating HMM Data...")
    # Generate HMM sequences
    hmm_data, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    hmm_data2, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    hmm_data3, _ = generate_hmm_data(
        start_probabilities,
        transition_matrix,
        emission_probabilities,
        num_seq,
        seq_len,
        outputs
    )

    print("Generating RNN Data...")
    # Generate RNN sequences
    rnn_outputs = run(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=outputs,
        time_steps=time_steps,
        dynamics_mode="full"
    )
    rnn_data = rnn_outputs["outs"].reshape(num_seq, seq_len, outputs)

    print("Evaluating Euclidean Distances...")
    # Evaluate Euclidean distances
    hmmvhmm_matched, hmmvhmm_matched2 = match(hmm_data2, hmm_data3)
    hmmvrnn_matched, rnnvhmm_matched = match(hmm_data, rnn_data)
    hmm_distances = dist(hmmvhmm_matched, hmmvhmm_matched2)
    rnn_distances = dist(hmmvrnn_matched, rnnvhmm_matched)  # Use the same distances as a placeholder

    print("Evaluating Volatilities...")
    # Evaluate observation volatilities
    hmm_volatilities = diffs(hmm_data)
    rnn_volatilities = diffs(rnn_data)

    print("Evaluating Frequencies...")
    # Evaluate output frequencies
    hmm_freqs = frequencies(hmm_data)
    rnn_freqs = frequencies(rnn_data)

    print("Evaluating Transition Matrices...")
    # Evaluate transition matrices
    hmm_trans = trans(hmm_data)
    rnn_trans = trans(rnn_data)

    print("Generating Plots...")
    # Generate and save plots
    generate_plots(hmm_distances, rnn_distances, hmm_volatilities, rnn_volatilities, hmm_freqs, rnn_freqs, hmm_trans, rnn_trans)


if __name__ == "__main__":
    main()
