import argparse
import torch
import torch.nn.functional as F
import pickle
import os
from scripts.VanillaRNN import VanillaRNN
from scripts.config import *

def run(model_path, input_size, hidden_size, num_layers, num_outputs, 
                          time_steps, dynamics_mode, device=None):
    """
    Generate hidden states (h), logits, and outputs for a trained VanillaRNN model.

    Parameters:
    - model_path (str): Path to the trained model (.pth file).
    - input_size (int): Size of the input features.
    - hidden_size (int): Size of the hidden layer.
    - num_layers (int): Number of RNN layers.
    - num_outputs (int): Number of output classes.
    - time_steps (int): Number of time steps to generate.
    - dynamics_mode (str): Dynamics mode ('full', 'recurrence_only', 'input_only').
    - device (torch.device, optional): Device to run the computation (CPU or CUDA).

    Returns:
    - outputs (dict): Dictionary containing hidden states (h), logits, and outputs (outs).
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = VanillaRNN(input_size, hidden_size, num_layers, num_outputs, biased=[False, False]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Extract weights
    ih = model.rnn.weight_ih_l0.data.to(device)
    hh = model.rnn.weight_hh_l0.data.to(device)
    fc = model.fc.weight.data.to(device)

    # Initialize tensors
    h = torch.zeros((time_steps, hidden_size)).to(device)
    logits = torch.zeros((time_steps, num_outputs)).to(device)
    outs = torch.zeros((time_steps, num_outputs)).to(device)
    h[0] = torch.zeros((hidden_size)).to(device)

    # Generate hidden states, logits, and outputs
    for t in range(1, time_steps):
        x = torch.normal(mean=0, std=1, size=(input_size,)).float().to(device)

        if dynamics_mode == 'full':
            h[t] = torch.relu(x @ ih.T + h[t-1] @ hh.T)
        elif dynamics_mode == 'recurrence_only':
            h[t] = torch.relu(h[t-1] @ hh.T)
        elif dynamics_mode == 'input_only':
            identity_matrix = torch.eye(hidden_size, device=device)
            h[t] = torch.relu(x @ ih.T + h[t-1] @ identity_matrix)
        else:
            raise ValueError("Invalid dynamics_mode. Choose from 'full', 'recurrence_only', 'input_only'.")

    logits = h @ fc.T
    outs = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1)

    # Convert tensors to numpy arrays
    h = h.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    outs = outs.cpu().detach().numpy()

    # Return results in a dictionary
    outputs = {"h": h, "logits": logits, "outs": outs}
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Generate outputs from a trained VanillaRNN model.")
    parser.add_argument("--dynamics_mode", type=str, required=True, 
                        choices=['full', 'recurrence_only', 'input_only'], 
                        help="Dynamics mode: 'full', 'recurrence_only', or 'input_only'.")

    args = parser.parse_args()

    outputs = run(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=outputs,
        time_steps=time_steps,
        dynamics_mode=args.dynamics_mode
    )

    # Save outputs as a pickle file in the current working directory
    output_rnn_file = "data/rnn_sequences.pkl"
    with open(output_rnn_file, 'wb') as f:
        pickle.dump(outputs, f)

    print(f"Outputs saved to {output_rnn_file}")

if __name__ == "__main__":
    main()
