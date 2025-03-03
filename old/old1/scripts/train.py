import os
import torch
import torch.optim as optim
import torch.nn as nn
import geomloss
import pickle
import argparse
import matplotlib.pyplot as plt
from scripts.VanillaRNN import VanillaRNN
from scripts.config import *

# Define the train function
def train(batch_size, lr, tau, epochs, train_seq, val_seq, outputs=3, num_states=3, gumbel=True, init=True, title=None):
    # Determine device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model
    model = VanillaRNN(input_size, hidden_size, num_layers, outputs, biased=[False, False]).to(device)

    criterion = geomloss.SamplesLoss(blur=0.3)  # Sinkhorn Divergence
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9697, 0.9230), eps=2.6e-07, weight_decay=1.9e-07)

    train_losses, val_losses = [], []
    min_loss, best_model = float('inf'), None

    for epoch in range(epochs):
        train_shape = (train_seq.shape[0], train_seq.shape[1], 100)
        val_shape   = (val_seq.shape[0], val_seq.shape[1], 100)
        train_noise = torch.normal(mean=0, std=1, size=train_shape)
        val_noise   = torch.normal(mean=0, std=1, size=val_shape)

        model.train()
        train_loss, val_loss = [], []

        for i in range(0, len(train_seq), batch_size):
            batch_end_index = min(i + batch_size, len(train_seq))
            train_inputs = train_noise[i:batch_end_index].float().to(device)
            train_targets = train_seq[i:batch_end_index].float().to(device)

            optimizer.zero_grad()
            train_outputs, _, _ = model(train_inputs, tau, init=init, gumbel=gumbel)

            # Reshape the outputs and targets for the loss computation
            train_outputs = train_outputs.reshape(train_outputs.shape[0], -1)
            train_targets = train_targets.reshape(train_targets.shape[0], -1)

            loss = criterion(train_outputs, train_targets)
            train_loss.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        avg_train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_seq), batch_size):
                batch_end_index = min(i + batch_size, len(val_seq))
                val_inputs = val_noise[i:batch_end_index].float().to(device)
                val_targets = val_seq[i:batch_end_index].float().to(device)

                val_outputs, _, _ = model(val_inputs, tau, init=init, gumbel=gumbel)

                # Reshape the outputs and targets for the loss computation
                val_outputs = val_outputs.reshape(val_outputs.shape[0], -1)
                val_targets = val_targets.reshape(val_targets.shape[0], -1)

                loss = criterion(val_outputs, val_targets)
                val_loss.append(loss.item())

        avg_val_loss = sum(val_loss) / len(val_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            best_model = model.state_dict()
            model_name = f'{num_states}HMM_{outputs}Outputs_{len(train_seq) // 1000}kData_{lr}lr_{round(min_loss,1)}Loss_{title}.pth'

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], lr:{round(current_lr,4)} Tau:{round(tau,1)}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

    return train_losses, val_losses, best_model, model_name, min_loss

# Define a separate function to plot losses
def plot_losses(train_losses, val_losses, title="Training and Validation Loss Curves", path=""):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path + "figs/loss_curves.pdf")
    plt.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a VanillaRNN model using sequences from a pickle file.")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the .pkl file containing the training and validation sequences.")
    args = parser.parse_args()

    # Load the data from the .pkl file
    with open(args.pkl_file, "rb") as f:
        data = pickle.load(f)

    train_seq = torch.tensor(data["train_seq"])
    val_seq = torch.tensor(data["val_seq"])

    # Train the model and get the loss history
    train_losses, val_losses, best_model, model_name, _ = train(
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

    if best_model is not None:
        model_file = "models/" + model_name
        torch.save(best_model, model_file)
        print(f'Model saved to {model_file}')

    # Call the function to plot training and validation loss curves
    plot_losses(train_losses, val_losses)
