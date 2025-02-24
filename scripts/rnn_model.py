import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class RNNModel(nn.Module):
    """
    A recurrent neural network model for sequence modeling.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, biased=[False, False]):
        """
        Initialize the RNN model.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden layer
            num_layers (int): Number of RNN layers
            output_size (int): Number of output classes
            biased (list): Whether to use bias in RNN and linear layers
        """
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', 
                         batch_first=True, bias=biased[0])
        self.fc = nn.Linear(hidden_size, output_size, bias=biased[1])
        self.softmax = nn.Softmax(dim=2)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.best_model = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x, tau=1.0, init=True, gumbel=True):
        """
        Forward pass through the RNN.
        
        Args:
            x (torch.Tensor): Input tensor
            tau (float): Temperature for Gumbel-Softmax
            init (bool): Whether to initialize hidden state with noise
            gumbel (bool): Whether to use Gumbel-Softmax
            
        Returns:
            tuple: (outputs, hidden_states, logits)
        """
        # Initialize hidden state
        if init:
            h = torch.normal(mean=0, std=1, size=(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        else:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            
        # Forward propagation
        out, h = self.rnn(x, h)
        hidden = out
        logits = self.fc(out)
        
        # Apply activation
        if gumbel:
            out = F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)
        else:
            out = self.softmax(logits)
            
        return out, hidden, logits
    
    def train_model(self, train_seq, val_seq, batch_size=4096, lr=0.001, tau=1.0, 
                   epochs=1000, grad_clip=0.9, init=True, criterion=None):
        """
        Train the RNN model.
        
        Args:
            train_seq (torch.Tensor): Training sequences
            val_seq (torch.Tensor): Validation sequences
            batch_size (int): Batch size
            lr (float): Learning rate
            tau (float): Temperature for Gumbel-Softmax
            epochs (int): Number of training epochs
            grad_clip (float): Gradient clipping threshold
            init (bool): Whether to initialize hidden state with noise
            criterion: Loss function (if None, uses MSE)
            
        Returns:
            tuple: (train_losses, val_losses, best_model, best_loss)
        """
        print(f"Training on device: {self.device}")
        
        # Move data to device
        train_seq = train_seq.to(self.device)
        val_seq = val_seq.to(self.device)
        
        # Define loss and optimizer
        if criterion is None:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
        # For each epoch
        for epoch in range(epochs):
            # Generate random noise as inputs
            train_shape = (train_seq.shape[0], train_seq.shape[1], self.input_size)
            val_shape = (val_seq.shape[0], val_seq.shape[1], self.input_size)
            train_noise = torch.normal(mean=0, std=1, size=train_shape).to(self.device)
            val_noise = torch.normal(mean=0, std=1, size=val_shape).to(self.device)
            
            # Training phase
            self.train()
            train_loss = []
            
            for i in range(0, len(train_seq), batch_size):
                batch_end = min(i + batch_size, len(train_seq))
                inputs = train_noise[i:batch_end].float()
                targets = train_seq[i:batch_end].float()
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _, _ = self(inputs, tau, init=init)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(outputs.shape[0], -1)
                targets = targets.reshape(targets.shape[0], -1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                train_loss.append(loss.item())
                
                # Backward pass and optimize
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
            
            # Calculate average training loss
            avg_train_loss = sum(train_loss) / len(train_loss)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            val_loss = []
            
            with torch.no_grad():
                for i in range(0, len(val_seq), batch_size):
                    batch_end = min(i + batch_size, len(val_seq))
                    inputs = val_noise[i:batch_end].float()
                    targets = val_seq[i:batch_end].float()
                    
                    # Forward pass
                    outputs, _, _ = self(inputs, tau, init=init)
                    
                    # Reshape for loss calculation
                    outputs = outputs.reshape(outputs.shape[0], -1)
                    targets = targets.reshape(targets.shape[0], -1)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    val_loss.append(loss.item())
            
            # Calculate average validation loss
            avg_val_loss = sum(val_loss) / len(val_loss)
            self.val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model = self.state_dict()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return self.train_losses, self.val_losses, self.best_model, self.best_loss
    
    def generate_sequences(self, time_steps, dynamics_mode="full"):
        """
        Generate sequences from the trained RNN model.
        
        Args:
            time_steps (int): Number of time steps to generate
            dynamics_mode (str): Dynamics mode ('full', 'recurrence_only', 'input_only')
            
        Returns:
            dict: Dictionary containing hidden states, logits, and outputs
        """
        self.eval()
        
        # Initialize tensors
        h = torch.zeros((time_steps, self.hidden_size)).to(self.device)
        logits = torch.zeros((time_steps, self.output_size)).to(self.device)
        outs = torch.zeros((time_steps, self.output_size)).to(self.device)
        h[0] = torch.zeros((self.hidden_size)).to(self.device)
        
        # Extract weights
        ih = self.rnn.weight_ih_l0.data
        hh = self.rnn.weight_hh_l0.data
        fc = self.fc.weight.data
        
        # Generate hidden states, logits, and outputs
        for t in range(1, time_steps):
            x = torch.normal(mean=0, std=1, size=(self.input_size,)).float().to(self.device)
            
            if dynamics_mode == 'full':
                h[t] = torch.relu(x @ ih.T + h[t-1] @ hh.T)
            elif dynamics_mode == 'recurrence_only':
                h[t] = torch.relu(h[t-1] @ hh.T)
            elif dynamics_mode == 'input_only':
                identity_matrix = torch.eye(self.hidden_size, device=self.device)
                h[t] = torch.relu(x @ ih.T + h[t-1] @ identity_matrix)
            else:
                raise ValueError("Invalid dynamics_mode. Choose from 'full', 'recurrence_only', 'input_only'.")
        
        logits = h @ fc.T
        outs = F.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1)
        
        # Convert tensors to numpy arrays
        h_np = h.cpu().detach().numpy()
        logits_np = logits.cpu().detach().numpy()
        outs_np = outs.cpu().detach().numpy()
        
        return {
            "h": h_np,
            "logits": logits_np,
            "outs": outs_np
        }
    
    def save_model(self, path):
        """Save the model to the specified path"""
        torch.save(self.best_model if hasattr(self, 'best_model') and self.best_model is not None 
                   else self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from the specified path"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {path}")
    
    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
