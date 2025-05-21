import torch, geomloss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path 

class RNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=150, num_layers=1, output_size=3, biased=[False, False]):
        super(RNN, self).__init__()
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
                    epochs=1000, grad_clip=0.9, init=True, criterion=None, verbose=False,
                    save_interval=None, save_path=None):
        """
        Train the RNN model.
        """
        if verbose:
            print(f"Training on device: {self.device}")
        
        # Move data to device
        train_seq = train_seq.to(self.device)
        val_seq = val_seq.to(self.device)
        
        # Define loss and optimizer
        criterion = geomloss.SamplesLoss(blur=0.2)  # Sinkhorn Divergence
        optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.9))
        
        # Reset training history
        self.train_losses, self.val_losses = [], []
        self.best_loss = float('inf')
        
        # Training loop
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
                
                optimizer.zero_grad()
                outputs, _, _ = self(inputs, tau, init=init)
                
                outputs = outputs.reshape(outputs.shape[0], -1)
                targets = targets.reshape(targets.shape[0], -1)
                
                loss = criterion(outputs, targets)
                train_loss.append(loss.item())
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
            
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
                    
                    outputs, _, _ = self(inputs, tau, init=init)
                    
                    outputs = outputs.reshape(outputs.shape[0], -1)
                    targets = targets.reshape(targets.shape[0], -1)
                    
                    loss = criterion(outputs, targets)
                    val_loss.append(loss.item())
            
            avg_val_loss = sum(val_loss) / len(val_loss)
            self.val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model = self.state_dict()
            
            # Save intermediate model if conditions are met
            if save_interval is not None and (epoch == 0 or (epoch % save_interval == save_interval - 1)):
                model_path = Path(save_path) / f"model_epoch_{epoch+1}.pth"
                torch.save(self.state_dict(), model_path)
                if verbose:
                    print(f"Model saved at epoch {epoch+1} to {model_path}")
            
            # Print progress
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return self.train_losses, self.val_losses, self.best_model, self.best_loss
    
    def gen_seq(self, time_steps=None, dynamics_mode="full", batch_mode=False, num_seq=None, seq_len=None):
        """Generate sequences from the trained RNN model."""
        self.eval()
        
        # Extract weights
        ih = self.rnn.weight_ih_l0.data
        hh = self.rnn.weight_hh_l0.data
        fc = self.fc.weight.data
        
        if batch_mode:
            if num_seq is None or seq_len is None:
                raise ValueError("num_seq and seq_len must be provided when batch_mode=True")
            
            # Initialize tensors for batch mode
            h = torch.zeros((num_seq, seq_len, self.hidden_size)).to(self.device)
            logits = torch.zeros((num_seq, seq_len, self.output_size)).to(self.device)
            outs = torch.zeros((num_seq, seq_len, self.output_size)).to(self.device)
            
            # Generate sequences
            for i in range(num_seq):
                # Initialize first hidden state for this sequence
                h[i, 0] = torch.zeros((self.hidden_size)).to(self.device)
                
                # Generate the rest of the sequence
                for t in range(1, seq_len):
                    x = torch.normal(mean=0, std=1, size=(self.input_size,)).float().to(self.device)
                    if dynamics_mode == 'full':
                        h[i, t] = torch.relu(x @ ih.T + h[i, t-1] @ hh.T)
                    elif dynamics_mode == 'recurrence_only':
                        h[i, t] = torch.relu(h[i, t-1] @ hh.T)
                    elif dynamics_mode == 'input_only':
                        identity_matrix = torch.eye(self.hidden_size, device=self.device)
                        h[i, t] = torch.relu(x @ ih.T + h[i, t-1] @ identity_matrix)
                    else:
                        raise ValueError("Invalid dynamics_mode. Choose from 'full', 'recurrence_only', 'input_only'.")
                
                # Calculate logits and outputs for this sequence
                logits[i] = h[i] @ fc.T
                outs[i] = F.gumbel_softmax(logits[i], tau=1, hard=True, eps=1e-10, dim=-1)
        else:
            if time_steps is None:
                raise ValueError("time_steps must be provided when batch_mode=False")
                
            # Original continuous sequence generation
            h = torch.zeros((time_steps, self.hidden_size)).to(self.device)
            logits = torch.zeros((time_steps, self.output_size)).to(self.device)
            outs = torch.zeros((time_steps, self.output_size)).to(self.device)
            h[0] = torch.zeros((self.hidden_size)).to(self.device)
            
            # Generate hidden states
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
            
            # Calculate logits and outputs
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
        torch.save(self.best_model if hasattr(self, 'best_model') and self.best_model is not None 
                   else self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
        #print(f"Model loaded from {path}")
    
    def plot_losses(self, save_path=None, title_prefix=None):
        # Increase figure size to accommodate title
        plt.figure(figsize=(12, 8))
        
        # Use suptitle for the title_prefix to avoid overlap
        if title_prefix:
            plt.suptitle(title_prefix, fontsize=14, y=0.98)
            
        # Plot the losses
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        
        # Use a shorter title for the plot itself
        plt.title('Training and Validation Loss', pad=20)
            
        plt.legend()
        plt.grid(True)
        
        # Add padding to ensure no overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        plt.close()
