import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VanillaRNN model
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, outputs, biased=[False, False]):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True, bias=biased[0])
        self.fc = nn.Linear(hidden_size, outputs, bias=biased[1])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, tau, init=True, gumbel=True):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if init:
            h = torch.normal(mean=0, std=1, size=(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)
        else:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, h = self.rnn(x, h)
        hidden = out
        logits = self.fc(out)
        if gumbel:
            out = F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)
        else:
            out = self.softmax(logits)
        return out, hidden, logits
