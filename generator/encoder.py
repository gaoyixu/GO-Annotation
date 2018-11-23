import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)
