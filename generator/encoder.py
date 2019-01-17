import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_max_length = 30
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, embedding = None, embedded = False, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        if embedded != True:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )

    def forward(self, input_seq, hidden=None):
        # if self.input_size != self.hidden_size:
        embedded = self.embedding(input_seq).view(1, 1, -1)
        # else:
        #     embedded = input_seq.view(1, 1, -1)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden


    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class CombineEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(CombineEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )

    def forward(self, input_seq, hidden=None):
        embedded = input_seq
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class AttenCombineEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1,max_length=word_max_length):
        super(AttenCombineEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )
        self.embedding_dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.atten_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.word_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_seq, hidden,encoder_output):
        embedded = input_seq
        embedded = self.embedding_dropout(embedded)
        if device == "cuda":
            encoder_output = encoder_output.type(torch.cuda.FloatTensor)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.atten_combine(output).unsqueeze(0)
        output = F.relu(output)
        outputs, hidden = self.gru(output, hidden)
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class GCNEncoderRNN(nn.Module):
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

class doubleAttenCombineEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1,max_length=word_max_length):
        super(doubleAttenCombineEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )
        self.embedding_dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.atten_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.word_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.connected_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_seq1,input_seq2, hidden,encoder_output):
        embedded = torch.cat((input_seq1[0],input_seq2[0]),1)
        embedded = self.connected_layer(embedded)
        embedded = F.relu(embedded)
        embedded = self.embedding_dropout(embedded)
        if device == "cuda":
            encoder_output = encoder_output.type(torch.cuda.FloatTensor)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.atten_combine(output).unsqueeze(0)
        output = F.relu(output)
        outputs, hidden = self.gru(output, hidden)
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class doubleCombineEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(doubleCombineEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )
        self.connected_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_seq1,input_seq2, hidden=None):
        embedded = torch.cat((input_seq1[0], input_seq2[0]), 1)
        embedded = self.connected_layer(embedded)
        embedded = F.relu(embedded)
        embedded = self.embedding_dropout(embedded)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

# class MSA