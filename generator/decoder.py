import torch
import torch.nn as nn
import torch.nn.functional as F
MAX_LENGTH = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Attn(nn.Module):
#     def __init__(self, method, hidden_size):
#         super(Attn, self).__init__()
#         self.method = method
#         if self.method not in ["dot", "general", "concat"]:
#             raise ValueError(self.method,
#                              "is not an appropriate attention method.")
#         self.hidden_size = hidden_size
#         if self.method == "general":
#             self.attn = nn.Linear(self.hidden_size, hidden_size)
#         elif self.method == "concat":
#             self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#             self.v = nn.Parameter(torch.FloatTensor(hidden_size))

#     def dot_score(self, hidden, encoder_output):
#         return torch.sum(hidden * energy, dim=2)

#     def general_score(self, hidden, encoder_output):
#         energy = self.attn(encoder_output)
#         return torch.sum(hidden * energy, dim=2)

#     def concat_score(self, hidden, encoder_output):
#         energy = self.attn(
#             torch.cat(
#                 hidden.expand(encoder_output.size(0), -1, -1), encoder_output),
#             2).tanh()
#         return torch.sum(self.v * energy, dim=2)

#     def forward(self, hidden, encoder_output):
#         if self.method == "general":
#             attn_energies = self.general_score(hidden, encoder_output)
#         elif self.method == "concat":
#             attn_energies = self.concat_score(hidden, encoder_output)
#         elif self.method == 'dot':
#             attn_energies = self.dot_score(hidden, encoder_output)
#         attn_energies = attn_energies.t()
#         return F.softmax(attn_energies, dim=1).unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size,
                 dropout=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.attn = Attn(attn_model, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.atten_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, hidden, encoder_output, device):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding_dropout(embedded)
        # rnn_output hidden = self.gru(embedded, last_hidden)
        # attn_weights = self.attn(rnn_output, encoder_output)
        if device == "cuda":
            encoder_output = encoder_output.type(torch.cuda.FloatTensor)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.atten_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# class GreedySearchDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(GreedySearchDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#     def forward(self, input_seq, input_length, max_length):
#         encoder_output, encoder_hidden = self.encoder(input_seq, input_length)
#         decoder_hidden = encoder_hidden[:decoder.n_layers]
#         decoder_input = torch.ones(1, 1, device = device, dtype = torch.long) * SOS_token
#         all_tokens = torch.zero([0], device = device, dtype = torch.long)
#         all_scores = torch.zero([0], device = device)
#         for _ in range(max_length):
#             decoder_output, decoder_hidden = torch.max(decoder_output, dim = 1)
#             all_tokens = torch.cat((all_tokens, decoder_input), dim =0)
#             all_scores = torch.cat((all_scores, decoder_scores), dim = 0)
#             decoder_input = torch.unsqueeze(decoder_input, 0)
#         return all_tokens, all_scores


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)