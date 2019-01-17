import torch
import torch.nn as nn
import torch.nn.functional as F
MAX_LENGTH = 397
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

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, 1, -1)
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
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        self.word_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, 1, -1)
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
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class WordAttnDecoderRNN(nn.Module):
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
        self.word_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, 1, -1)
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
    def __init__(self, hidden_size, output_size, dropout = 0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.embedding_dropout(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# class CopyNetDecoder(nn.Module):
#     def __init__(self, hidden_size, embedding_size, max_length):
#         super(CopyNetDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding_size = embedding_size
#         self.max_length = max_length
#         # self.embedding = nn.Embedding(len(self.lang.tok_to_idx), self.embedding_size, padding_idx=0)
#         self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
#         self.embedding.weight.data[0, :] = 0.0
#
#         self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
#         self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)
#
#         self.gru = nn.GRU(2 * self.hidden_size + self.embedding.embedding_dim, self.hidden_size, batch_first=True)  # input = (context + selective read size + embedding)
#         self.out = nn.Linear(self.hidden_size, len(self.lang.tok_to_idx))
#
#     def forward(self, encoder_outputs, inputs, final_encoder_hidden, targets=None, keep_prob=1.0, teacher_forcing=0.0):
#         batch_size = encoder_outputs.data.shape[0]
#         seq_length = encoder_outputs.data.shape[1]
#
#         hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
#         # if next(self.parameters()).is_cuda:
#             hidden = hidden.cuda()
#         # else:
#         #     hidden = hidden
#
#         # every decoder output seq starts with <SOS>
#         sos_output = Variable(torch.zeros((batch_size, self.embedding.num_embeddings + seq_length)))
#         sampled_idx = Variable(torch.ones((batch_size, 1)).long())
#         if next(self.parameters()).is_cuda:
#             sos_output = sos_output.cuda()
#             sampled_idx = sampled_idx.cuda()
#
#         sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding
#
#         decoder_outputs = [sos_output]
#         sampled_idxs = [sampled_idx]
#
#         if keep_prob < 1.0:
#             dropout_mask = (Variable(torch.rand(batch_size, 1, 2 * self.hidden_size + self.embedding.embedding_dim)) < keep_prob).float() / keep_prob
#         else:
#             dropout_mask = None
#
#         selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
#         one_hot_input_seq = to_one_hot(inputs, len(self.lang.tok_to_idx) + seq_length)
#         if next(self.parameters()).is_cuda:
#             selective_read = selective_read.cuda()
#             one_hot_input_seq = one_hot_input_seq.cuda()
#
#         for step_idx in range(1, self.max_length):
#
#             if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
#                 # replace some inputs with the targets (i.e. teacher forcing)
#                 teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
#                 if next(self.parameters()).is_cuda:
#                     teacher_forcing_mask = teacher_forcing_mask.cuda()
#                 sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])
#
#             sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)
#
#             decoder_outputs.append(output)
#             sampled_idxs.append(sampled_idx)
#
#         decoder_outputs = torch.stack(decoder_outputs, dim=1)
#         sampled_idxs = torch.stack(sampled_idxs, dim=1)
#
#         return decoder_outputs, sampled_idxs

    # def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read, one_hot_input_seq, dropout_mask=None):
    #
    #     batch_size = encoder_outputs.shape[0]
    #     seq_length = encoder_outputs.shape[1]
    #     vocab_size = len(self.lang.tok_to_idx)
    #
    #     # Attention mechanism
    #     transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
    #     attn_scores = torch.bmm(encoder_outputs, transformed_hidden)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
    #     attn_weights = F.softmax(attn_scores, dim=1)  # apply softmax to scores to get normalized weights
    #     context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)  # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)
    #
    #     # Call the RNN
    #     out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
    #     unks = torch.ones_like(prev_idx).long() * 3
    #     prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding
    #     embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)
    #
    #     rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
    #     if dropout_mask is not None:
    #         if next(self.parameters()).is_cuda:
    #             dropout_mask = dropout_mask.cuda()
    #         rnn_input *= dropout_mask
    #
    #     self.gru.flatten_parameters()
    #     output, hidden = self.gru(rnn_input, prev_hidden)  # state.shape = [b, 1, hidden]
    #
    #     # Copy mechanism
    #     transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
    #     copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)  # this is linear. add activation function before multiplying.
    #     copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)  # [b, vocab_size + seq_length]
    #     missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)  # tokens not present in the input sequence
    #     missing_token_mask[:, 0] = 1  # <MSK> tokens are not part of any sequence
    #     copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)
    #
    #     # Generate mechanism
    #     gen_scores = self.out(output.squeeze(1))  # [b, vocab_size]
    #     gen_scores[:, 0] = -1000000.0  # penalize <MSK> tokens in generate mode too
    #
    #     # Combine results from copy and generate mechanisms
    #     combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
    #     probs = F.softmax(combined_scores, dim=1)
    #     gen_probs = probs[:, :vocab_size]
    #
    #     gen_padding = Variable(torch.zeros(batch_size, seq_length))
    #     if next(self.parameters()).is_cuda:
    #         gen_padding = gen_padding.cuda()
    #     gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]
    #
    #     copy_probs = probs[:, vocab_size:]
    #
    #     final_probs = gen_probs + copy_probs
    #
    #     log_probs = torch.log(final_probs + 10**-10)
    #
    #     _, topi = log_probs.topk(1)
    #     sampled_idx = topi.view(batch_size, 1)
    #
    #     # Create selective read embedding for next time step
    #     reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
    #     pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
    #     selected_scores = pos_in_input_of_sampled_token * copy_score_seq
    #     selected_scores_norm = F.normalize(selected_scores, p=1)
    #
    #     selective_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)
    #
    #     return sampled_idx, log_probs, hidden, selective_read
