<<<<<<< HEAD
# coding=UTF-8
=======
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
<<<<<<< HEAD
        # self.embedding = nn.Embedding(input_size, hidden_size)
=======
        self.embedding = nn.Embedding(input_size, hidden_size)
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
        self.gru = nn.GRU(
            hidden_size,
            hidden_size
           )

<<<<<<< HEAD
    def forward(self, embedded,hidden):
        # embedded = self.embedding(input_seq).view(1, 1, -1)
        # 残余问题：如何正确地将结果输入进去
        # print("1",embedded.view(1, 1, -1).size())
        # print("2",hidden.size())
        outputs, hidden = self.gru(embedded.view(1, 1, -1), hidden)
=======
    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        outputs, hidden = self.gru(embedded, hidden)
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
        return outputs, hidden

    def initHidden(self,device = device):
        return torch.zeros(1, 1, self.hidden_size, device = device)
