from torch import nn
import torch.nn.functional as F
import torch
class GraphConvolution(nn.Module):
    def __init__(self, f_in, f_out, use_bias=True, activaiton = F.relu_):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_bias = use_bias
        self.activation = activaiton
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))
        self.bias = nn.Parameter(torch.FloatTensor(f_out)) if use_bias else None
        self.initialize_weight()

    def initialize_weight(self):
        if self.activation is None : nn.init.xavier_uniform_(self.weight)
        else: nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.use_bias: nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # print(input.size(),self.weight.size())
        support =  torch.mm(input,adj)
        output = torch.mm(support,self.weight)
        if self.use_bias: output = output.add_(self.bias)
        if self.activation is not None: output = self.activation(output)
        return output

class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden=[16], dropouts = [0.5],embedding = None, embedded = False):
        super().__init__()
        layers = []
        for f_in, f_out in zip([input_size]+hidden[:-1], hidden):
            # print("in",f_in,"out",f_out)
            layers += [GraphConvolution(f_in, f_out)]
        self.layers = nn.Sequential(*layers)
        self.dropouts = dropouts
        self.out_layer = GraphConvolution(f_out, output_size, activaiton= None)
        if embedded != True:
            self.embedding = nn.Embedding(input_size, output_size)
        else:
            self.embedding = embedding

    def forward(self, x, adj):
        x = x
        # x = self.embedding(x).view(1, 1, -1)
        for layer, d in zip(self.layers, self.dropouts):
            x = layer(x.type(torch.cuda.FloatTensor), adj)
            if d > 0: F.dropout(x, d, training=self.training, inplace=True)
        return self.out_layer(x, adj)
