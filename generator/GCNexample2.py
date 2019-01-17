from gcnfile.reader import *
import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from gcnfile.GCN import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)

def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    A = tuple(A)
    # print("size",size)
    # Get the degrees for each node
    degrees = []
    for node_adjaceny in A:
        num = 0
        for node in node_adjaceny:
            if node == 1.0:
                num = num + 1
        # Add an extra for the "self loop"
        num = num + 1
        degrees.append(num)
    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(degrees)
    # Cholesky decomposition of D
    D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Turn adjacency matrix into a numpy matrix
    A = np.matrix(A)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    return A_hat, D

def accuracy(output, yorigin):
    # print(output.argmax(1),yorigin)
    return (output.argmax(1).type(torch.cuda.LongTensor) == yorigin.type(torch.cuda.LongTensor)).type(torch.float).mean().item()

def evaluate(idx):
    model.eval()
    output = model(x, A)
    loss = F.cross_entropy(output, y.type(torch.cuda.LongTensor))
    return loss, accuracy(output[idx], y[idx])

def step(y):
    model.train()
    optimizer.zero_grad()
    output = model(x, A)
    # print("q",output,"p",y)
    # acc = accuracy(output, yorigin)
    acc = 0
    mse = torch.nn.MSELoss()
    # loss = mse(output, y)
    loss = F.cross_entropy(output, y.type(torch.cuda.LongTensor))
    loss.backward()
    optimizer.step()
    return loss.item(), acc

x , y = read_data('karate.data', 'label.data',current_dir)
print("^^^^^",x)

A , D = preprocess(x)

# turn 1 2 to be 0 1
for i in range(len(y)):
    y[i] = y[i][0]-1
embedding = nn.Embedding(2, 2)
y = torch.tensor(y).type(torch.cuda.LongTensor)
# y = embedding(torch.tensor(y).type(torch.LongTensor))
# print("verify",y)
FloatTensor = torch.cuda.FloatTensor

# Turn the input and output into FloatTensors for the Neural Network
x = torch.tensor(x).to(device)
    # Variable(FloatTensor(x), requires_grad=False)
y = torch.tensor(y).to(device)
y = y.type(torch.cuda.FloatTensor)
    # Variable(FloatTensor(y), requires_grad=False)
A = torch.from_numpy(A)
A = A.type(torch.cuda.FloatTensor)
A = torch.tensor(A)
    # Variable(A, requires_grad=False)
D = torch.from_numpy(D)
D = D.type(torch.cuda.FloatTensor)
D = D.type(torch.cuda.FloatTensor)
    # Variable(D, requires_grad=False)

# Create random tensor weights
# W1 = torch.randn(34, 100).type(FloatTensor)
# W1.requires_grad_(True)
# W2 = torch.randn(100, 2).type(FloatTensor)
# W2.requires_grad_(True)
# softmax = nn.LogSoftmax(dim = 1)
# softmax.requires_grad_(True)
lr = 1e-3
model = GCN(34,2, hidden=[16, 32, 16], dropouts=[0.5, 0, 0.5]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)

# learning_rate = 1e-6
# for t in range(100000):
#
#     hidden_layer_1 = F.relu(D.mm(A).mm(D).mm(x).mm(W1))
#     hidden_layer_2 = F.relu(D.mm(A).mm(D).mm(hidden_layer_1).mm(W2))
#     y_pred = softmax(F.relu(D.mm(A).mm(D).mm(hidden_layer_2)))
#     topv, topi = y_pred.topk(1)
#     y_predresult = topi.squeeze().detach()
#     loss = (y_pred - y).pow(2).sum()
#     # print(y_predresult, y)
#     if t % 100 == 1 or t == 0:
#         print(t, loss.data)
#
#     loss.backward()

epochs = 10000
print_steps = 50
train_loss, train_acc = [], []
val_loss, val_acc = [], []
idx = torch.LongTensor(list(range(size)))
for i in range(epochs):
    tl, ta = step(y)
    train_loss += [tl]
    train_acc += [ta]
    if (i+1) % print_steps == 0 or i == 0:
        tl, ta = evaluate(idx)
        vl, va = evaluate(idx)
        val_loss += [vl]
        val_acc += [va]
        print(f'{i+1:6d}/{epochs}: train_loss={tl:.4f}, train_acc={ta:.4f}' +
              f', val_loss={vl:.4f}, val_acc={va:.4f}')

final_train = evaluate(x)
print(f'Train     : loss={final_train[0]:.4f}, accuracy={final_train[1]:.4f}')

