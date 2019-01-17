import numpy as np
import torch
from scipy.sparse import csr_matrix,coo_matrix,diags,eye
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gcnfile.GCN import *
from tqdm import tnrange
import pandas as pd
import os

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

paper_feature_label = np.genfromtxt(current_dir + "\cora.content", dtype = np.str)

features = csr_matrix(paper_feature_label[:,1:-1], dtype = np.float32)

labels = paper_feature_label[:,-1]

lbl2idx = {k:v for v,k in enumerate(sorted(np.unique(labels)))}

labels = [lbl2idx[e] for e in labels]

papers = paper_feature_label[:,0].astype(np.int32)

paper2idx = {k:v for v,k in enumerate(papers)}
edges = np.genfromtxt(current_dir + "\cora.cities", dtype = np.int32)
edges = np.asarray(
    [paper2idx[e] for e in edges.flatten()],
    np.int32).reshape(edges.shape)

adj = coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
    shape=(len(labels), len(labels)), dtype=np.float32)
print("adj",adj.T.multiply(adj.T > adj))
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

def normalize(mx):
    rowSum = np.array(mx.sum(1))
    r_inv = (rowSum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

adj = normalize(adj + eye(adj.shape[0]))

adj = torch.FloatTensor(adj.todense())

features = torch.FloatTensor(features.todense())

labels = torch.LongTensor(labels)

n_train = 200
n_val = 200
n_test = len(features) - n_train - n_val
idxs = np.random.permutation(len(features))
idx_train = torch.LongTensor(idxs[:n_train])
idx_val = torch.LongTensor(idxs[n_train:n_train+n_val])
idx_test = torch.LongTensor(idxs[n_train+n_val:])

adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_test= idx_test.to(device)

def accuracy(output, y):
    return (output.argmax(1) == y).type(torch.float32).mean().item()

def step():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item(), acc

def evaluate(idx):
    model.eval()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx], labels[idx]).item()
    return loss, accuracy(output[idx], labels[idx])

n_labels = labels.max().item() + 1
n_features = features.shape[1]
# n_labels, n_features
lr = 1e-3
model = GCN(n_features, n_labels, hidden=[16, 32, 16], dropouts=[0.5, 0, 0.5]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)

epochs = 500
print_steps = 50
train_loss, train_acc = [], []
val_loss, val_acc = [], []
for i in tnrange(epochs):
    tl, ta = step()
    train_loss += [tl]
    train_acc += [ta]
    if (i+1) % print_steps == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]
        print(f'{i+1:6d}/{epochs}: train_loss={tl:.4f}, train_acc={ta:.4f}' +
              f', val_loss={vl:.4f}, val_acc={va:.4f}')

final_train, final_val, final_test = evaluate(idx_train), evaluate(idx_val), evaluate(idx_test)
print(f'Train     : loss={final_train[0]:.4f}, accuracy={final_train[1]:.4f}')
print(f'Validation: loss={final_val[0]:.4f}, accuracy={final_val[1]:.4f}')
print(f'Test      : loss={final_test[0]:.4f}, accuracy={final_test[1]:.4f}')
print()

fig, axes = plt.subplots(1, 2, figsize=(15,5))
ax = axes[0]
axes[0].plot(train_loss[::print_steps] + [train_loss[-1]], label='Train')
axes[0].plot(val_loss, label='Validation')
axes[1].plot(train_acc[::print_steps] + [train_acc[-1]], label='Train')
axes[1].plot(val_acc, label='Validation')
for ax,t in zip(axes, ['Loss', 'Accuracy']): ax.legend(), ax.set_title(t, size=15)

output = model(features, adj)
samples = 10
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]

idx2lbl = {v:k for k,v in lbl2idx.items()}
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
print(df)