from gcnfile.reader import *
import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from gcnfile.GCN import *
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

