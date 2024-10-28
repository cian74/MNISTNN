import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mnist_test.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

#print(X_train[:, 0].shape)

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10,10)
    b2 = np.random.rand(10, 1)
    return W1,b1,W2,b2

def ReLu(Z):
    return Z * (Z > 0)

def dReLu(Z):
    return 1 if Z > 0 else 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

