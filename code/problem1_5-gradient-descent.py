#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import math
import matplotlib.pyplot as plt


data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])


def K(W, x_i, x_j):
    x_i = np.array(x_i)
    x_j = np.array(x_j)
    return np.exp(np.dot(np.dot(-(x_i - x_j), W), (x_i - x_j)))


def compute_loss(W):
    loss = 0.
    for i in range(len(data)):
        x_i = list(data[i][:2])

        num = 0.
        for j in range(len(data)):
            if i != j:
                x_j = list(data[j][:2])
                num += K(W, x_i, x_j) * data[j][2]

        denom = 0.
        for j in range(len(data)):
            if i != j:
                x_j = list(data[j][:2])
                denom += K(W, x_i, x_j)

        loss += (data[i][2] - (num / denom)) ** 2.

    return loss

def compute_gradient(W):
    N = len(data)

    
    e_store = np.zeros((N, N))
    a_store = np.zeros((N, N))
    b_store = np.zeros((N, N))

            

    grad_11 = 0.
    grad_12 = 0.
    grad_22 = 0.

    for i in range(N):
        sum_eij = 0.
        sum_aij2_eij_yj = 0.
        sum_eij_yj = 0.
        sum_aij2_eij = 0.

        sum_aij_bij_eij_yj = 0.
        sum_aij_bij_eij = 0.

        sum_bij2_eij_yj = 0.
        sum_bij2_eij = 0.

        for j in range(N):
            if i != j:
                a_store[i][j] = data[i][0] - data[j][0]
                b_store[i][j] = data[i][1] - data[j][1]
                e_store[i][j] = math.exp(-(a_store[i][j]**2) * W[0][0] - 2 * a_store[i][j] * b_store[i][j] * W[0][1] - (b_store[i][j] ** 2) * W[1][1])

                sum_eij += e_store[i][j]
                sum_aij2_eij_yj += (a_store[i][j] ** 2) * e_store[i][j] * data[j][2]
                sum_eij_yj += e_store[i][j] * data[j][2]
                sum_aij2_eij += (a_store[i][j] ** 2) * e_store[i][j]

                sum_aij_bij_eij_yj = a_store[i][j] * b_store[i][j] * e_store[i][j] * data[j][2]
                sum_aij_bij_eij = a_store[i][j] * b_store[i][j] * e_store[i][j]

                sum_bij2_eij_yj = (b_store[i][j] ** 2) * e_store[i][j] * data[j][2]
                sum_bij2_eij = (b_store[i][j] ** 2) * e_store[i][j]

        grad_11 += (data[i][2] - sum_eij_yj/sum_eij) * (sum_aij2_eij_yj * sum_eij - sum_eij_yj * sum_aij2_eij)/(sum_eij ** 2) 
        grad_12 += (data[i][2] - sum_eij_yj/sum_eij) * (sum_aij_bij_eij_yj * sum_eij - sum_eij_yj * sum_aij_bij_eij)/(sum_eij ** 2) 
        grad_22 += (data[i][2] - sum_eij_yj/sum_eij) * (sum_bij2_eij_yj * sum_eij - sum_eij_yj * sum_bij2_eij)/(sum_eij ** 2)

    grad_11 *= 2.
    grad_12 *= 4.
    grad_22 *= 2.

    return grad_11, grad_12, grad_22

W = W1
MAX_EPOCHS = 5000
lr = 5
loss_history = np.zeros(MAX_EPOCHS)

for epoch in range(MAX_EPOCHS):
    l = compute_loss(W)
    loss_history[epoch] = l
    print("loss:", l)
    print("W:", W)

    g_11, g_12, g_22 = compute_gradient(W)
    g_mat = np.array([[g_11, g_12], [g_12, g_22]])
    print("gradient mat:", g_mat)

    W -= lr * g_mat


print(W)
plt.plot(loss_history)
plt.show()
plt.savefig("loss_history_gd.png")




