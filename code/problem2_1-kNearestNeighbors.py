#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################
import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import sys

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

def calc_kernel(x1,x2,W):
    diff = np.subtract(x2,x1)
    return np.exp(-1*diff.T @ W @ diff)

def predict_kernel(alpha=0.1):
    # TODO: your code here
    W = alpha * np.array([[1., 0.], [0., 1.]])
    y_pred = []
    for i,x1 in enumerate(X):
        numerator = 0
        denominator = 0
        for j,x2 in enumerate(X):
            if i != j:
                kernel = calc_kernel(x1,x2,W)
                denominator += kernel
                numerator += y[j]*kernel
            else:
                pass
        y_pred.append(numerator/denominator)
    return y_pred

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    y_pred = []
    W = np.array([[1., 0.], [0., 1.]])
    for i,x1 in enumerate(X):
        distances = []
        for j,x2 in enumerate(X):
            if i == j:
                distances.append(sys.float_info.max)
            else:
                distances.append(calc_kernel(x1,x2,W))
        y_pred.append(sum(y[np.argsort(distances)[::-1][1:k+1]])/k)
    return y_pred

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)
    y_pred = predict_kernel(alpha)
    loss = sum((y-y_pred)**2)
    print('The total least squares loss for alpha = ' + str(alpha) + ' is: ' + str(loss))
    

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
    y_pred = predict_knn(k)
    loss = sum((y-y_pred)**2)
    print('The total least squares loss for k = ' + str(k) + ' is: ' + str(loss))
