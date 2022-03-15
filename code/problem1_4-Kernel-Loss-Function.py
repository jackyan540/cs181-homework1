import numpy as np

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


def compute_loss(W):
    ## TO DO
    loss = 0
    for i,d1 in enumerate(data):
        numerator = 0
        denominator = 0
        for j,d2 in enumerate(data):
            if i != j:
                diff = np.subtract(d2[0:2],d1[0:2])
                kernel = np.exp(-1*diff.T @ W @ diff)
                denominator += kernel
                numerator += d2[2]*kernel
            else:
                pass
        loss += (d1[2]-numerator/denominator)**2
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))