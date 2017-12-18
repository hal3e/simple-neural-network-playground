import numpy as np

def sigmoid(x):
    "Numerically-stable Sigmoid function"
    out = np.copy(x)
    greater_than_zero = np.nonzero(x >= 0)
    out[greater_than_zero] = 1 / (1 + np.exp(-x[greater_than_zero]))

    smaller_than_zero = np.nonzero(x < 0)
    z = np.exp(x[smaller_than_zero])
    out[smaller_than_zero] = z / (1 + z)
    return out

def sigmoid_d(x):
    "Derivative of Sigmoid function"
    return x * (1 - x)


def relu(x):
    "Rectified Linear Unit function"
    x[np.nonzero(x < 0)] = 0
    return x

def relu_d(x):
    "Derivative of Rectified Linear Unit function"
    x[np.nonzero(x >= 0)] = 1
    x[np.nonzero(x < 0)] = 0
    return x


def tanh(x):
    "Hyperbolic Tangent function"
    return np.tanh(x)

def tanh_d(x):
    "Derivative of Hyperbolic Tangent function"
    return 1 - x ** 2


def softmax(X):
    "Numerically-stable softmax function"
    if X.ndim > 1:
        out = []
        for x in X:
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            out.append(exps / np.sum(exps))
        return np.array(out)
    else:
        shiftx = X - np.max(X)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
