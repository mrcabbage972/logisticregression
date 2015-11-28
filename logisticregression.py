import numpy as np
from optimization import GradientDescent

BOUND = 20.0


def sigmoid(b, x):
    return 1.0 / (1.0 + np.exp(np.minimum(BOUND, np.maximum(-BOUND, -np.dot(b, x)))))


def lr_loss_gradient(x, y, b, reg_param=0.0):
    grad = (np.dot(np.transpose((sigmoid(x, np.transpose(b)))), x) - np.dot(np.transpose(x), y))[0, :]
    grad += 2.0 * reg_param * b[0, :]
    return grad


def lr_loss(X, y, b, reg_param=0.0):
    np.seterr(all='raise')

    loss = -np.sum(np.multiply(np.log(sigmoid(b, np.transpose(X))), np.transpose(y))
                   + np.multiply(np.log(1.0 - sigmoid(b, np.transpose(X))), 1.0 - np.transpose(y)))

    reg = reg_param * np.dot(b, np.transpose(b))
    loss += reg[0]
    return loss[0]


class LogisticRegression:
    def __init__(self, reg_param=0.0, is_verbose=False):
        self.reg_param = reg_param
        self.is_verbose = is_verbose

    def fit(self, X, y):
        opt_loss_func = lambda b: lr_loss(X, y, b, self.reg_param)
        opt_grad_loss_func = lambda b: lr_loss_gradient(X, y, b, self.reg_param)

        gd = GradientDescent(is_verbose=self.is_verbose)

        self.coef = gd.fit(opt_loss_func, opt_grad_loss_func, np.zeros((1, X.shape[1])))

    def predict(self, X):
        return sigmoid(self.coef, np.transpose(X))[0]
