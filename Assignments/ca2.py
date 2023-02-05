from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import sys
import time

from datasets import get_data


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def gradient_single(X, y, w, lmbda):
    l1 = np.matmul(X, w) * y
    yx = y * X

    return 2 * lmbda * w - sigmoid(-l1) * yx

# Gradient of logistic function
def f_gradient(X, y, w):
    yX = y[:, np.newaxis] * X
    l1 = np.matmul(yX, w)
    return -sigmoid(-l1)[:, np.newaxis] * yX

# Ridge regularizer gradient
def reg_gradient(w, lmbda):
    return 2 * lmbda * w

# Full gradient of the logistic ridge regression with average of the logistic 
# gradient    
def gradient(X, y, w, lmbda):
    return reg_gradient(w, lmbda) + np.mean(f_gradient(X, y, w), axis=0)

# Full gradient without the average
def gradient_full(X, y, w, lmbda):
    return reg_gradient(w, lmbda) + f_gradient(X, y, w)

# Cost function - logistic ridge regression
def cost(X, y, w, lmbda):

    l1 = np.matmul(X, w) * y
    return np.mean(np.log(1 + np.exp(-l1)), axis=0) + \
        lmbda * np.linalg.norm(w, 2)**2

# Random sampler function useful to SGD and SVRG
def random_sampler(N, batch=1, buffersize=10000):
    """
    A generator of random indices from 0 to N.
    params:
    N: upper bound of the indices
    batch: Number of indices to return per iteration
    buffersize: Number of numbers to generate per batch
                (this is only a computational nicety)
    """

    S = int(np.ceil(buffersize / batch))

    while True:
        buffer = np.random.randint(N, size=(S, batch))
        for i in range(S):
            yield buffer[i]

# Stochastic Gradient Descent function
def SGD(X, y, w, lmbda, learning_rate, batch_size=1):
    N, D = X.shape
    sampler = random_sampler(N, batch_size)
    
    # The foor loop iterates over the random samples, not the entire vector
    for ix in sampler:
        grad = gradient(X[ix], y[ix], w, lmbda)
        w = w - learning_rate * grad
        yield w

# Stochastic Variance Reduction Gradient function
def SVRG(X, y, w, lmbda, learning_rate, T=5):
    """
    Stochastic variance reduced gradient
    """

    sampler = random_sampler(X.shape[0], T)

    for batch in sampler:

        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        for ix in batch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)
            w = w - learning_rate * (grad - full_grad[ix] + mean_grad)

        yield w

# Stochastic Average Gradient function
def SAG(X, y, w, lmbda, learning_rate, batch_size=1):
    N, D = X.shape
    P, = w.shape
    sampler = random_sampler(N, batch_size)

    grad = np.zeros((N, P))
    delta = np.zeros(P)
    non_zero_v = np.zeros(N)
    m = 0
    for ix in sampler:
        # update the number of seen examples m
        m -= np.sum(non_zero_v[ix], axis=0)
        non_zero_v[ix] = 1
        m += np.sum(non_zero_v[ix], axis=0)

        # update the sum of the gradient
        delta -= np.sum(grad[ix], axis=0)
        grad[ix] = f_gradient(X[ix], y[ix], w)
        delta += np.sum(grad[ix], axis=0)

        reg = reg_gradient(w, lmbda)

        w = w - learning_rate * (delta / m + reg)
        yield w


def initialize_w(N):
    return np.random.randn(N)


def gradient_descent(X, y, w, learning_rate=0.1, lmbda=0.01, iterations=1000):

    for iteration in range(iterations):

        grad = gradient(X, y, w, lmbda)
        w = w - learning_rate * grad

        yield w


def iterate(opt, w_0, training_loss,
            testing_loss, iterations=100, inner=1, name="NoName"):
    """
    This function takes an optimizer and returns a loss history for the
    training and test sets.
    """

    loss_hist_train = [training_loss(w_0)]
    loss_hist_test = [testing_loss(w_0)]

    ws = [w_0]
    clock = [0]

    start = time.time()
    for iteration in range(iterations):
        for _ in range(inner):
            w = next(opt)
        clock.append(time.time() - start)
        ws.append(w)

    #for iteration, w in enumerate(ws):
        train_loss = training_loss(w)
        loss_hist_train.append(train_loss)

        test_loss = testing_loss(w)
        loss_hist_test.append(test_loss)

        print("{}; {}; {}; {}".format(name, iteration, train_loss, test_loss))
        sys.stdout.flush()

    return loss_hist_train, loss_hist_test, clock


def main():

    # Should be a hyperparameter that we tune, not an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda', type=float, default=0.1, dest='lmbda')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=100)

    args = parser.parse_args()

    # Prepare dataset

    X, y = get_data("power")
    print("X,y types: {} {}".format(type(X), type(y)))
    print("X size {}".format(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    # Create a categorical variable from one of the columns.

    idx = y_train >= 0
    notidx = y_train < 0
    y_train[idx] = 1
    y_train[notidx] = -1

    # Initialize weights
    w_0 = initialize_w(X.shape[1])

    # BEGIN testing code for optimizers

    def training_loss(w):
        return cost(X_train, y_train, w, args.lmbda)

    def testing_loss(w):
        return cost(X_test, y_test, w, args.lmbda)

    iterations = args.iterations

    fig, ax = plt.subplots(2, 1, figsize=(16, 8))

    optimizers = [
        {
            "opt": SGD(X_train, y_train, w_0, args.lmbda, args.lr, 1),
            "name": "SGD",
            "inner": 5
        },
        {
            "opt": SAG(X_train, y_train, w_0, args.lmbda, args.lr),
            "name": "SAG",
            "inner": 1
        },
        {
            "opt": SVRG(X_train, y_train, w_0, args.lmbda, args.lr),
            "name": "SVRG",
            "inner": 1
        },
        {
            "opt": gradient_descent(
                X_train, y_train, w_0, learning_rate=args.lr,
                lmbda=args.lmbda, iterations=iterations),
            "name": "GD",
            "inner": 1
        },
    ]

    for opt in optimizers:

        # training_loss and testing_loss includes references to the training
        # and test set respectively.

        loss_hist_train, loss_hist_test, clock = iterate(
            opt['opt'],
            w_0,
            training_loss, testing_loss, iterations, inner=opt['inner'],
            name=opt['name'])

        color = next(ax[0]._get_lines.prop_cycler)['color']

        iterations_axis = range(0, iterations + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
                   label="Train loss ({})".format(opt['name']), linestyle="-",
                   color=color)

        ax[0].plot(iterations_axis, loss_hist_test,
                   label="Test loss ({})".format(opt['name']), linestyle="--",
                   color=color)

        ax[1].plot(clock, loss_hist_train,
                   label="Train loss ({})".format(opt['name']), linestyle="-",
                   color=color)

        ax[1].plot(clock, loss_hist_test,
                   label="Test loss ({})".format(opt['name']), linestyle="--",
                   color=color)

    ax[0].legend(loc="upper right")
    ax[0].set_xlabel(r"Iteration", fontsize=16)
    ax[0].set_ylabel("Loss", fontsize=16)
    ax[0].set_title("CA2 - Deterministic/stochastic algorithms in practice")
    ax[0].set_ylim(ymin=0)

    ax[1].legend(loc="upper right")
    ax[1].set_xlabel(r"Time [s]", fontsize=16)
    ax[1].set_ylabel("Loss", fontsize=16)
    ax[1].set_ylim(ymin=0)

    plt.savefig("ca2.png")


if __name__ == "__main__":
    main()
