from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import sys
import time
import keras

from datasets import get_data


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def feed_forward_propagation(X, y, w_1, w_2, w_3, lmbda):

    N, D = X.shape


    # print("Size of X {}".format(X.shape))
    # print("Size of y {}".format(y.shape))


    layer_0 = X
    # print("Shape of layer_0 {}".format(layer_0))
    layer_1 = sigmoid(np.matmul(layer_0,w_1))
    # print("Shape of layer_1 {}".format(layer_1.shape))
    layer_2 = sigmoid(np.matmul(layer_1, w_2))
    # print("Shape of layer_2 {}".format(layer_2.shape))
    layer_3 = np.matmul(layer_2, w_3)
    # print("Shape of layer_3 {}".format(layer_3.shape))

    return layer_0, layer_1, layer_2, layer_3

def back_propagation(y, w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3):

    # print("Shape of Layer_0 {}".format(layer_0.shape))
    # print("Shape of Layer_1 {}".format(layer_1.shape))
    # print("Shape of Layer_2 {}".format(layer_2.shape))
    # print("Shape of Layer_3 {}".format(layer_3.shape))
    # #
    # print("Shape of W1 {}".format(w_1.shape))
    # print("Shape of W2 {}".format(w_2.shape))
    # print("Shape of W3 {}".format(w_3.shape))

    layer_3_error = (layer_3[:,0] - y)[:,np.newaxis]

    layer_3_delta = layer_3_error * sigmoid(layer_3, derivative=True)

    # print("Shape of Layer_3_delta {}".format(layer_3_delta.shape))

    layer_2_error = layer_3_delta.dot(w_3.T)

    layer_2_delta = layer_2_error * sigmoid(layer_2, derivative=True)

    # print("Shape of Layer_2_delta {}".format(layer_2_delta.shape))

    layer_1_error = layer_2_delta.dot(w_2.T)

    # print("Shape of layer_1_error {}".format(layer_1_error.shape))

    layer_1_delta = layer_1_error * sigmoid(layer_1, derivative=True)

    # print("Shape of Layer_1_delta {}".format(layer_1_delta.shape))

    return layer_1_delta, layer_2_delta, layer_3_delta

def cost(X, y, w_1, w_2, w_3, lmbda):
    N, d = X.shape
    a1,a2,a3,a4 = feed_forward_propagation(X,y,w_1,w_2,w_3,lmbda)

    return np.linalg.norm(a4[:,0] - y,2) ** 2 / N


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


def SGD(X, y, w_1, w_2, w_3, lmbda, learning_rate, batch_size=1):
    N, D = X.shape
    sampler = random_sampler(N, batch_size)


    for ix in sampler:

        #Feed Forward propagation
        layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X[ix], y[ix], w_1,w_2,w_3,lmbda)

        layer_1_delta, layer_2_delta, layer_3_delta = back_propagation(
                y[ix], w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)

        w_1 = w_1 - (1 / batch_size) * learning_rate * (layer_0.T.dot(layer_1_delta)) + (lmbda / batch_size * w_1)
        w_2 = w_2 - (1 / batch_size) * learning_rate * (layer_1.T.dot(layer_2_delta)) + (lmbda / batch_size * w_2)
        w_3 = w_3 - (1 / batch_size) * learning_rate * (layer_2.T.dot(layer_3_delta)) + (lmbda / batch_size * w_3)

        yield w_1, w_2, w_3


def SVRG(X, y, w_1, w_2, w_3, lmbda, learning_rate, T=5):
    """
    Stochastic variance reduced gradient
    """
    N, D = X.shape

    sampler = random_sampler(N, T)

    for batch in sampler:

        layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X, y, w_1,w_2,w_3,lmbda)

        # TODO: Full gradient (include regularizer)
        full_grad_1, full_grad_2, full_grad_3 = back_propagation(
                y, w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)



        mean_grad_1 = np.mean(full_grad_1, axis=0)
        mean_grad_2 = np.mean(full_grad_2, axis=0)
        mean_grad_3 = np.mean(full_grad_3, axis=0)

        for ix in batch:

            layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X[ix][np.newaxis,:], y[ix], w_1,w_2,w_3,lmbda)

            grad_1, grad_2, grad_3 = back_propagation(
                    y[ix], w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)

            w_1 = w_1 - learning_rate * layer_0.T.dot(grad_1 - full_grad_1[ix] + mean_grad_1)
            w_2 = w_2 - learning_rate * layer_1.T.dot(grad_2 - full_grad_2[ix] + mean_grad_2)
            w_3 = w_3 - learning_rate * layer_2.T.dot(grad_3 - full_grad_3[ix] + mean_grad_3)

        yield w_1,w_2,w_3


def initialize_w(N, d):
    return 2*np.random.random((N,d)) - 1


def GD(X, y, w_1,w_2,w_3, learning_rate=0.1, lmbda=0.01, iterations=1000):

    N = X.shape[0]

    for iteration in range(iterations):

        layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X, y, w_1,w_2,w_3,lmbda)

        layer_1_delta, layer_2_delta, layer_3_delta = back_propagation(
                y, w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)

        w_1 = w_1 - (1 / N) * learning_rate * (layer_0.T.dot(layer_1_delta)) + (lmbda / N * w_1)
        w_2 = w_2 - (1 / N) * learning_rate * (layer_1.T.dot(layer_2_delta)) + (lmbda / N * w_2)
        w_3 = w_3 - (1 / N) * learning_rate * (layer_2.T.dot(layer_3_delta)) + (lmbda / N * w_3)


        # print("Shape of W1 {}".format(w_1.shape))
        # print("Shape of W2 {}".format(w_2.shape))
        # print("Shape of W3 {}".format(w_3.shape))

        yield w_1,w_2,w_3

def PGD(X, y, w_1,w_2,w_3, learning_rate=0.1, lmbda=0.01, iterations=1000, noise=(0.9,1.1)):

    N = X.shape[0]

    for iteration in range(iterations):

        layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X, y, w_1,w_2,w_3,lmbda)

        layer_1_delta, layer_2_delta, layer_3_delta = back_propagation(
                y, w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)

        p = np.random.uniform(noise[0],noise[1])

        w_1 = w_1 - (p / N) * learning_rate * (layer_0.T.dot(layer_1_delta)) + (lmbda / N * w_1)
        w_2 = w_2 - (p / N) * learning_rate * (layer_1.T.dot(layer_2_delta)) + (lmbda / N * w_2)
        w_3 = w_3 - (p / N) * learning_rate * (layer_2.T.dot(layer_3_delta)) + (lmbda / N * w_3)

        yield w_1,w_2,w_3


def BCD(X, y, w_1,w_2,w_3, learning_rate=0.1, lmbda=0.01, iterations=1000):
    N, D = X.shape
    W_D = w_1.shape[1]


    for iteration in range(iterations):
        p = np.random.choice(W_D, replace=False, size=np.random.randint(W_D))
        # print(p)
        for i in p:

            layer_0, layer_1, layer_2, layer_3 = feed_forward_propagation(X, y, w_1,w_2,w_3,lmbda)

            layer_1_delta, layer_2_delta, layer_3_delta = back_propagation(
                y, w_1, w_2, w_3, layer_0, layer_1, layer_2, layer_3)

            # print(layer_0.T.dot(layer_1_delta).shape)

            w_1[:,i] = w_1[:,i] - (1 / N) * learning_rate * (layer_0.T.dot(layer_1_delta)[:,i]) + (lmbda / N * w_1[:,i])
            w_2[:,i] = w_2[:,i] - (1 / N) * learning_rate * (layer_1.T.dot(layer_2_delta)[:,i]) + (lmbda / N * w_2[:,i])
            w_3 = w_3 - (1 / N) * learning_rate * (layer_2.T.dot(layer_3_delta)) + (lmbda / N * w_3)

            yield w_1,w_2,w_3


def iterate(opt, w_1, w_2, w_3, training_loss,
            testing_loss, iterations=100, inner=1, name="NoName"):
    """
    This function takes an optimizer and returns a loss history for the
    training and test sets.
    """

    loss_hist_train = [training_loss(w_1,w_2,w_3)]
    loss_hist_test = [testing_loss(w_1,w_2,w_3)]

    ws = [w_1,w_2,w_3]
    clock = [0]

    start = time.time()
    for iteration in range(iterations):
        for _ in range(inner):
            w_1,w_2,w_3 = next(opt)
        clock.append(time.time() - start)
        ws.extend([w_1,w_2,w_3])


    #for iteration, w in enumerate(ws):
        train_loss = training_loss(w_1,w_2,w_3)
        loss_hist_train.append(train_loss)

        test_loss = testing_loss(w_1,w_2,w_3)
        loss_hist_test.append(test_loss)

        print("{}; {}; {}; {}".format(name, iteration, train_loss, test_loss))
        sys.stdout.flush()

    return loss_hist_train, loss_hist_test, clock


def main():

    # Should be a hyperparameter that we tune, not an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda', type=float, default=0.1, dest='lmbda')
    parser.add_argument('--w_size', type=int, default=10, dest='w_size')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=50)

    args = parser.parse_args()

    # Prepare dataset

    X_train, X_test, y_train, y_test, no_class = get_data("MNIST")
    #print("X,y types: {} {}".format(type(X), type(y)))
    #print("X size {}".format(X.shape))
    #print("Y size {}".format(y.shape))

    # Create a categorical variable from one of the columns.

    #idx = y_train >= 0
    #notidx = y_train < 0
    #y_train[idx] = 1
    #y_train[notidx] = -1



    # Initialize weights
    w_1 = initialize_w(X_train.shape[1], args.w_size)

    w_2 = initialize_w(args.w_size,args.w_size)

    w_3 = initialize_w(args.w_size, 1)


    #
    # BEGIN testing code for optimizers
    #
    def training_loss(w_1, w_2, w_3):
        return cost(X_train, y_train, w_1, w_2, w_3, args.lmbda)

    def testing_loss(w_1, w_2, w_3):
        return cost(X_test, y_test, w_1, w_2, w_3, args.lmbda)
    #
    iterations = args.iterations
    #
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    #
    optimizers = [
        {
            "opt": SGD(X_train, y_train, w_1, w_2, w_3, args.lmbda, args.lr, 100),
            "name": "SGD",
            "inner": 5
        },
        {
            "opt": SVRG(X_train, y_train, w_1, w_2, w_3, args.lmbda, args.lr),
            "name": "SVRG",
            "inner": 1
        },
        {
            "opt": GD(
                X_train, y_train, w_1, w_2, w_3, learning_rate=args.lr,
                lmbda=args.lmbda, iterations=iterations),
            "name": "GD",
            "inner": 1
        },
        {
            "opt": PGD(
                X_train, y_train, w_1, w_2, w_3, learning_rate=args.lr,
                lmbda=args.lmbda, iterations=iterations, noise=(0.9,1.1)),
            "name": "PGD",
            "inner": 1
        },
        {
            "opt": BCD(
                X_train, y_train, w_1, w_2, w_3, learning_rate=args.lr,
                lmbda=args.lmbda, iterations=iterations),
            "name": "BCD",
            "inner": 1
        }
    ]
    #
    for opt in optimizers:
    #
    #     # training_loss and testing_loss includes references to the training
    #     # and test set respectively.
    #
        loss_hist_train, loss_hist_test, clock = iterate(
            opt['opt'],
            w_1, w_2, w_3,
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
    ax[0].set_title("CA3 - Training a deep neural network MNIST Dataset")
    ax[0].set_ylim(ymin=0)

    ax[1].legend(loc="upper right")
    ax[1].set_xlabel(r"Time [s]", fontsize=16)
    ax[1].set_ylabel("Loss", fontsize=16)
    ax[1].set_ylim(ymin=0)

    plt.savefig("power.png")


if __name__ == "__main__":
    main()
