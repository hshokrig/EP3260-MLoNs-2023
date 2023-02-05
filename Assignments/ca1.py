from sklearn.model_selection import train_test_split
# import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_data

def loss_fn_cf(X, y, w):
    """
    The loss function || w^Tx ||_2^2
    """

    wtX = np.dot(X, w)
    return np.linalg.norm(wtX - y, 2)**2


def mse_cf(X, y, w):

    N = X.shape[0]
    return (1.0 / N) * loss_fn_cf(X, y, w)


def closed_form(X, y, lmbda=0):
    """
    w_fit = inv(X.T @ X + (N/2) * lambda * I)) @ X.T @ y
    """

    N = X.shape[0]
    M = X.shape[1]

    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    Xty = np.dot(Xt, y)

    w_fit = np.matmul(np.linalg.inv(XtX + (N / 2.0) * lmbda * np.eye(M)), Xty)
    # w_fit = np.linalg.solve(XtX + (N / 2.0) * lmbda * np.eye(M), Xty)
    return w_fit


def predict(X, w):
    return np.dot(X, w)


def main():
    """
    TODO: Loop over datasets
    """

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))

    for d, dataset in enumerate(["power"]):#"communities", 

        X, y = get_data(dataset)
        print("X,y types: {} {}".format(type(X), type(y)))
        print("X size {}".format(X.shape))

        m, n = X.shape

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5)

        lambd_values = np.logspace(-4, 2, 100)
        train_errors = []
        test_errors = []
        # beta_values = []

        for v in lambd_values:
            print("closed form: Now working on lambda={}".format(v))
            w_opt = closed_form(X_train, y_train, lmbda=v)

            train_errors.append(mse_cf(X_train, y_train, w_opt))
            test_errors.append(mse_cf(X_test, y_test, w_opt))
            # beta_values.append(w_opt)

        ax[d].plot(lambd_values, train_errors, label="Train MSE")
        ax[d].plot(lambd_values, test_errors, label="Test MSE")
        ax[d].set_xscale("log")
        ax[d].legend(loc="upper left")
        ax[d].set_xlabel(r"$\lambda$", fontsize=16)
        ax[d].set_title(
            "{} - Closed form - Mean Squared Error (MSE)".format(dataset))
        ax[d].set_ylabe("Mean Squared Error (MSE)".format(dataset))
        ax[d].set_ylim(ymin=0, ymax=0.1)

    plt.savefig("ca1.png")


if __name__ == "__main__":
    main()
