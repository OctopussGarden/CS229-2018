import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    mse = np.linalg.norm(y_pred-y_eval, ord=2) ** 2
    print("MSE is:", mse)

    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_eval, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)
        # w = np.exp(-((np.linalg.norm(x - self.x, ord=2)) ** 2) / (2 * self.tau ** 2))
        # self.theta = np.linalg.inv(x.T.dot(W).dot(x)).dot(x.T.dot(W).dot(self.y))
        for i in range(m):
            # W = np.diag(np.exp(-(np.linalg.norm(x[i] - self.x, ord=2, axis=1) ** 2) / (2 * self.tau ** 2)))
            W = np.diag(np.exp(-np.sum((self.x - x[i])**2, axis=1) / (2 * self.tau ** 2)))
            # Update theta
            y_pred[i] = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T.dot(W).dot(self.y)).dot(x[i])

        return y_pred
        # *** END CODE HERE ***
