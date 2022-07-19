import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    mse_values = np.array([])
    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        # Fit a LWR model
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        # Get MSE value on the validation set
        x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
        y_pred = model.predict(x_eval)
        mse_values = np.append(mse_values, np.linalg.norm(y_pred - y_eval, ord=2) ** 2)
    print("MSE is:", mse_values)
    print("Tau is:", tau_values)
    min_mse = mse_values.min()
    for i in range(len(mse_values)):
        if mse_values[i] == min_mse:
            best_tau = tau_values[i]
            print('Min Tau is :', best_tau)
            break

    # Fit a LWR model with the best tau value
    model_best = LocallyWeightedLinearRegression(best_tau)
    model_best.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model_best.predict(x_test)
    mse = np.linalg.norm(y_pred - y_test, ord=2) ** 2
    print("Best tau is :", best_tau)
    print("Mse of best tau is :", mse)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05c.png')

    # *** END CODE HERE ***
