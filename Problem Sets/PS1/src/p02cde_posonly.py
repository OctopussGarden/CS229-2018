import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    # Train Logistic Regression
    model_t = LogisticRegression(eps=1e-5)
    model_t.fit(x_train, t_train)
    util.plot(x_test, t_test, model_t.theta, 'output/p02c.png')
    # Output test prediction
    t_pred_c = model_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred_c > 0.5, fmt='%d')
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Make sure to save outputs to pred_path_c
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # Train Logistic Regression
    model_y = LogisticRegression(eps=1e-5)
    model_y.fit(x_train, y_train)
    util.plot(x_test, y_test, model_y.theta, 'output/p02d.png')
    # Output test prediction
    y_pred_d = model_t.predict(x_test)
    np.savetxt(pred_path_d, y_pred_d > 0.5, fmt='%d')
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(train_path, add_intercept=True)

    # alpha = np.sum(model_v.predict(x_valid[t_valid == 1.0])) / sum(t_train == 1.0)
    alpha = np.mean(model_y.predict(x_valid))
    # print(model_y.theta)
    correction = 1+np.log(2/alpha -1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02e.png', correction)
    # Output test prediction
    t_pred_e = model_y.predict(x_test)
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt='%d')
    # *** END CODER HERE

