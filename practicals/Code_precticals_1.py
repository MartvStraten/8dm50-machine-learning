import numpy as np

def linear_regresiion(X_train, X_test, y_train, y_test):
    # Add a column of ones to X_train for the intercept term
    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

    # Calculate the coefficients using the normal equation with bias term
    parameters = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    # Add a column of ones to X_test for the intercept term
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Predicting the target values for the test set
    y_predict = X_test_bias @ parameters

    # Calculating the Mean Square Error (MSE)
    mse = np.mean((y_predict - y_test) ** 2)

    return mse