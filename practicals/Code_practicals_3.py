import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def hyperparamter_selection(X_train, X_test, y_train, y_test, alpha):
    """ Function that performs a grid search for a lasso linear regression model. """
    # Make the Lasso model
    lasso = Lasso()

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=lasso, param_grid=alpha, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']
    
    return best_alpha



def get_sorted_features(X_train, y_train, columns,  best_alpha):
    """ Function that performs feature selection. 
    params:
    X_train: the training data
    y_train: the training labels
    columns: the column names of the training data
    best_alpha: the best alpha value found by the hyperparameter selection function
    returns:
    coefficient_bundle: a numpy array with the column names and the absolute values of the coefficients, sorted in descending
    """
    lasso = Lasso() # Create a Lasso model
    lasso.set_params(alpha=best_alpha) # Set the alpha value
    lasso.fit(X_train, y_train) # Fit the model to the training data
    coefficient_bundle = np.array([columns, abs(lasso.coef_)]).transpose() # Create a numpy array with the column names and the absolute values of the coefficients
    coefficient_bundle = coefficient_bundle[coefficient_bundle[:, 1].argsort()[::-1]] # Sort the numpy array in descending order
    return coefficient_bundle