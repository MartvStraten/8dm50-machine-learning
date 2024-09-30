import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

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

def coefficient_profile(X_train, X_test, y_train, y_test, log_alpha):
    """ Function that performs feature selection. 
    params:
    X_train: the training data
    X_test: the test data
    y_train: the training labels
    y_test: the test labels
    columns: the column names of the training data
    alpha: a log grid of alpha values to be evaluated
    returns:
    A plot showing how the coefficients vary with an increasing alpha
    """
    
    
    mean_mse_list = []
    errors = []
    bootstrap_nr = 100
    coefs = []

    for a in log_alpha:
        lasso = Lasso() # Create a Lasso model
        lasso.set_params(alpha=a)
        
        mse_list = []
        for i in range(bootstrap_nr):
            X_bs, y_bs = resample(X_train, y_train, replace=True)        
            lasso.fit(X_bs, y_bs)
            
            y_pred = lasso.predict(X_test)
            mse = mean_squared_error(y_test,y_pred)
            mse_list.append(mse)
            
        coefs.append(lasso.coef_)
        mean_mse = sum(mse_list)/len(mse_list)
        mean_mse_list.append(mean_mse)
        errors.append(np.std(mse_list))
    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].plot(log_alpha,coefs)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('alpha')
    ax[0].set_ylabel('Standardized Coefficients')
    ax[0].set_title('Lasso coefficients as a function of alpha');
    
    ax[1].errorbar(log_alpha,mean_mse_list,yerr = errors, capsize=3)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('alpha')
    ax[1].set_ylabel('Mean Squared Error')
    ax[1].set_title('MSE as a function of alpha');

    
    

    