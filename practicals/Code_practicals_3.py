import pandas as pd
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



def feature_selection():
    """ Function that performs feature selection. """
    
