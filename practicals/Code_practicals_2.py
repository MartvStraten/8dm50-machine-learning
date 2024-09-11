import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

def polynomial_regression(X, y, max_order=10, max_fold=5):
    """ Function that performs grid search for the order of polynomial regression. """
    # Create model pipeline
    model = make_pipeline(
        PolynomialFeatures(), 
        linear_model.LinearRegression()
    )	
    
    # Create parameter grid and cross validation object
    param_grid = {"polynomialfeatures__degree": np.arange(max_order + 1)}
    poly_grid = GridSearchCV(model, param_grid, cv=max_fold)

    # Fit models with cross validation
    poly_grid.fit(X.reshape(-1, 1), y)
    results = poly_grid.cv_results_
    
    # Gathering the validation accuracy for each polynomial order
    poly_order = []
    val_acc = []
    for order in range(max_order + 1):
        sum_acc = 0
        for fold in range(max_fold):
            key = f"split{fold}_test_score"
            sum_acc += results[key][order]
        poly_order.append(order)
        val_acc.append(sum_acc/max_fold)

    # Visualization of CV results
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].plot(poly_order, val_acc, marker="o")
    ax[0].set_title("Evaluation polynomial regression models")
    ax[0].set_xlabel("Polynomial order")
    ax[0].set_ylabel("Model accuracy")
    ax[0].grid()
    
    # Visualization of best model
    X_test = np.linspace(min(X), max(X), 50).reshape(-1, 1)
    y_hat = poly_grid.best_estimator_.predict(X_test)
    ax[1].scatter(X, y)
    ax[1].plot(X_test, y_hat, "r-")
    ax[1].set_title(f"Best performing polynomial: {poly_grid.best_index_}th order")
    ax[1].legend(["Data", "Model"])
    ax[1].grid()
