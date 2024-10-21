from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

def hyperparameter_selection(X_train, X_test, y_train, y_test, grid):
    """ Function that performs a grid search for a SVM model. """

    SVM = svm.SVC()
    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=SVM, param_grid=grid, cv=5, scoring='balanced_accuracy', verbose = 2, n_jobs=-1)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_estimator_

def objective(trial, X_train, y_train, max_fold=5):
    base_params = {
        "oob_score": True,
        "bootstrap": True,
        "random_state": 333
    }

    # Define tunable hyperparameters
    params = {
        "n_estimators": trial.suggest_int(name="n_estimators", low=100, high=1000),
        "max_depth": trial.suggest_int("max_depth", low=10, high=100), 
        "max_features": trial.suggest_categorical(name="max_features", choices=["sqrt", "log2"]),
        "min_samples_split": trial.suggest_int(name="min_samples_split", low=2, high=50),
        "min_samples_leaf": trial.suggest_int(name="min_samples_leaf", low=1, high=50)
    }

    # Create the model
    params.update(base_params)
    model = RandomForestClassifier(**params, class_weight="balanced_subsample")

    # Cross validation
    cv_score = cross_val_score(model, X_train, y_train, scoring="balanced_accuracy", cv=max_fold)
    
    return cv_score.mean()

def plot_feature_importance(model, data):
    # Gather feature importance
    importances = model.feature_importances_
    feature_names = [col_name for col_name in data.columns]

    # Sort feature importance
    zipped_data = sorted(zip(importances, feature_names), reverse=True)
    importances, feature_names = zip(*zipped_data)

    # Plot feature importance
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.bar(x = feature_names[:50], height=importances[:50])
    ax.set_title("Top 50 informative features", size=18)
    ax.set_ylabel("Feature importance", size=12)
    plt.xticks(size=12, rotation=90)
    plt.xticks(size=12)
    fig.tight_layout()
    ax.grid() 

def confusion_mat(true_labels, pred_labels):	
	# Create confusion matrix
	conf_matrix = confusion_matrix(true_labels, pred_labels)
	matrix = ConfusionMatrixDisplay(conf_matrix)

	# Calculate sensitivity and specificity
	tn, fp, fn, tp = conf_matrix.ravel()
	sens = tp/(tp + fn)
	spec = tn/(tn + fp)

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	matrix.plot(cmap="gray", ax=ax);

	print(f"Sensitivity: {sens:.2f}")
	print(f"Specificity: {spec:.2f}")