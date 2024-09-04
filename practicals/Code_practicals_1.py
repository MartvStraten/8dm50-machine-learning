import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def knn(k, X_train, y_train, X_test):
    """ K-Nearest Neighbours can be implemented from scratch in three steps:
    
    	Step 1: calculate Euclidean distance between all points with the unknown classified point.
        Step 2: find the k nearest neighbours of the point with unknown class.
        Step 3: make a prediction for the unknown classified point.
    """
    y_hat_test = []
    
	# Loop over all test samples
    for idx_test in range(len(X_test)):
        test_sample = X_test[idx_test,:] # 30 dimensional
        distances = []
        
		# Loop over all training samples to calculate Euclidean distances
        for idx_train in range(len(X_train)):
            train_sample = X_train[idx_train,:]
            dist = euclidean(test_sample, train_sample)
            
			# Fill distances list with all Euclidean distances and training index
            distances.append((dist, idx_train))
        
		# Sorting the distances
        distances.sort(key=lambda tup: tup[0])
        
		# Selecting k-nearest neighbours
        y_neighbours = []
        for idx in range(k):
            label = y_train[distances[idx][1]]
            y_neighbours.append(label)
        
		# Make prediction based on class labels of neighbours
        y_hat = np.round(np.mean(y_neighbours))
        y_hat_test.append(y_hat)
        
    return y_hat_test 

def elbow_plot(max_k, X_train, y_train, X_test, y_test):
    """ Function to plot elbow plot for multiple hyperparameter values of k. """
    all_errors = []
    k_values = []
    
	# Loop over all values of k to find the errors
    for k in range(1, max_k + 1):
        y_hat = knn(k, X_train, y_train, X_test)
        error = np.mean(abs(y_test - y_hat))
        all_errors.append(error)
        k_values.append(k)
        
	# Plot the elbow plots
    fig,ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(k_values, all_errors, marker="o")
    ax.set_title("Elbow plot for k-nearest neighbour algorithm")
    ax.set_xlabel("K value")
    ax.set_ylabel("Error")
    ax.grid()

def confusion_mat(true_labels, pred_labels):
    """ Function to plot confusion matrix and calculate the specificity and sensitivity of the model. """	
	# Create confusion matrix
    conf_matrix = confusion_matrix(true_labels, np.round(pred_labels))
    matrix = ConfusionMatrixDisplay(conf_matrix)

	# Calculate sensitivity and specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    sens = tp/(tp + fn)
    spec = tn/(tn + fp)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    matrix.plot(ax=ax);

    print(f"Sensitivity: {sens:.2f}")
    print(f"Specificity: {spec:.2f}")
