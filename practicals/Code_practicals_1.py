import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm

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

def knn(k, X_train, y_train, X_test, regression=False):
    """ input: k, X_train, y_train, X_test, regression=False
        output: y_hat_test
    
        K-Nearest Neighbours can be implemented from scratch in three steps:
    
    	Step 1: calculate Euclidean distance between all points with the unknown classified point.
        Step 2: find the k nearest neighbours of the point with unknown class.
        Step 3: make a prediction for the unknown classified point.
    """
    y_hat_test = []
    
	# Loop over all test samples
    for idx_test in range(len(X_test)):
        test_sample = X_test[idx_test, :] # 30 dimensional
        distances = []
        
		# Loop over all training samples to calculate Euclidean distances
        for idx_train in range(len(X_train)):
            train_sample = X_train[idx_train, :]
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
        if regression:
            y_hat = np.mean(y_neighbours) # regression prediction 
        else:
            y_hat = np.round(np.mean(y_neighbours)) # classification prediciton 
            
        y_hat_test.append(y_hat)
        
    return np.array(y_hat_test)[:, np.newaxis]

def elbow_plot(max_k, X_train, y_train, X_test, y_test):
    """ Function to plot elbow plot for multiple hyperparameter values of k. """
    all_errors = []
    k_values = []
    
	# Loop over all values of k to find the errors
    for k in range(1, max_k + 1):
        y_hat = knn(k, X_train, y_train, X_test)
        mse = np.mean((y_test - y_hat)**2)
        all_errors.append(mse)
        k_values.append(k)
        
	# Plot the elbow plots
    fig,ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(k_values, all_errors, marker="o")
    ax.set_title("Elbow plot for k-nearest neighbour algorithm")
    ax.set_xlabel("K value")
    ax.set_ylabel("Mean Squared Error")
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
    ax.set_title("Confusion matrix")

    print(f"Sensitivity: {sens:.2f}")
    print(f"Specificity: {spec:.2f}")

def conditional_probability(data_set):
	# Extract data and labels
	X = data_set.data
	Y = data_set.target
	
	# Get feature names
	feature_names = data_set.feature_names
	
	# Number of features
	n_features = X.shape[1]
	
	# Create a plot with subplots for each feature
	fig, axes = plt.subplots(n_features // 3, 3, figsize=(15, 30))
	axes = axes.ravel()
	
	# Loop over each feature
	for i in range(n_features):
	    # Separate data for each class (Y=0 and Y=1)
		X_class_0 = X[Y == 0, i]
		X_class_1 = X[Y == 1, i]
	    
	    # Fit a Gaussian distribution (mean and std dev) for each class
		mean_0, std_0 = np.mean(X_class_0), np.std(X_class_0)
		mean_1, std_1 = np.mean(X_class_1), np.std(X_class_1)
	    
	    # Generate a range of values for the x-axis (feature values)
		x_range = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
	    
	    # Compute the PDFs for each class
		gauss_0 = norm.pdf(x_range, mean_0, std_0)
		gauss_1 = norm.pdf(x_range, mean_1, std_1)
	    
	    # Plot the PDFs
		axes[i].plot(x_range, gauss_0, label='Class 0 (Malignant)', color='red')
		axes[i].plot(x_range, gauss_1, label='Class 1 (Benign)', color='blue')
	    
	    # Set plot labels and title
		axes[i].set_title(f'{feature_names[i]}')
		axes[i].set_xlabel('Feature value')
		axes[i].set_ylabel('Probability Density')
		axes[i].legend()
	
	plt.tight_layout()
	plt.show()
		
