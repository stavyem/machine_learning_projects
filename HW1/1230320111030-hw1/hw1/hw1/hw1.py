###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X_mean, X_min, X_max = np.mean(X, axis=0), np.min(X, axis=0), np.max(X, axis=0)
    y_mean, y_min, y_max = np.mean(y, axis=0), np.min(y, axis=0), np.max(y, axis=0)

    X = (X - X_mean) / (X_max - X_min)
    y = (y - y_mean) / (y_max - y_min)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ##########################################################################
    X = np.c_[np.ones(len(X), dtype=int), X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###################################
    J = (np.sum(np.power((np.dot(X, theta)) - y, 2)) / (2 * len(y)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    m = len(y)

    for i in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        error = X.dot(theta) - y
        gradient_derivative_by_theta = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient_derivative_by_theta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    X_pinv = np.linalg.inv(X.T.dot(X)).dot(X.T)  # Compute X pinv according to the formula in the lecture.
    pinv_theta = X_pinv.dot(y)  # 𝜃 = pinv(X)y.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    m = len(y)
    tolerance = 1e-8
    J_history.append(compute_cost(X, y, theta))

    for i in range(num_iters):
        error = X.dot(theta) - y
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient
        new_cost = compute_cost(X, y, theta)
        improvement = abs(new_cost - J_history[-1])
        J_history.append(new_cost)
        if improvement < tolerance or new_cost > np.power(10,10):
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    np.random.seed(42)
    theta = np.random.random(size=2)
    for alpha in alphas:
        theta_alpha, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)  # train the model
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta_alpha)  # alpha as the key and the loss on the validation set as the value.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################  
    features_left = set(range(0, X_train.shape[1]))  # exclude bias from features
    current_cost = float("inf")
 
    while len(selected_features) < 5:  # finish when we have 5 best features
        best_feature = None
        for feature in features_left:

            features_to_try = selected_features + [feature]  # tries the next feature
            X_train_temp = X_train[:, features_to_try]  # train a spacific feature

            X_val_temp = X_val[:, features_to_try]  # valdition for the spacific feature
            np.random.seed(42) 
            theta = np.random.random(len(features_to_try) + 1)  # create a random theta

            new_theta, J_history = efficient_gradient_descent(apply_bias_trick(X_train_temp), y_train, theta, best_alpha, iterations)
            cost = compute_cost(apply_bias_trick(X_val_temp), y_val, new_theta)
            if cost < current_cost:  # check if the current cost is better.
                current_cost = cost
                best_feature = feature
        selected_features.append(best_feature)  # add the feature
        features_left.remove(best_feature)  # remove this feature, for next iters.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    # Create square features for each pair of features
    for i, feature1 in enumerate(df.columns):
        for feature2 in df.columns[i:]:
            if feature1 != feature2:
                col_name = f"{feature1}*{feature2}"
                df_poly[col_name] = df_poly[feature1] * df_poly[feature2]

            # Add square features for each feature squared
            else:
                col_name = f"{feature1}^2"
                df_poly[col_name] = df_poly[feature1] ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly