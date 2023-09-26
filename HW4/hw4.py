import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
  h = sigmoid(X, theta)
  return np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / X[0].shape

def sigmoid(X, theta):
  return 1 / (1 + np.exp(-X.dot(theta)))


def update_theta(X, y, theta, rate):
    return theta - (rate * (X.T).dot((sigmoid(X, theta) -y))) 


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []


    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # bias trick
        w_0 = np.ones(X.shape[0])
        X = np.column_stack((w_0, X))

        # random weights for each feature
        theta = np.random.random(size=X.shape[1])

        Js = []
        for _ in range(self.n_iter):
          Js.append(compute_cost(X, y, theta)) #?
          theta = update_theta(X, y, theta, self.eta)
          self.thetas.append(theta)
          
          if len(Js) > 1 and (Js[-2] - Js[-1]) < self.eps:
            break
        
        self.theta = theta
        self.Js = Js
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # bias trick
        w_0 = np.ones(X.shape[0])
        X = np.column_stack((w_0, X))

        preds = np.array([1 if sigmoid(X[i], self.theta) > 0.5 else 0 for i in range(len(X))])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    data = np.column_stack((X, y))
    # Shuffle the data
    np.random.shuffle(data)
    shuffled_train = data

    X_train_shuffle, y_train_shuffle = shuffled_train[:, [0, 1]], shuffled_train[:, 2]

    split_X_train, split_Y_train = np.array(np.array_split(X_train_shuffle, folds)), np.array(np.array_split(y_train_shuffle, folds))
    acc_list = []
    
    for i in range(folds): 
        
        temp_train_X, temp_train_Y = np.concatenate(np.delete(split_X_train, i, 0)), np.concatenate(np.delete(split_Y_train, i, 0))
        temp_test_X, temp_test_Y = split_X_train[i], split_Y_train[i]

        algo.fit(temp_train_X, temp_train_Y)
        
        temp_predict = algo.predict(temp_test_X)
        
        acc_list.append(np.sum(temp_test_Y == temp_predict) / len(temp_test_Y))
       
    cv_accuracy =  np.array(acc_list).mean()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    numerator = np.power(np.e, -0.5 * np.power(((data - mu)/sigma), 2))
    denominator = sigma * np.power(2 * np.pi, 0.5)
    p = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None # determine the assignment probabilities of data points to each component
        self.weights = None # weights determine the relative importance of each component in the overall mixture.
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        number_davisions_data = (int)(data.shape [0]/self.k )# assuming that the Gaussian distributions are uniformly distributed.
        start_index = 0
        end_index = number_davisions_data
        self.weights = []
        self.mus = []
        self.sigmas = []
        for i in range (self.k): # We need the if else, for case that number_davisions_data is not devisbile for example k=4 and we have 10 samples.
              if i == self.k - 1 : # last iteration get all the remaining data
                    data_i = data [start_index:]
                    self.weights.append(1/self.k)
                    self.mus.append(data_i.mean())
                    self.sigmas.append(data_i.std())

              else:
                    data_i = data[start_index:end_index]
                    self.weights.append(1/self.k)
                    self.mus.append(data_i.mean())
                    self.sigmas.append(data_i.std())

              start_index = end_index
              end_index += number_davisions_data
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihoods = [self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i]) for i in range(self.k)]
        dev = sum(likelihoods)
        return [likelihood / dev for likelihood in likelihoods]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        responsibilities = self.expectation(data)

        for i in range(self.k):
            self.weights[i] = responsibilities[i].mean()
            self.mus[i] = (responsibilities * data).mean() * (1/self.weights[i])
            self.sigmas[i] = np.sqrt((responsibilities * np.square(data - self.mus[i])).mean() * (1/self.weights[i]))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.costs = []
        for j in range (self.n_iter):
            
              cost = self.compute_costs(data,self.mus,self.sigmas,self.weights,self.k)
              self.maximization(data)
              self.costs.append (cost)
              if j > 0 and np.absolute(self.costs[-1] - cost) < self.eps:
                    break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.mus, self.sigmas, self.weights
    

    def compute_costs(self,data, mus, sigma, weights, k):
        cost = 0
        for i in range(k):
               likelihood = weights[i] * norm_pdf(data, mus[i], sigma[i])
               cost += likelihood
    
        return np.sum(-np.log(cost))

    
    

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = 0
    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])

        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.emX1 = [EM(k=self.k) for _ in range(2)]
        self.emX2 = [EM(k=self.k) for _ in range(2)]

    def Get_prior(self,data): 
          self.prior = [0,0] # init prior - 0 probabilities for the 2 classes.
          number_instances = data.shape[0] # 1000
          for i in range(data.shape[1]-1): # excluding the last column
              class_i_data = [i for j in range(number_instances) if data[j][2] == i] # create a list of all the instances in specific class
              self.prior[i] = len (class_i_data) / number_instances    

          
    def likelihood(self, x, j):
          # j - represents the class. i - represents the i gaussians from the k components. 
      likelihood_x1 = sum([self.emX1[j].weights[i] * norm_pdf(x[0], self.emX1[j].mus[i], self.emX1[j].sigmas[i]) for i in range(self.k)])
      likelihood_x2 = sum([self.emX2[j].weights[i] * norm_pdf(x[1], self.emX2[j].mus[i], self.emX2[j].sigmas[i]) for i in range(self.k)])
      return likelihood_x1 * likelihood_x2 # The assumption here is that the feature dimensions are independent - Naive Bayes classifiers.
    

    def posterior(self,x):
          return (self.likelihood(x=x,j=0)*self.prior[0],self.likelihood(x=x,j=1)*self.prior[1])
      
                  
                

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        data = np.column_stack([X, y]) # Create Array that each row conatain it's features value and corresponding class label.
        self.Get_prior(data)
        for class_value in range(2):
          class_data = data[data[:, -1] == class_value]
          self.emX1[class_value].fit(class_data[:,0])
          self.emX2[class_value].fit(class_data[:,1])
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        predictions = []

        for Row in X:
              predict_0,predict_1 = self.posterior(Row)
              if predict_0 >= predict_1:
                    predictions.append(0)
              else:
                    predictions.append(1)

        preds = np.asarray(predictions)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds
    

def accuracy_evaluation(X , y, classifier): # calcaluate the accuracy of the model.
    prediction_res = y - classifier.predict(X)
    prediction = len(prediction_res) - np.count_nonzero(prediction_res)
    return (prediction/ len(X)) * 100


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # fit Logistic Regression model
    LR = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    LR.fit(x_train, y_train)
    # Logistic Regression -> accuracy evaluation (train)
    lor_train_acc = accuracy_evaluation(x_train, y_train, LR)

    # Logistic Regression -> accuracy evaluation (test)
    lor_test_acc = accuracy_evaluation(x_test, y_test, LR)

    # fit Naive Bayes model
    NB = NaiveBayesGaussian(k=1)
    NB.fit(x_train, y_train)

    # Naive Bayes -> accuracy evaluation (train)
    bayes_train_acc = accuracy_evaluation(x_train, y_train, NB)

    # Naive Bayes -> accuracy evaluation (test)
    bayes_test_acc = accuracy_evaluation(x_test, y_test, NB)
    print("Logistic Regression -> for the training data the accuracy is ", lor_train_acc, " for the test data the accuracy is ", lor_test_acc)
    print("Naive Bayes -> for the training data the accuracy is ", bayes_train_acc, " for the test data the accuracy is ", bayes_test_acc)


    # introduce diagrams for part of the data (1000 samples for the training data and 500 samples for the test data)
    # explanation: We can observe the 2 gaussians in our first 1000 points of training data and a clear linear separator that LOR algorithem yields.
    plot_decision_regions(x_train,y_train,LR,resolution=0.01, title="Decision boundaries For Logistic Regression")

    # explanation: We can observe the 2 gaussians in our first 1000 points of training data. NB is a classifier so it is capble of yielding linear and non-linear separators.
    # in this case, a linear separator was enough
    plot_decision_regions(x_train,y_train,NB,resolution=0.01, title="Decision boundaries For Naive Bayes")

    # explanation: We can observe the loss decreasing as the number of iterations increasing. this is becuase LOR is an iterative algorithem which improves its cost as time goes by.
    plt.plot(np.arange(len(LR.Js)), LR.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations')
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}




# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()



def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # First dataset
    x1 = np.random.normal(5, 1, 1000)
    x2 = np.random.normal(5.5, 1, 1000)
    x3 = np.random.normal(4.5, 2, 1000)

    y = np.random.normal(10.0, 2.0, 1000)

    dataset_a_features = np.stack((x1, x2, x3), axis=1)
    dataset_a_labels = y

    # Second dataset
    x1 = np.random.normal(5, 0.5, 1000)
    x2 = np.random.normal(20, 1, 1000)
    x3 = np.random.normal(4.8, 0.5, 1000)

    y = np.random.normal(10.0, 2.0, 1000)

    dataset_b_features = np.stack((x1, x2, x3), axis=1)
    dataset_b_labels = y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }