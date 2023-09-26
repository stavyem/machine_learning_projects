import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1] # Get the labels from the last column
    unique_labels, count_occurrences_of_label = np.unique(labels, return_counts=True)#Count occurrences of each label
    probabilities = count_occurrences_of_label / len(data)
    gini = 1 - np.sum(probabilities ** 2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    labels = data[:, -1] # Get the labels from the last column
    unique_labels, count_occurrences_of_label = np.unique(labels, return_counts=True)#Count occurrences of each label
    probabilities = count_occurrences_of_label / len(data)
    entropy = - np.sum(probabilities *np.log2(probabilities))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy
def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
  
    goodness = 0
    sortedFAndClass = data[:, [feature, -1]][data[:, [feature, -1]][:, 0].argsort()]# 2d array: first col is feature, second col is class. array is sorted by first col
    size_entire_data_set = len(data) # 𝜑 () can be calc_gini or calc_entropy depnded by gain_ratio flag. 
    phi_impurity_measure_function_impurity_measure_function = None # 𝜑 () can be calc_gini or calc_entropy depnded by gain_ratio flag. 

    if gain_ratio:
        phi_impurity_measure_function = calc_entropy
    else:
        phi_impurity_measure_function = impurity_func

    goodness = phi_impurity_measure_function(data) # Calculate 𝜑(𝑆) 
    sum = 0 # Sum of (|Sv|/|S|) * 𝜑(𝑆) 
    last_index = 0
    last_cell = sortedFAndClass[0, 0]

    for i, cell in enumerate(sortedFAndClass):
        if cell[0] != last_cell:
            group = sortedFAndClass[last_index:i]
            group_size = len(group)
            sum += (group_size / size_entire_data_set) * phi_impurity_measure_function(group)
            groups[last_cell] = group
            last_index = i
        last_cell = cell[0]
    # don't forget last value
    group = sortedFAndClass[last_index:]
    group_size = len(group)
    sum += (group_size / size_entire_data_set) * phi_impurity_measure_function(group)
    groups[last_cell] = group

    if gain_ratio:
        goodness -= sum
        goodness = goodness / calc_entropy(np.column_stack([data[:, feature]]))
    else:
        goodness = goodness - sum

###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################




    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups



class DecisionNode:
        



    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Get the labels of the data points in the node
        labels = self.data[:, -1]

        # Calculate the unique values and their frequencies in the labels
        unique_vals, counts = np.unique(labels, return_counts=True)

        # Get the index of the most frequent value
        max_count_idx = np.argmax(counts)

        # Return the most frequent value
        pred = unique_vals[max_count_idx]


      


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):
    
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        best_feature = None
        max_goodness = float('-inf')
        chosen_group = None
        current_impurity = impurity_func(self.data)

        if current_impurity == 0: # Check if node is already pure
            self.terminal = True
            return

        amount_of_features = self.data.shape[1] - 1
        for feature_index in range(amount_of_features):
            feature_goodness,feature_group = goodness_of_split(self.data,feature_index,impurity_func,self.gain_ratio)
            if (feature_goodness is not None) and (feature_goodness > max_goodness):
                    max_goodness = feature_goodness
                    best_feature = feature_index
                    chosen_group = feature_group
    
        # check if chi or max depth conditions are met
        if self.depth >= self.max_depth or max_goodness is None :
            self.terminal = True
            return
        
         # create the children nodes
        if chosen_group is not None:

            for value, data_subset in  chosen_group.items():
            
                child_data = chosen_group[value]
                child_node = DecisionNode(child_data, feature=best_feature, depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(child_node,value)
        else:
            self.terminal = True        

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
  
    
    root = DecisionNode(data,-1,0,chi,max_depth,gain_ratio)
    queue = []
    queue.append(root)

    while(len(queue)>0):
        
        node = queue.pop(0)
        if node.terminal: #node is a leaf
            continue
        else:
            node.split(impurity) #split node
            queue.extend(node.children)    

    ###########################################################################
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
     
    node = root
    while not node.terminal:
        feature_value = instance[node.feature]
        found_child = False
        for i, child_value in enumerate(node.children_values):
            if feature_value == child_value:
                node = node.children[i]
                found_child = True
                break
        if not found_child:
            # if the feature value doesn't match any of the child nodes,
            # return the majority class of the current node
            pred = node.pred
            break
    else:
        # if node is a terminal node, return its prediction
        pred = node.pred



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    counter_correct_prediction = 0
    for instance in dataset :
        if predict(node,instance)==instance[-1]:
            counter_correct_prediction = counter_correct_prediction + 1

   
    
    accuracy = (counter_correct_prediction / len(dataset)) * 100        


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
         tree_classifier_entropy = build_tree(X_train, impurity=calc_entropy, gain_ratio=False, max_depth=max_depth)
         tree_classifier_gini = build_tree(X_train, impurity=calc_gini, gain_ratio=False, max_depth=max_depth)
         tree_classifier_ratio = build_tree(X_train, impurity=calc_gini, gain_ratio=True, max_depth=max_depth)

         training_accuracy = calc_accuracy(tree_classifier, X_train)
         test_accuracy = calc_accuracy(tree_classifier, X_test)
         

         training.append(training_accuracy)
         testing.append(test_accuracy)

   

   # best_depth = np.argmax(testing)
   # training.append(training[best_depth])
   # testing.append(testing[best_depth])    


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






