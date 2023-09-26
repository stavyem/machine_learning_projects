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
    if len(data.shape) == 1:
        labels = data
    else:
        labels = data[:, -1]
    # getting the counts of each class apperance
    class_types, counts = np.unique(labels, return_counts=True) 
    sigma = np.square(counts / len(labels))
    gini = 1 - np.sum(sigma)
    
    
    # labels = data[:, -1] # Get the labels from the last column
    # unique_labels, count_occurrences_of_label = np.unique(labels, return_counts=True)#Count occurrences of each label
    # probabilities = count_occurrences_of_label / len(data)
    # gini = 1 - np.sum(probabilities ** 2)
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
    if len(data.shape) == 1:
        labels = data 
    else:
        labels = data[:, -1]
    # getting the counts of each class apperance
    class_types, counts = np.unique(labels, return_counts=True) 
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities.dot(np.log2(probabilities)))
    
    # labels = data[:, -1] # Get the labels from the last column
    # unique_labels, count_occurrences_of_label = np.unique(labels, return_counts=True)#Count occurrences of each label
    # probabilities = count_occurrences_of_label / len(data)
    # entropy = - np.sum(probabilities *np.log2(probabilities))
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
    # Get the unique feature values
    unique_feature_values = np.unique(data[:, feature])
    chosen_impurity = None
    if gain_ratio:
        chosen_impurity = calc_entropy
    else:
        chosen_impurity = impurity_func
    # Split the data according to the unique feature values
    for feature_value in unique_feature_values:
        groups[feature_value] = data[data[:, feature] == feature_value]

    # Calculate the impurity of the data before splitting
    initial_impurity = chosen_impurity(data)

    goodness = 0
    # Calculate the impurity of the data after splitting and compute the goodness of split
    for feature_value in unique_feature_values:
        group_data = groups[feature_value]
        group_impurity = chosen_impurity(group_data)
        group_size = group_data.shape[0]
        goodness += (group_size / data.shape[0]) * group_impurity
    goodness = initial_impurity - goodness

    # Compute gain ratio if specified
    if gain_ratio:
        split_information = 0
        for feature_value in unique_feature_values:
            group_data = groups[feature_value]
            group_size = group_data.shape[0]
            split_information += -(group_size / data.shape[0] * np.log2(group_size / data.shape[0]))
        gain_ratio = goodness / split_information
        return gain_ratio, groups
    
    # #split
    # cls_types, counts = np.unique(data[:, feature], return_counts=True)
    # cls_rows = [np.where(data[:, feature] == c) for c in cls_types]
    # children = np.array([data[idx, :] for type_c in cls_rows for idx in type_c])
    
    # for type_c in cls_rows:
    #     for idx in type_c:
    #         groups[data[idx, :][0][0]] = data[idx, :]
   
    # #goodness(S, A) = delta(S) - sum(|Sv|/|S| - delta(Sv) for any v in values of A)  
    # goodness = impurity_func(data) - np.sum((counts[i] / data.shape[0]) * impurity_func(children[i]) for i in range(len(cls_types)) if children[i] != [])
    
    # if gain_ratio:
    #     # goodness(S, A) = information_gain(=old goodness) / Split_in_information
    #     #split_in_indormation(S, A) = -sum(|Sv|/|S| log(|Sv|/|S|) for any v in values of A)
    #     info_gain = goodness 
    #     size = [counts[i] / data.shape[0] for i in range(len(counts))]
    #     split_info = -1 * np.sum(size * np.log2(size))
    #     goodness = info_gain / split_info
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
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
    
    # auxiliary function
    def get_class_split(self):    
        num_e = np.where(self.data[:, -1] == 'e')[0].shape[0]
        num_p = np.where(self.data[:, -1] == 'p')[0].shape[0]
        return [num_e, num_p]

    # auxiliary function - stav
    def compute_chi_value(self, temp_children):
        """
        comput the chi value of a specific node
        
        Input:
        - list of DecisionNode: the children that create by this split(according to specific feature)
        
        Returns:
        - chi value
        - the children that create by this split
        """
        class_split = self.get_class_split()
        cls0 = class_split[0] / self.data.shape[0]
        cls1 = class_split[1] / self.data.shape[0] 
        list_df = [node.data.shape[0] for node in temp_children]
        list_p_f = [node.get_class_split()[0] for node in temp_children]
        list_n_f = [node.get_class_split()[1] for node in temp_children]
        list_E0 =  [df * cls0 for df in list_df]
        list_E1 = [df * cls1 for df in list_df]
        x1 = np.divide(np.square(np.subtract(list_p_f, list_E0)), list_E0)
        x2 = np.divide(np.square(np.subtract(list_n_f, list_E1)), list_E1)
        x1 = np.sum(x1)
        x2 = np.sum(x2)
        x = x1 + x2
        return x, temp_children

    def split(self, impurity_func): # TODO: check if good
    
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
        max_goodness = 0
        best_feature = -1
        groups = {}
        flag_chi = 1 # we have that flag for the randomness checking of the selected feature

        for feature in range(self.data.shape[1] - 1):
            feature_goodness, current_groups = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)
            if feature_goodness > max_goodness:
                max_goodness = feature_goodness
                best_feature = feature
                groups = current_groups
        self.feature = best_feature

        # degree_of_freedom = (number of atribute - 1) * (number of classes - 1)
        degree_of_freedom = len(groups.values()) - 1
        
        children = [DecisionNode(child) for child in groups.values()]
        children_values = groups.keys()       

        if self.chi != 1: #pruning, don't split to children
            chi_square_stat, split_children = self.compute_chi_value(children)
            # we choose beforehand the alpha risk, and then we have chi as array of values and we take the value according to degree of freedom
            if self.chi[degree_of_freedom] >= chi_square_stat:
                flag_chi = 0
    
        if self.chi == 1 or flag_chi == 1: #no pruning, split to children
            for child_node, val in zip(children, children_values):
                child_node.depth = self.depth + 1
                child_node.chi = self.chi
                child_node.gain_ratio = self.gain_ratio
                self.add_child(child_node, val)

        return   
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
    
    # self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False
    queue = [DecisionNode(data,-1,0,chi,max_depth,gain_ratio)]
    #num_of_nodes = 0
    depth = 0
    
    while queue and depth < max_depth:
        n = len(queue)
        depth += 1
        for i in range(n):    
            node = queue.pop(0)
            class_split = node.get_class_split() 
            if max(class_split) == node.data.shape[0]: #check if node pure
                node.terminal = True
                pass #continue = continue to the next iteration no matter what
            else:
                node.split(impurity)

                if root == None:
                    root = node
                root.depth = depth
                    
                if len(node.children) != 0:
                    queue += node.children
    # if(len(root.children) == 0):
    #     root.terminal = True      
    root.depth -= 1
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
    - instance: a row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    node = root
    
    while node.children:
        current_feature = node.feature
        instance_ans = instance[current_feature]
    
        if instance_ans in node.children_values:
            idx_c = node.children_values.index(instance_ans)
            child = node.children[idx_c]
            node = child
        else:
            break
    pred = node.pred
    
    # node = root
    # while not node.terminal:
    #     feature_value = instance[node.feature]
    #     found_child = False
    #     for i, child_value in enumerate(node.children_values):
    #         if feature_value == child_value:
    #             node = node.children[i]
    #             found_child = True
    #             break
    #     if not found_child:
    #         # if the feature value doesn't match any of the child nodes,
    #         # return the majority class of the current node
    #         pred = node.pred
    #         break
    # if node.terminal:
    #     # if node is a terminal node, return its prediction
    #     pred = node.pred
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
    # size_data = dataset.shape[0]
    # correct = 0
    # for i in range(size_data):
    #     instance = dataset[i, :]
    #     if predict(node, instance) == instance[-1]:
    #         correct += 1
    # accuracy = 100 * (correct / size_data)
    
    counter_correct_prediction = 0
    for instance in dataset:
        if predict(node, instance) == instance[-1]:
            counter_correct_prediction += 1
    
    accuracy = (counter_correct_prediction / len(dataset)) * 100        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

# #auxiliary function
# def calc_depth(root):
#     if root.terminal == True:
#         return 0
#     return 1 + np.max([calc_depth(child) for child in root.children])

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
    depths = np.arange(10) + 1
    # test_depth_trees = []
    # index, max_val = 0, 0
    for depth in depths:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
        # test_depth_trees.append(tree)
        # # get the index of the best result for us to mark it with a red circle!
        # if max_val < training[-1]:
        #     max_val = training[-1]
        #     index = depth

        
    # for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #      tree_classifier_entropy = build_tree(X_train, impurity=calc_entropy, gain_ratio=False, max_depth=max_depth)
    #      tree_classifier_gini = build_tree(X_train, impurity=calc_gini, gain_ratio=False, max_depth=max_depth)
    #      tree_classifier_ratio = build_tree(X_train, impurity=calc_gini, gain_ratio=True, max_depth=max_depth)

    #      training_accuracy = calc_accuracy(tree_classifier, X_train)
    #      test_accuracy = calc_accuracy(tree_classifier, X_test)
         

    #      training.append(training_accuracy)
    #      testing.append(test_accuracy)

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
    best_tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True)
    depth_of_best_tree_with_chi_1 = best_tree.depth
    depth += [depth_of_best_tree_with_chi_1]

    train_entropy_gain_ratio_accuracy = calc_accuracy(best_tree, X_train)
    test_entropy_gain_ratio_accuracy = calc_accuracy(best_tree, X_test)

    chi_training_acc += [train_entropy_gain_ratio_accuracy] 
    chi_testing_acc += [test_entropy_gain_ratio_accuracy]
    
    p_value_cut_off = [0.5, 0.25, 0.1, 0.05, 0.0001]
    for p_cut in p_value_cut_off:
        chi = [val[p_cut] for key, val in chi_table.items()] 
        temp_tree = build_tree(X_train, calc_entropy, True, chi)
        curr_depth = temp_tree.depth
        chi_training_acc += [calc_accuracy(temp_tree, X_train)]
        chi_testing_acc += [calc_accuracy(temp_tree, X_test)]
        depth += [curr_depth]
    ############################
    # p_value_cut_off.append(1)

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
    if not node.children:
        return 1
    
    n_nodes = 1 + sum(count_nodes(child) for child in node.children)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
    


