import pandas as pd
import math
from collections import Counter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Descretize synthetic data by placing data in 10 bins of 20
def descretize(data, index, count):
    counter, value = 1, 1
    # Iterate through each row of the given column, placing the value in a bin
    for i in range(len(data)):
        if counter < count:
            data.iat[i, index] = value
            counter += 1
        else:
            data.iat[i, index] = value
            counter = 1
            value += 1
    return data

# Simple function to help descretize the synthetic data by sorting the rows based on columns
def syntheticData(data, count):
    data = data.sort_values(by = 0)
    data = descretize(data, 0, count)
    data = data.sort_values(by = 1)
    data = descretize(data, 1, count)
    return data

def pokemon(data, count):
    for i in range(7):
        data = data.sort_values(by = i)
        data = descretize(data, i, count)

    return data

# Calculate entropy for a dataset
def entropy(data):
    yes = 0
    no = 0

    # Iterate through each class label, keeping count of Yes and No or 1 and 0
    for i in data:
        if i == 1:
            yes += 1
        else:
            no += 1

    # Calculate the entropy after finding the number of each class label
    if yes == 0 or no == 0:
        return 0
    else: return ((-yes/len(data))*math.log((yes/len(data)),2) - ((no/len(data))*math.log((no/len(data)),2)))

# Calculate Information gain for an attribute
def informationGain(parentEntropy, attribute):
    lastIndex = 0 # Keep track of the first index for creating a sublist of the data that are in the same bin
    count = 1 # Keep track of the number of data points that are in the same bin
    total = 0 # Keep track of the total entropy for the given attribute

    # Iterate through every value for the given attribute
    for i in range(1, len(attribute) + 1):
        # If the current value is in the same bin as the last element
        if i < len(attribute) and attribute.iat[i, 0] == attribute.iat[i-1, 0]:
            count += 1
        else: # Else find the entropy of the last bin and reset the bin variables
            total += ((count/len(attribute))*entropy(attribute.iloc[:, -1].tolist()[lastIndex:i]))
            count = 1
            lastIndex = i
    
    # Return the total information gain on the specific attribute
    return parentEntropy - total

# Find the best attribute to split on
def bestAttribute(data, parentEntropy):
    myAttribute = 0 # Set the best attribute as the first one available
    temp = data.iloc[[0, -1]].sort_values(by = data.columns[0]) # Create a sub matrix for that attribute
    maxInfo = informationGain(parentEntropy, temp) # Calculate the information gain for that attribute
    # Repeat the above steps for every available attribute
    for i in range(1, len(data.columns)-1):
        if i in data.columns:
            print(data.columns[i], i) 
            print(data.iloc[[i, -1]].head())
            temp = data.iloc[[i, -1]].sort_values(by = data.columns[i])
            infoGain = informationGain(parentEntropy, temp)

            # If the new attribute has a higher information gain, make it current best
            if infoGain > maxInfo:
                maxInfo = infoGain
                myAttribute = i
    
    # Return the best attribute to split on
    return myAttribute

# Node to keep track of either an attribute and decisoins to make or a single value
class Node: 
    def __init__(self, attribute=None, branches=[], value=None):
        self.attribute = attribute
        self.branches = branches
        self.value = value

    # Check if the node is a leaf node
    def leafNode(self):
        return self.value is not None

# Tree class to train and test our data
class Tree:
    def __init__(self, max_depth = 3, num_attributes = None):
        self.max_depth = max_depth
        self.num_attributes = num_attributes
        self.root = None

    # ID3 decision tree algorithm to build our tree
    def _ID3(self, data, class_label, not_class_label, parent_entrophy, depth):
        # Compute most common label variable
        mostCommon = self.most_common_label(data.iloc[:, -1].tolist())

        myRoot = Node(branches = []) # Create a Node with empty sub branches
        
        # If our data only has one class label or there are no more attributes to split or max depth reached
        if class_label not in data.iloc[:, -1].tolist(): 
            myRoot.value = not_class_label  
            return myRoot
        elif not_class_label not in data.iloc[:, -1].tolist():
            myRoot.value = class_label
            return myRoot
        elif len(data.columns) == 1:
            myRoot.value = mostCommon
            return myRoot
        elif depth >= self.max_depth:
            myRoot.value = mostCommon
            return myRoot

        # Find the best attribute to split on
        myAttribute = bestAttribute(data, parent_entrophy)
        newEntropy = entropy(data.iloc[:, -1].tolist())

        myRoot.attribute = myAttribute    # Set the nodes attribute
        data = data.sort_values(by = data.columns[myAttribute])    # Sort the data based on the best attribute
        newData = data.drop(columns= data.columns[myAttribute])    # Remove the best attrbiute from the data 

        lastIndex = 0
        curVal = 1

        # Iterate through every value for the given attribute
        for i in range(0, len(data) + 1):

            # For each empty data set create a branch that holds the most common label
            while i != 0 and curVal <= self.num_attributes and i < len(data) and int(data.iat[i-1, data.columns[myAttribute]]) != curVal:
                    myRoot.branches.append(Node(value=mostCommon))
                    curVal += 1

            # If the current value is in the same bin as the previous continue
            if i != 0 and i < len(data) and data.iat[i, data.columns[myAttribute]] == data.iat[i-1, data.columns[myAttribute]]:
                continue
            elif i != 0: # Else create a new branch of the node
                myRoot.branches.append(self._ID3(newData.iloc[lastIndex:i], class_label, not_class_label, newEntropy, depth+1))
                curVal += 1
                lastIndex = i

        return myRoot 

    # Return the most common label
    def most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # Build the Decision tree
    def growMyTree(self, data, class_label, not_class_label, parent, depth):
        self.root = self._ID3(data, class_label, not_class_label, parent, depth)

    # Traverse through the tree to predict the label
    def traverseTree(self, data, node):
        if node.leafNode():
            return node.value
        return self.traverseTree(data.drop(columns=data.columns[node.attribute]), node.branches[int(data.iat[0, data.columns[node.attribute]])-1])

def main():
    # Read in and discretize the synthetic data files
    data = []
    data.append(syntheticData(pd.read_csv('synthetic-1.csv', header=None), 20))
    data.append(syntheticData(pd.read_csv('synthetic-2.csv', header=None), 20))
    data.append(syntheticData(pd.read_csv('synthetic-3.csv', header=None), 40))
    data.append(syntheticData(pd.read_csv('synthetic-4.csv', header=None), 20))

    # For every synthetic data file
    for i in data:
        myTree = Tree(max_depth=3, num_attributes=10)     # Create a Tree
        parent = entropy(i.iloc[:, -1].tolist())          # Calculate parent entropy
        myTree.growMyTree(i, 1, 0, parent, 0)             # Build the Tree

        count = 0
        # Traverse the tree for every piece of data and calculate the accuracy of our Tree
        for j in range(len(i)):
            if int(i.iat[j, -1]) == myTree.traverseTree(i.iloc[[j]], myTree.root):
                count += 1

        print(100*(count/len(i)))


if __name__ == "__main__":
    main()

    data = pd.read_csv('pokemonStats.csv', header=None)
    data = data.iloc[1:, :]
    
    label = pd.read_csv('pokemonLegendary.csv', header=None)
    data.insert(44, 44, label.iloc[1:, -1].tolist())
    for i in range(len(data)):
        if str(data.iat[i, -1]) == "True":
            data.iat[i, -1] = 1
        else: data.iat[i, -1] = 0
    
    data = pokemon(data, 105)

    myTree = Tree(max_depth=3, num_attributes=14)     # Create a Tree
    parent = entropy(data.iloc[:, -1].tolist())          # Calculate parent entropy
    myTree.growMyTree(data, 1, 0, parent, 0)             # Build the Tree

    count = 0
    # Traverse the tree for every piece of data and calculate the accuracy of our Tree
    for j in range(len(data)):
        if data.iat[j, -1] == myTree.traverseTree(data.iloc[[j]], myTree.root):
            count += 1
            

    print(100*(count/len(data)))