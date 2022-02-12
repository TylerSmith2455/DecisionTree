import pandas as pd
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import PIL

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Descretize synthetic data by placing data in bins
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

# Simple function to help discretize the pokemon data
def pokemon(data, count):
    for i in range(8):
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
    lastIndex = 0   # Keep track of the first index for creating a sublist of the data that are in the same bin
    count = 1       # Keep track of the number of data points that are in the same bin
    total = 0       # Keep track of the total entropy for the given attribute

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
    temp = data.iloc[[0, -1]].sort_values(by = 0) # Create a sub matrix for that attribute
    maxInfo = informationGain(parentEntropy, temp) # Calculate the information gain for that attribute

    # Repeat the above steps for every available attribute
    for i in range(1, len(data.columns)-1):
        if i in data.columns:
            temp = data.iloc[:, [i, -1]].sort_values(by = i)
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
        elif len(data.columns) == 1 or depth >= self.max_depth:
            myRoot.value = mostCommon
            return myRoot

        # Find the best attribute to split on
        myAttribute = bestAttribute(data, parent_entrophy)
        newEntropy = entropy(data.iloc[:, -1].tolist())
        
        myRoot.attribute = myAttribute                  # Set the nodes attribute
        data = data.sort_values(by = myAttribute)       # Sort the data based on the best attribute
        newData = data.drop(columns = myAttribute)      # Remove the best attribute from the data 

        # Rename the columns
        for i in range(len(newData.columns)):
            newData = newData.rename(columns={newData.columns[i]: i})


        lastIndex = 0
        curVal = 1

        # Iterate through every value for the given attribute
        for i in range(0, len(data)+1):
            
            # For each empty data set create a branch that holds the most common label
            while i != 0 and curVal <= self.num_attributes and i <= len(data) and int(data.iat[i-1, myAttribute]) != curVal:
                myRoot.branches.append(Node(value=mostCommon))
                curVal += 1

            # If the current value is in the same bin as the previous continue
            if i != 0 and i < len(data) and data.iat[i, myAttribute] == data.iat[i-1, myAttribute]:
                continue
            elif i != 0: # Else create a new branch of the node
                myRoot.branches.append(self._ID3(newData.iloc[lastIndex:i], class_label, not_class_label, newEntropy, depth+1))
                curVal += 1
                lastIndex = i
        
        # Make a branch for the remaining values not found
        while curVal <= self.num_attributes:
            myRoot.branches.append(Node(value=mostCommon))
            curVal += 1

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

        # Create new data
        newData = data.drop(columns=node.attribute)
        for i in range(len(newData.columns)):
            newData = newData.rename(columns={newData.columns[i]: i})

        # Traverse the tree with the new data
        return self.traverseTree(newData, node.branches[int(data.iat[0, node.attribute])-1])

# Function to create a plot for the dataset and its Tree 
def approximateImage(myTree, data, filename):
    # Compute all points that have class label of 0 and 1
    xValuesFail = []
    yValuesFail = []
    xValuesSuccess = []
    yValuesSuccess = []
    for i in range(len(data)):
        if int(data.iat[i,2]) == 0:
            xValuesFail.append(data.iat[i,0])
            yValuesFail.append(data.iat[i,1])
        else:
            xValuesSuccess.append(data.iat[i,0])
            yValuesSuccess.append(data.iat[i,1])

    # Plot class label 1 points as green nad label 0 points as black
    plt.plot(xValuesFail,yValuesFail,'o',color="#000000",label="Class Label 0")
    plt.plot(xValuesSuccess,yValuesSuccess,'o',color="#77d582",label="Class Label 1")
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)

    plt.margins(x=0, y=0)
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()
    x_min, x_max = axes.get_xlim()

    # Create an image for the background of the plot
    im = PIL.Image.new(mode="RGB", size=(200, 200))
    for i in range(1,21):
        for j in range(1, 21):
            if myTree.traverseTree(pd.DataFrame(data={0: i, 1: j}, index = [0]), myTree.root) == 1:
                for a in range((i-1)*10, i*10):
                    for b in range((j-1)*10, j*10):
                        im.putpixel((a,199-b), (0,0,255))
            else: 
                for a in range((i-1)*10, i*10):
                    for b in range((j-1)*10, j*10):
                        im.putpixel((a,199-b), (255,0,0))
    
    plt.imshow(im, extent=[x_min,x_max,y_min,y_max])
    plt.title(filename, y=1.1)
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 2')
    plt.show()

# Extra Credit Function
# Find the best number of bins for the synthetic data
def bestBinNumber(data):
    bins = [5, 10, 20] # Calculate the accuracy for 5, 10, or 20 bins
    res = 0
    accuracy = 0
    for i in bins:
        data = syntheticData(data, len(data)//i)
        myTree = Tree(max_depth=3, num_attributes=i)     # Create a Tree
        parent = entropy(data.iloc[:, -1].tolist())          # Calculate parent entropy
        myTree.growMyTree(data, 1, 0, parent, 0)             # Build the Tree

        count = 0
        # Traverse the tree for every piece of data and calculate the accuracy of our Tree
        for j in range(len(data)):
            if int(data.iat[j, -1]) == myTree.traverseTree(data.iloc[[j]], myTree.root):
                count += 1

        percent = 100*(count/len(data))

         # If this bin number produces a better accuracy update the variables
        if percent >= accuracy:
            accuracy = percent
            res = i
    return res


def main():
    filenames = ["synthetic-1.csv", "synthetic-2.csv", "synthetic-3.csv", "synthetic-4.csv"]

    # Read in and discretize the synthetic data files
    filenames = ["synthetic-1.csv", "synthetic-2.csv", "synthetic-3.csv", "synthetic-4.csv"]
    data = []
    data.append(pd.read_csv('synthetic-1.csv', header=None))
    data.append(pd.read_csv('synthetic-2.csv', header=None))
    data.append(pd.read_csv('synthetic-3.csv', header=None))
    data.append(pd.read_csv('synthetic-4.csv', header=None))

    # For every synthetic data file
    for i in range(len(data)):
        # Extra Credit: Find best number of bins
        bin = bestBinNumber(data[i])

        newData = syntheticData(data[i], len(data[i])//bin)
        myTree = Tree(max_depth=3, num_attributes=bin)          # Create a Tree
        parent = entropy(newData.iloc[:, -1].tolist())          # Calculate parent entropy
        myTree.growMyTree(newData, 1, 0, parent, 0)             # Build the Tree

        count = 0
        # Traverse the tree for every piece of data and calculate the accuracy of our Tree
        for j in range(len(newData)):
            if int(newData.iat[j, -1]) == myTree.traverseTree(newData.iloc[[j]], myTree.root):
                count += 1
        print(filenames[i], "Tree has an accuracy of", 100*(count/len(newData)), "%")

        # Create an image for the decision tree
        #approximateImage(myTree, data[i], filenames[i])

    # Read in pokemon stats and add class label to last column
    data = pd.read_csv('pokemonStats.csv', skiprows=1, header=None)
    label = pd.read_csv('pokemonLegendary.csv', header=None)
    data.insert(44, 44, label.iloc[1:, -1].tolist())

    # Make True = 1 and False = 0
    for i in range(len(data)):
        if str(data.iat[i, -1]) == "True":
            data.iat[i, -1] = 1
        else: data.iat[i, -1] = 0

    data = pokemon(data, 70)                             # Discretize data

    myTree = Tree(max_depth=3, num_attributes=21)        # Create a Tree
    parent = entropy(data.iloc[:, -1].tolist())          # Calculate parent entropy
    myTree.growMyTree(data, 1, 0, parent, 0)             # Build the Tree

    count = 0
    # Traverse the tree for every piece of data and calculate the accuracy of our Tree
    for j in range(len(data)):
        if data.iat[j, -1] == myTree.traverseTree(data.iloc[[j]], myTree.root):
            count += 1
            
    print("Pokemon Tree has an accuracy of", 100*(count/len(data)), "%")

if __name__ == "__main__":
    main()