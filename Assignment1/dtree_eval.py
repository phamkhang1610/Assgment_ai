"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

# Set value of numOfTrials and numOfFoldsPerTrial
numOfFoldsPerTrial = 10

numOfTrials = 100


def evaluatePerformance(numTrials=numOfTrials):
    """
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree

    ** Note that your implementation must follow this API**
    """

    # Load data from dataset SPECTF.dat
    filename = "./Assignment1/data/SPECTF.dat"
    data = np.loadtxt(filename, delimiter=",")
    
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # create lists to store data
    treeAccuracies = []
    stumpAccuracies = []
    dt3Accuracies = []
    
    training_sizes = np.linspace(1, 100, 100)
    print(training_sizes)

    # perform 100 trials
    for x in range(0, numTrials):
        # shuffle the data
        idx = np.arange(n) # idx = [0, 1,... n-1]
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Split the data randomly into 10 folds
        folds = []
        interval_divider = len(X) / numOfFoldsPerTrial
        for fold in range(0, numOfFoldsPerTrial):
            # design a new testing range
            Xtest = X[int(fold * interval_divider) : int((fold + 1) * interval_divider) :]
            ytest = y[int(fold * interval_divider) : int((fold + 1) * interval_divider) :]
            Xtrain = X[: int((fold * interval_divider)) :]
            ytrain = y[: int((fold * interval_divider)) :]
            Xtrain = Xtrain.tolist()
            ytrain = ytrain.tolist()

            # complete the training data set so that it contains all
            # data except for the current test fold
            for dataRow in range(int((fold + 1) * interval_divider), len(X)):
                Xtrain.append(X[dataRow])
                ytrain.append(y[dataRow])

            # train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain, ytrain)

            # train the 1-level decision tree
            oneLevel = tree.DecisionTreeClassifier(max_depth=1)
            oneLevel = oneLevel.fit(Xtrain, ytrain)

            # train the 3-level decision tree
            threeLevel = tree.DecisionTreeClassifier(max_depth=3)
            threeLevel = threeLevel.fit(Xtrain, ytrain)

            # output predictions on the remaining data
            y_pred_tree = clf.predict(Xtest)
            y_pred_stump = oneLevel.predict(Xtest)
            y_pred_dt3 = threeLevel.predict(Xtest)

            # compute the training accuracy of the model and save to the
            # list of all accuracies
            treeAccuracies.append(accuracy_score(ytest, y_pred_tree))
            stumpAccuracies.append(accuracy_score(ytest, y_pred_stump))
            dt3Accuracies.append(accuracy_score(ytest, y_pred_dt3))

    # Update these statistics result
    meanDecisionTreeAccuracy = np.mean(treeAccuracies)
    stddevDecisionTreeAccuracy = np.std(treeAccuracies)
    yTreeAccuracies = []
    for i in range(0, len(treeAccuracies), 10):
        yTreeAccuracies.append(np.mean(treeAccuracies[i : i + 9]))
    print(len(yTreeAccuracies))
    
    # stump
    meanDecisionStumpAccuracy = np.mean(stumpAccuracies)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracies)
    yStumpAccuracies = []
    for i in range(0, len(treeAccuracies), 10):
        yStumpAccuracies.append(np.mean(stumpAccuracies[i : i + 9]))
    print(len(yTreeAccuracies))

    # dt3
    meanDT3Accuracy = np.mean(dt3Accuracies)
    stddevDT3Accuracy = np.std(dt3Accuracies)
    yDT3Accuracies = []
    for i in range(0, len(treeAccuracies), 10):
        yDT3Accuracies.append(np.mean(dt3Accuracies[i : i + 9]))
    print(len(yTreeAccuracies))
    
    # plot chart
    # plt.plot(training_sizes, yTreeAccuracies, training_sizes, yStumpAccuracies, training_sizes, yDT3Accuracies);
    # plt.title("Tree Decision Accuracy")
    # plt.show()

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanDecisionStumpAccuracy
    stats[1, 1] = stddevDecisionStumpAccuracy
    stats[2, 0] = meanDT3Accuracy
    stats[2, 1] = stddevDT3Accuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
