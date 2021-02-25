# -------------------------------------------------------------------------
# AUTHOR: Thuan Le
# FILENAME: decision_tree(1).py
# SPECIFICATION: decision tree with accuracy and pruning from sklearn
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']


def convertDBFeaturesToNumber(dbName, xArray, yArray):
    for index, element in enumerate(dbName):
        for indexInElement, elementName in enumerate(element):
            if indexInElement == 0:
                if elementName == "Young":
                    xArray[index][0] = 1
                elif elementName == "Prepresbyopic":
                    xArray[index][0] = 2
                elif elementName == "Presbyopic":
                    xArray[index][0] = 3
            if indexInElement == 1:
                if elementName == "Myope":
                    xArray[index][1] = 1
                elif elementName == "Hypermetrope":
                    xArray[index][1] = 2
            if indexInElement == 2:
                if elementName == "Yes":
                    xArray[index][2] = 1
                elif elementName == "No":
                    xArray[index][2] = 2
            if indexInElement == 3:
                if elementName == "Reduced":
                    xArray[index][3] = 1
                elif elementName == "Normal":
                    xArray[index][3] = 2
            if indexInElement == 4:
                if elementName == "Yes":
                    yArray.append(1)
                elif elementName == "No":
                    yArray.append(2)

def convertArrayFeaturesToNumber(arrayName, xArray, yArray):
    for indexInElement, element in enumerate(arrayName):
            if indexInElement == 0:
                if element == "Young":
                    xArray[0] = 1
                elif element == "Prepresbyopic":
                    xArray[0] = 2
                elif element == "Presbyopic":
                    xArray[0] = 3
            if indexInElement == 1:
                if element == "Myope":
                    xArray[1] = 1
                elif element == "Hypermetrope":
                    xArray[1] = 2
            if indexInElement == 2:
                if element == "Yes":
                    xArray[2] = 1
                elif element == "No":
                    xArray[2] = 2
            if indexInElement == 3:
                if element == "Reduced":
                    xArray[3] = 1
                elif element == "Normal":
                    xArray[3] = 2
            if indexInElement == 4:
                if element == "Yes":
                    yArray.append(1)
                elif element == "No":
                    yArray.append(2)

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    size = len(dbTraining)
    X = [[0] * 4 for i in range(size)]
    Y = []
    convertDBFeaturesToNumber(dbTraining, X, Y)


    # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    # Y = ( I did it in the function above, convertFeaturesToNumber )

    lowestAccuracy = 1000000
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            sizeTest = len(dbTest)
            xTest = [0] * 4
            convertArrayFeaturesToNumber(data, xTest, [] )
            class_predicted = clf.predict([xTest])[0]
            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # --> add your Python code here
            trueLabel = data[4]
            predicted = class_predicted
            if (trueLabel == 'Yes' and predicted == 1):
                TP = TP + 1
            elif (trueLabel == 'Yes' and predicted == 2):
                FN = FN + 1
            elif (trueLabel == 'No' and predicted == 2):
                TN = TN + 1
            elif (trueLabel == 'No' and predicted == 1):
                FP = FP + 1
        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        dbTestAccuracy = (TP + TN)/(TP + TN + FP + FN)
        if( dbTestAccuracy < lowestAccuracy ):
            lowestAccuracy = dbTestAccuracy

    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that:
    # final accuracy when training on contact_lens_training_1.csv: 0.2
    # final accuracy when training on contact_lens_training_2.csv: 0.3
    # final accuracy when training on contact_lens_training_3.csv: 0.4
    # --> add your Python code here
    print("final accuracy when training on", ds, ": ", lowestAccuracy)
