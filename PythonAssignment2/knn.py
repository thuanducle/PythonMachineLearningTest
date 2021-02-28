#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

def convertDBFeaturesToNumber(dbName, xArray, yArray,indexToExclude):
    counterX = 0;
    for index, element in enumerate(dbName):
        if index == indexToExclude:
            continue
        xArray[counterX][0] = int(element[0])
        xArray[counterX][1] = int(element[1])
        if element[2] == "+":
            yArray.append(1)
        else:
            yArray.append(2)
        counterX = counterX + 1

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

size = len(db)

wrongPrediction = 0
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    X = [[0] * 2 for j in range(size-1)]
    Y = []
    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    #--> add your Python code here
    convertDBFeaturesToNumber(db, X, Y, i)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    #--> add your Python code here
    # Y = I did it in above code

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [int(instance[0]), int(instance[1])]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]
    if class_predicted == 1:
        class_predicted = '+'
    else:
        class_predicted = '-'
    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if (class_predicted != instance[2]):
        wrongPrediction= wrongPrediction+1

#print the error rate
#--> add your Python code here
print(wrongPrediction/size)





