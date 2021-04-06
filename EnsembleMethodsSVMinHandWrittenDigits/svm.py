#-------------------------------------------------------------------------
# AUTHOR: Thuan L
# FILENAME: svm.py
# SPECIFICATION: Utilize svm techniques to predict pattern of handwritten digits based on a 32x32 bitsmap
# FOR: CS 4200- Assignment #3
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[len(row)-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append(row)


#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for i,cValue in enumerate(c) : #iterates over c
    for j,dValue in enumerate(degree)  : #iterates over degree
        for k,kValue in enumerate(kernel) : #iterates kernel
           for m, dsValue in enumerate(decision_function_shape) : #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C= cValue,degree = dValue, kernel= kValue,decision_function_shape= dsValue)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)
                accuracy = 0
                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                for i, testSample in enumerate(dbTest):
                    testSampleBits = testSample[:-1]
                    testSampleResult = int(testSample[len(testSample) - 1])
                    class_predicted = clf.predict([testSampleBits])[0]
                    classPredictedInt = int(class_predicted)
                    if (classPredictedInt == testSampleResult):
                        accuracy = accuracy + 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = accuracy/len(dbTest)
                if accuracy >= highestAccuracy:
                    highestAccuracy = accuracy
                    print("Highest SVM accuracy so far " + str(highestAccuracy) + " Parameters: c: " + str(cValue) +" degree: " + str(dValue) + "Kernel: " + str(kValue) + " Decision_function_shape: " + str(dsValue))

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
#--> add your Python code here












