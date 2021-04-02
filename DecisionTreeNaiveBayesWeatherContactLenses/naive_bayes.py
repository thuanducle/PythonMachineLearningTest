#-------------------------------------------------------------------------
# AUTHOR: Thuan L
# FILENAME: naive_bayes.py
# SPECIFICATION: naive_bayes algorithm
# FOR: CS 4200- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []

def convertDBFeaturesToNumber(dbName, xArray, yArray):
    for index, element in enumerate(dbName):
        for indexInElement, elementName in enumerate(element):
            if indexInElement == 0:
                continue
            if indexInElement == 1:
                if elementName == "Sunny":
                    xArray[index][0] = 1
                elif elementName == "Overcast":
                    xArray[index][0] = 2
                elif elementName == "Rain":
                    xArray[index][0] = 3
            if indexInElement == 2:
                if elementName == "Hot":
                    xArray[index][1] = 1
                elif elementName == "Mild":
                    xArray[index][1] = 2
            if indexInElement == 3:
                if elementName == "High":
                    xArray[index][2] = 1
                elif elementName == "Normal":
                    xArray[index][2] = 2
            if indexInElement == 4:
                if elementName == "Weak":
                    xArray[index][3] = 1
                elif elementName == "Strong":
                    xArray[index][3] = 2
            if indexInElement == 5:
                if elementName == "Yes":
                    yArray.append(1)
                elif elementName == "No":
                    yArray.append(2)

#reading the training data
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTraining.append (row)


#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
size = len(dbTraining)
X = [[0] * 4 for i in range(size)]
Y = []
convertDBFeaturesToNumber(dbTraining, X, Y)


#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y = I dd it in the above code

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

dbTest = []
#reading the data in a csv file
#--> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append (row)
size = len(dbTest)
featuresConvertedTest = [[0] * 4 for i in range(size)]
convertDBFeaturesToNumber(dbTest,featuresConvertedTest,[])

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]

def convertFeatureToString(array,index, confidence, playOrNot):
    if playOrNot == 1:
        dbTest[index][5] = "Yes"
    else:
        dbTest[index][5] = "No"
    print(dbTest[index][0] , dbTest[index][1] , dbTest[index][2] , dbTest[index][3] , dbTest[index][4] , dbTest[index][5] , confidence)


for index, element in enumerate(dbTest):
    predicted = clf.predict_proba([featuresConvertedTest[index]])[0]
    if predicted[0] > 0.75:
        convertFeatureToString(featuresConvertedTest[index], index, predicted[0],1)
    if predicted[1] > 0.75:
        convertFeatureToString(featuresConvertedTest[index], index, predicted[1],2)


