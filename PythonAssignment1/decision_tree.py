#-------------------------------------------------------------------------
# AUTHOR: Thuan Le
# FILENAME: decosion_tree.py
# SPECIFICATION: Applying ID3 algorithm
# FOR: CS 4200- Assignment #1
# TIME SPENT: 2 hours because I dont have prior knowledge of python
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
       #  print(row)

#transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
size = len(db)
X = [[0]*4 for i in range(size)]
Y = []

for index, element in enumerate(db):
    for indexInElement, elementName in enumerate(element):
        if indexInElement == 0:
            if elementName == "Young":
                X[index][0] = 1
            elif elementName == "Prepresbyopic":
                X[index][0] = 2
            elif elementName == "Presbyoppic":
                X[index][0] = 3
        if indexInElement == 1:
            if elementName == "Myope":
                X[index][1] = 1
            elif elementName == "Hypermetrope":
                X[index][1] = 2
        if indexInElement == 2:
            if elementName == "Yes":
                X[index][2] = 1
            elif elementName == "No":
                X[index][2] = 2
        if indexInElement == 3:
            if elementName == "Reduced":
                X[index][3] = 1
            elif elementName == "Normal":
                X[index][3] = 2
        if indexInElement == 4:
            if elementName == "Yes":
                Y.append(1)
            elif elementName == "No":
                Y.append(2)
print(X)
print(Y)


#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# I did this part in the above loop, save some time

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()


