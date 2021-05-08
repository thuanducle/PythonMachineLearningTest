#-------------------------------------------------------------------------
# AUTHOR: Thuan Le
# FILENAME: association_Rule_mining.py
# SPECIFICATION: retail dataset using association rule mining to predict most likely item with minsup > 0.2 and conf > 0.6
# FOR: CS 4200- Assignment #5
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library
minsup = 0.2
minconf = 0.6
#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below
label = {
    "+": 1,
    "-": 2,
}
encoded_vals = []
for index, row in df.iterrows():
    #Intialize everything to empty
    bread = 0
    wine = 0
    eggs = 0
    meat = 0
    cheese = 0
    pencil = 0
    diaper = 0
    for data in row:
        if data == 'Bread':
            bread = 1
        elif data == 'Wine':
            wine = 1
        elif data == 'Eggs':
            eggs = 1
        elif data == 'Meat':
            meat = 1
        elif data == 'Cheese':
            cheese = 1
        elif data == 'Pencil':
            pencil = 1
        elif data == 'Diaper':
            diaper=1
    labels = {
    'Bread' : bread,
    'Wine' : wine,
    'Eggs' : eggs,
    'Meat' : meat,
    'Cheese' : cheese,
    'Diaper' : diaper,
    'Pencil' : pencil

    }

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
for index, data in enumerate(rules.consequents):
    supportCount = 0
    antecedents = rules.antecedents[index]
    consequents = rules.consequents[index]
    confidence = rules.confidence[index]

    # calculate supportCount
    listOfConsequents = consequents.split(",")
    count = len(listOfConsequents)

    for index2, row in df.iterrows():
        for rowData in row:
            if rowData in listOfConsequents:
                supportCount += 1

    #Calculate Prior
    prior = supportCount / len(encoded_vals)
    #Calculate GainInConfidence
    gainInConfidence = str(100 * (confidence - prior) / prior)
    #Printing
    print(antecedents, "->", consequents)
    print("Confidence: ",confidence)
    print("Prior: ",prior)
    print("Gain in Confidence: ", gainInConfidence)
    print()

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()