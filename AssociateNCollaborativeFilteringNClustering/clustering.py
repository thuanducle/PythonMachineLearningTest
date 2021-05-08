#-------------------------------------------------------------------------
# AUTHOR: Thuan Le
# FILENAME: clustering.py
# SPECIFICATION: Using normal clustering with random training data, compare with actual test data. Predict highest KClusteringGroups
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

X_training = [];
#assign your training data to X_training feature matrix
for data in df.values:
    X_training.append(data)

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
arrayOfValues =[]
kValue = []
silValue =[]
for k in range(2,21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    silScore = silhouette_score(X_training, kmeans.labels_)
    tempPair = (k,silScore)
    kValue.append(k)
    silValue.append(silScore)
    arrayOfValues.append(tempPair)
     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
maxkValue = kValue[silValue.index(max(silValue))]
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(kValue,silValue)
plt.show()
#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
testData = pd.read_csv('testing_data.csv', sep=',', header=None)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(testData).reshape(1,-1)[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters= maxkValue, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
