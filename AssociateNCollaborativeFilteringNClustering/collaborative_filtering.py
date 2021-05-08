#-------------------------------------------------------------------------
# AUTHOR: Thuan Le
# FILENAME: collaborative_filtering.py
# SPECIFICATION: utilize collaborative filtering in a sociocultural training dataset with user-based technique, among with 10 similar users with cosine similarity to find the predicted rating for galleries and restaurants of the 100th users
# FOR: CS 4200- Assignment #5
# TIME SPENT: 2:30 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import heapq

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here

UserData = df.drop(['User ID', 'galleries', 'restaurants'],axis=1).values
cosArray = []
for i in range(0,99):
    cosValue = cosine_similarity([UserData[99]],[UserData[i]])
    cosArray.append(cosValue[0][0])

   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here
cos10Largest = heapq.nlargest(10,cosArray) #Extract top 10 cosine similarity
similarUsersWithoutColumns = [] #This is storing all columns without user id, galleries and restaurants column
similarUsersColumns = [] #One column is only galleries score, the other is restaurants score
for i in cos10Largest:
    tempIndex = cosArray.index(i)
    similarUsersWithoutColumns.append(df.drop(['User ID', 'galleries', 'restaurants'],axis=1).values[tempIndex])
    similarUsersColumns.append(df.loc[:, ['galleries','restaurants']].values[tempIndex])

print()
   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here
averagePredictedUser = sum(UserData[99])/len(UserData[99])
averageRatingNumForGalleries=0
averageRatingNumForRestaurants=0
averageRatingDem= sum(cos10Largest)
for index,row in enumerate(similarUsersWithoutColumns):
    #calculate average rating from 1 user
    tempAverageOf1Users = sum(row)/len(row)
    #less average rating from galleries score of that user and multiply by that user's similarity
    galleriesRatingForThatUser = float(similarUsersColumns[index][0])
    restaurantsRatingForThatUser = float(similarUsersColumns[index][1])
    averageRatingNumForGalleries += cos10Largest[index]*(galleriesRatingForThatUser - tempAverageOf1Users)
    averageRatingNumForRestaurants += cos10Largest[index]*(restaurantsRatingForThatUser - tempAverageOf1Users)

finalRatingForGalleries = averagePredictedUser + averageRatingNumForGalleries/averageRatingDem
finalRatingForRestaurants = averagePredictedUser + averageRatingNumForRestaurants/averageRatingDem

print("Rating for Gallery: ", finalRatingForGalleries, "Rating for Restaurants: ",finalRatingForRestaurants)


