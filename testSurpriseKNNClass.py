from pprint import pprint

import pandas as pd
import numpy as np
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

def load_csv():
    # # Dataset Load

    # In[2]:

    # import csv file in python
    csv_file = pd.read_csv('data.csv', delimiter=';')

    # conversion of the table in a numpy matrix with only ratings
    # method .to_numpy() does not copy the header row
    # np.delete deletes the first column

    temp = np.delete(csv_file.to_numpy(), np.s_[0], axis=1)

    # Matrix is converted in User-Movies from Movies-User
    # Later, the matrix is flatten to a vector of values.
    # For each user, all the movie ratings are reported
    # user1item1, user1item2, user1item3,...

    ratings = temp.T.flatten()

    # Vectors users and movies are the corresponding columns in the dataframne.
    # As the ratings are ordered user1[allratings], user2[allratings], ...
    # the user and movies vectors follow the same logic

    i = 0
    j = 0
    users = []
    movies = []
    users.clear()
    movies.clear()
    while i < 50:
        while j < 20:
            movies.append(j)
            users.append(i)
            j += 1
        j = 0
        i += 1

    movies = np.array(movies)
    users = np.array(users)

    # The user, movies, and rating numpy vectors are converted in a rating dictionary
    # and later, in a Pandas dataframe

    ratings_dict = {'userID': users,
                    'itemID': movies,
                    'rating': ratings}

    df = pd.DataFrame(ratings_dict)

    # The dataframes are converted into a dataset suitable for Surprise

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# Dataset splitting in trainset and testset for 25% sparsity

trainset25, testset25 = train_test_split(data, test_size=.25,
                                         random_state=22)

sim_options_KNN = {'name': "pearson",
                   'user_based': True  # compute similarities between users
                   }
# number of neighbors
k = 11

# prepare user-based KNN for predicting ratings from trainset25
algo = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
algo.fit(trainset25)

# estimate the ratings for all the pairs (user, item) in testset25
predictions25KNN = algo.test(testset25)

# pprint(predictions25KNN)

# the first user has uid=0 and first item iid=0
for (uid, iid, real, est, _) in predictions25KNN:
    if uid == 1:
        print(f'{uid} {iid} {real} {est}')

mae(predictions25KNN)
