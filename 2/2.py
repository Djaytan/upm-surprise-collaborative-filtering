import numpy as np
from matplotlib import pyplot
from surprise import Dataset
from surprise import KNNBasic
from surprise import SVD
from surprise.accuracy import mae
from surprise import accuracy
from surprise.model_selection import train_test_split

def runSVD(data, _testSize):
    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainset, testset = train_test_split(data, test_size=_testSize)

    # We'll use the famous SVD algorithm.
    algo = SVD()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    return mae(predictions)


barX = ["25% Sparcity", "75% Sparcity"]
barY = []

data = Dataset.load_builtin('ml-100k')

barY.append(runSVD(data, .25))
barY.append(runSVD(data, .75))

pyplot.bar(barX, barY)
pyplot.show()
