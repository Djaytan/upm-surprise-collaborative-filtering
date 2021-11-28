from matplotlib import pyplot
from surprise import Dataset
from surprise import KNNBasic
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

sim_options_KNN = {'name': "pearson",
                   'user_based': True  # compute similarities between users
                   }
# number of neighbors
k_list = list(range(40, 50, 1))
print(len(k_list))
maes = []
trainset, testset = train_test_split(data, test_size=.25)

for k in k_list:
    print(k)
    algo = KNNBasic(k=k, sim_options=sim_options_KNN)
    predictions = algo.fit(trainset).test(testset)
    maes.append(mae(predictions))

pyplot.plot(k_list, maes)
pyplot.title('MAE evolution depending on K value')
pyplot.xlabel('K value')
pyplot.ylabel('MAE value')
pyplot.show()
