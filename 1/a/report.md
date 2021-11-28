1. Given this data set and the algorithm of K-NN explained in class for user-based CF:

a) Find out the value for K that minimizes the MAE with 25% of missing ratings.

First of all, it's good to remind us our main goal: based on a dataset, predict the ratings of all items not rated yet
by a given user, and for all users. After that, we just want to recommend to user the top N items with the highest
predicted rating by expecting that he will like them (and then increase chances to sell the item for example). Of
course, according to algorithms used, the results can vary.

The objective for this first step is to test different values of K for K-NN algorithm and examine the results. We want
the lowest K that minimizes the MAE. We start here with 25% of missing ratings which mean that we define a training
dataset for the algorithm of 75% or ratings and a test one of 25% of the remaining ones.

Because we use here K-NN algorithm, we must select the similarity function to use. We are going to use here the
"Pearson" one in our case study.

To realize this, we set up this program by following these preliminary steps:

```python
# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split dataset in two ones: one for training, the other one to realize tests
# (to determine MAE or top-N items for example)
train_set, test_set = train_test_split(data, test_size=.25)

# Use the "Pearson" similarity function for K-NN algorithm
sim_options_knn = {
    'name': "pearson",
    'user_based': True  # compute similarities between users
}
```

Next, we run the K-NN algorithm several times with different values of K in order to determine MAEs values:

```python
# Adjust range values according to the experience to realize
k_list = list(range(40, 50, 1))

# Store MAEs values to display them later
maes = []

for k in k_list:
    algo = KNNBasic(k=k, sim_options=sim_options_knn)
    predictions = algo.fit(train_set).test(test_set)
    maes.append(mae(predictions))
```

Then, we display the result in a plot with MatPlotLib library:

```python
pyplot.plot(k_list, maes)
pyplot.title('MAE evolution depending on K value')
pyplot.xlabel('K nearest-neighbors')
pyplot.ylabel('MAE')
pyplot.show()
```

Now, we just need to change k_list in order to find the sought K-value for the given context of ratings.

***Note:** for the following experiments, we use the dichotomy approach (manually executed) to determine the K value.*

***Note 2:** we realize after experiments that the use of histogram would have been preferable for discrete values.
Anyway, results still exploitable, so we don't made change.*

We can try with `k_list = list(range(10, 10000, 1000))` as a first try, we see the following result:  
![img.png](img/img-1.png)

We see that when `K > 1000`, then MAE is equal to `0.8015` constantly. So we conclude here that the K shouldn't
overflow `1000` as value.

Let's try analyzing MAE values with K < 1000 with a new k_list: `k_list = list(range(10, 1000, 100))`:  
![img.png](img/img-13.png)
![img.png](img/img-14.png)

Here we have make calculations a second time to verify the stability of the phenomenon observed. Of course, additional
ones can improve the precision and certainty of this observation. We observe here that the K value must not be too low,
but not too high either. So, we conclude that `10 < K < 200`.

Let's try analyzing MAE values with a new k_list: `k_list = list(range(10, 200, 10))`:  
![img.png](img/img-15.png)
![img.png](img/img-16.png)

Again, it seems to confirm the phenomenon observed previously with these two other experiments. We conclude here
that `40 < K < 100`. Another experiment with `k_list = list(range(40, 100, 5))`:  
![img.png](img/img-17.png)
![img.png](img/img-18.png)

Here we see that it become harder to exclude values of K that can't be the lowest because of the unstable behavior of
the curve between 40 and 80. This is why we are going to stop using dichotomy approach and use another one:
probabilistic analysis. For the k_list defined as follows: `k_list = list(range(40, 80, 4))`, we relaunch calculations
30 times in order to permit quantitative analysis in order to conclude about which value of K we may use. We set a step
of 4 in order to make tests faster even if we lose a bit of precision. Anyway, this is just a case study so this last
criterion does not highly constraint us. We assume that the ideas explained and steps followed here are more important.

This is a snippet of the developed algorithm:

```python
# Number of experiments (>= 30 to permit quantitative analysis)
n = 30

# We focus on the range of values of K where MAEs remain unstable
k_list = list(range(40, 80, 4))

# Matrix of MAEs values: row correspond to an experiment and column to a K value
# A way to read a cell: for the experiment n°X with K=Y the MAE value is equal to Z (the value of the cell)
m_maes = np.zeros((n, len(k_list)))

for i in range(0, n, 1):
    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=.25)

    # Store MAEs values of the current experiment
    maes = []

    for k in k_list:
        algo = KNNBasic(k=k, sim_options=sim_options_knn)
        predictions = algo.fit(train_set).test(test_set)
        maes.append(mae(predictions))

    # Update the corresponding in the matrix of MAEs values
    m_maes[i, :] = maes

# Best K information
best_k_index = -1
best_k_mae_value = 0

average_maes_per_k = np.zeros(len(k_list))

# Analyse MAEs values for each K and search the best one
for k_index in range(0, len(k_list), 1):
    k_maes = m_maes[:, k_index]
    average_maes = sum(k_maes) / len(k_maes)
    average_maes_per_k[k_index] = average_maes
    if k_index == 0 or average_maes < best_k_mae_value:
        best_k_index = k_index
        best_k_mae_value = average_maes
    print("K={}, average MAEs={}".format(k_list[k_index], average_maes))

# Display the best value of K
if best_k_index >= 0:
    best_k = 40 + best_k_index * 4
    print("Best K: {}, average MAEs: {}".format(best_k, best_k_mae_value))
else:
    print("No best K found")

# Plot a curve of average MAEs per K
plot_maes(k_list, average_maes_per_k)
```

When running it, we obtain the following results:

```
K=40, average MAEs=0.8059287110554011
K=44, average MAEs=0.8057077602282138
K=48, average MAEs=0.8055491029694546
K=52, average MAEs=0.8054369597826778
K=56, average MAEs=0.8053985015884169
K=60, average MAEs=0.8053677454013536
K=64, average MAEs=0.8053894656489773
K=68, average MAEs=0.8054018452040784
K=72, average MAEs=0.8054459548081628
K=76, average MAEs=0.8054883515342929
Best K: 60, average MAEs: 0.8053677454013536
```

![img.png](img/img-19.png)

Like we can see, the curve seem to be more stable and smooth. If we do experiment with only n=1 the difference is
blatant:  
![img.png](img/img-21.png)
![img.png](img/img-22.png)

Finally, we found the better value of K according to the steps followed until now and the use of MAE as measure.

The results may vary if we use another measure like RMSE but in this case study we are not going to experiment it in
order to compare both measures.
