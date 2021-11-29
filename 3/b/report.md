1. Top-N recommendations: calculate the precision, recall and F1 with different values for N (10..100) using user-based
   K-NN (with the best Ks)  and SVD. To do this, you must suppose that the relevant recommendations for a specific user
   are those rated with 4 or 5 stars in the data set.

a) With SVD

Similar to the first experiment we calculate the predictions for all users. This time we use the SVD algorithm.

```python
# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
evaluate_recommendations_users(predictions)
```
