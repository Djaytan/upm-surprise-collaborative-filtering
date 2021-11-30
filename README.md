# upm-surprise-collaborative-filtering

Program is run by executing ```python3 main.py [arg1 [arg2 ...]]```

## Exercise 1

Parameters of command:

* Exercise to run: here "e1"
* The size of test set to allocate (between 0 and 1) (here 0.25 for 25%)
* The start value of K to analysis
* The end value of K to analysis (excluded)
* The step of the K values (the increment over each iteration of the analysis)
* The number of experiments (recommended value: at least 30)

### Exercise 1.a - Search K value for KNNa algorithm

The command to run the solution is : ```python3 main.py e1 .25 1 200 1 30``` (~30 minute of computation)  
=> A fast version: ```python3 main.py e1 .25 40 80 4 1``` (~30 seconds of computation)

This command calculates and plots the optimal value for k with a size of 25% for tests.

### Exercise 1.b - Sparsity problem

The command to run the solution is : ```python3 main.py e1 .75 1 200 1 30```  
=> A fast version: ```python3 main.py e1 .75 40 80 4 1``` (~30 seconds of computation)

This command calculates and plots the optimal value for k with a size of 75% for tests.

## Exercise 2

The Command for exercise 2 is ```python3 main.py e2```

To compare the MAE values of SVD and KNN run the command ```python3 main.py e2 compare```

## Exercise 3

The command for exercise 3 is ```python3 main.py e3 KNN``` to calculate precision, recall and F1s values for the KNN
algorithm.

The command ```python3 main.py e3 SVD``` calculates them for the SVD algorithm.