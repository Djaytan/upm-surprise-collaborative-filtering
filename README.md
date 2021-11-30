# upm-surprise-collaborative-filtering

Program is run by executing ```python3 main.py [arg1] [arg2]```

## Exercise 1

Parameters of command:
* Exercise to run: here "e1"
* The size of test set to allocate (between 0 and 1)
* The start value of K to analysis
* The end value of K to analysis (excluded)
* The step of the K values (the increment over each iteration of the analysis)
* The number of experiments (recommended value: at least 30)

**a)** the command to run the solution for exercise 1a is : ```python3 main.py e1 .25 1 200 30```  
(a fast version: ```python3 main.py e1 .25 40 80 4 1``` )  
This calculates and plots the optimal value for k with a sparcity of .25<br><br>
**b)** for exercise 1b it is : ```python3 main.py e1 .75 10 100 10```  
(faster version: ```python3 main.py e1 .75 40 80 4``` )  
This calculates and plots the optimal value for k with a sparcity of .75<br>

**Exercise 2**<br>
The Command for exercise 2 is ```python3 main.py e2```

To compare the MAE values of SVD and KNN run the cmmand ```python3 main.py compare```

**Exercise 3**<br>
The command for exercise 3 is ```python3 main.py e3 KNN``` to calculate all values for the KNN algorithm<br>
The command ```python3 main.py e3 SVD``` calculates for the SVD algorithm