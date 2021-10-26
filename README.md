## About
Final project for "Mathematics for engeneers" course. 
This repository is devoted to the PageRank problem.
Here you can find the implementation of Transition matrices (Erdős–Rényi model and Bollobás–Riordan), description of possible problems (dangling, segmented, looped) for generated matrices, and corrections to avoid such problems (random connections, weak connections with dampling factor). 
Also, here we have three kinds of solvers: 1) based on NumPy eig function. 2) Power method. 3) Markov chain Monte-Carlo approach. 

Finally, this package can be split into two: "slow" (with usual python implementation) and "fast" with jitted python implementation.

## How to use it?
In order to repeat our experiments, you can use `main.ipynb` file. If you want to use our package, just use the first cell from `main.ipynb` to download the module and then use it as you with.

## Our reults
Firstly, we tried to understand when each kind of problem can appear in order to use parametrs where it's difficult to get some one.
To do this for Erdős–Rényi model we created a grid of matrix sizes and probabilities for nodes to be connected, then for each point of grid we generated 5000 matrices and check them for problems. On the following graph the color represents the probability to get matrix without problems: 

2D plot                                                    |  3D plot
:---------------------------------------------------------:|:-------------------------:
<img src="./img/ER1.png" width="90%" height="90%">  |  <img src="./img/ER2.png" width="100%" height="100%"> 

On the following graphs we checked how often each king of problem appear. Here colors represents probability to get specific kind of the problem 
<p align="center">
<img src="./img/ER3.png" width="90%" height="90%"> 
</p>


# Power method convergence method 
Power method convergers very fast (for some cases even for 2-3 iterations). However, it converges fast in case if ratio between 2 max eigenvalues are low (they are very different). In case of segmented matrices we have two almost equal eiqenvalues, it such case algoritm converges randomly to any of eigenvector which correspond to max eigenvalues. That's why the result is quite bad.

No problems                                                |  Segmented matrix
:---------------------------------------------------------:|:-------------------------:
<img src="./img/ER6.png" width="90%" height="90%">  |  <img src="./img/ER8.png" width="100%" height="100%"> 


# Markov chain Monte-Carlo convergence method
This method converges slowly, than Power method, but it works faster in case of high-dimentional cases, where matrix multiplication is hard to do. In case of single chain we also have the problem of convergence to random eigenvector (because from one part of matrix it's impossibe to go to another one). However in case of parallel implementation, we can combine results from different processes and get correct results. 
No problems                                                |  Segmented matrix
:---------------------------------------------------------:|:-------------------------:
<img src="./img/ER4.png" width="100%" height="100%">  |  <img src="./img/ER5.png" width="100%" height="100%"> 
