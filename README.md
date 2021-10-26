# About
Final project for "Mathematics for engeneers" course. 
This repository is devoted to the PageRank problem.
Here you can find implementation of Transition matrices (Erdős–Rényi model and Bollobás–Riordan), describtion of possible problems (dangling, segmented, looped) for generated matrices and corrections to avoid such problems (random connections, weak connections with dampling factor). 
Also, here we have three kinds of solvers: 1) based on numpy eig function. 2) Power method. 3) Markov chain Monte-Carlo approach. 

Finaly this package can be splitted into two: "slow" (with usual python implementation) and "fast" with jitted python implementation

