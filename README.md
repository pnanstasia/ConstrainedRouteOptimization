# Optimization of Daily Route Planning for a Logistic Company
This repository contains synthetic data, code and experiments for Bachelor's Thesis Constrained Optimization of Daily Route Planning for a Logistic Company.

This thesis formulates the problem as a Vehicle Routing Problem with Time Windows and proposes a three-stage optimization pipeline. The approach combines clustering-based decomposition using K-medoids, Agglomerative clustering, and Gaussian Mixture Models, greedy construction of initial routes with route start-time selection, and iterative improvement using Variable Neighborhood Search and Adaptive Large Neighborhood Search.

The results demonstrate that combining K-medoids clustering with Adaptive Large Neighborhood Search for large instances and with Variable Neighborhood Search for smaller ones provides a scalable and efficient solution, significantly reducing route duration and improving planning efficiency.

# Code structure