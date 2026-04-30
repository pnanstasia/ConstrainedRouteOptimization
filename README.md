# Optimization of Daily Route Planning for a Logistic Company
This repository contains synthetic data, code and experiments for Bachelor's Thesis Constrained Optimization of Daily Route Planning for a Logistic Company.

# Abstract
Daily route planning in logistics companies is a complex task due to operational constraints such as time windows, service durations, and limited crew availability. In practice, many companies rely on manual or semi-manual planning, which is time-consuming and may require up to 40% of a planner’s working time.

This thesis formulates the problem as a Vehicle Routing Problem with Time Windows and proposes a three-stage optimization pipeline. The approach combines clustering-based decomposition using K-medoids, Agglomerative clustering, and Gaussian Mixture Models, greedy construction of initial routes with route start-time selection, and iterative improvement using Variable Neighborhood Search and Adaptive Large Neighborhood Search.

Experiments on synthetic datasets representing four Ukrainian cities with 30–600 service points, along with validation on real-world data, show that K-medoids and agglomerative clustering outperform Gaussian Mixture Models. Adaptive Large Neighborhood Search reduces total route duration by up to 37% for large-scale instances, while Variable Neighborhood Search achieves competitive results for smaller datasets.

The results demonstrate that combining K-medoids clustering with Adaptive Large Neighborhood Search for large instances and with Variable Neighborhood Search for smaller ones provides a scalable and efficient solution, significantly reducing route duration and improving planning efficiency.

# Code structure

The folder ***data*** contains 2 folders with datasets, 2 Python files that were used to generate the data and 1 notebook for data visualization.
- folder *synthetic_data_kyiv_varash* - contains data for Kyiv and Varash
- folder *synthetic_data_ternopil_dubno* - contains data for Ternopil and Dubno
- *generate_synthetic_data.py* - file for generation data for Kyiv and Varash, contains 25 file for each city with points to be served
- *generate_data_simple.py* - file for generation data for Ternopil and Dubno, contains 25 file for each city with points to be served
- *visualization_sythetic_data.ipynb* - notebook for visualization of synthetic data
