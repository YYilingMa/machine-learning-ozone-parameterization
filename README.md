# machine-learning-ozone-parameterization
Code for the submission "mloz: A Highly Efficient Machine Learning-Based Ozone Parameterization for Climate Sensitivity Simulations"
Author: Yiling Ma - yiling.ma@kit.edu

# Repository description
mloz_parameterization_offline_training: Python scripts used for training the mloz ridge coefficients offline, using piCTRL and 4xCO2 full chemistry simulations from UKESM. See below for repository for traning datasets. The trained ridge coefficients are also provided in the repository below.
mloz_UKESM_implementation: Fortran scripts that contain various subroutines for mloz calculation, data fetching and MPI broadcast within UKESM-UKCA.
mloz_ICON_implementation: Fortran scripts for mloz caculation and vertical interpolation within ICON-ART.
simulation_data_postprocessing_and_visualization: Python scripts for analysing the climate model simulations with mloz. 

# Data
The training datasets for the mloz, and the trained mloz coefficients for piCTRL and 4xCO2 runs are provided on Zenodo:


