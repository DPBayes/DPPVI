# DP-PVI

Implementation of [Differentially Private Partitioned Variational Inference](https://openreview.net/forum?id=55BcghgicI). The implementation is based on the original [PVI](https://arxiv.org/abs/2202.12275) code base.

## Basic setup before running experiments

Install dependencies in notebooks/examples/requirement.txt (pip) and environment.yml (Conda/Mamba). Download and preprocess required data sets.

## Running experiments

All codes for running experiments are located in notebooks/examples.

To run logistic regression model experiments, use 'dp_logistic_regression.py' after checking the argparser arguments. For BNN, use instead 'run_bnn.py'.

[![DOI](https://zenodo.org/badge/626846942.svg)](https://zenodo.org/badge/latestdoi/626846942)
