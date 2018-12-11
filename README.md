# Training and hyperparamter optimization for neural networks on the ISIC 2017 dataset

Toy project for training neural networks on the ISIC 2017 skin cancer dataset using convolutional neural network (CNN)-based transfer learning and the HpBandSter hyperparameter optimization model. 

The repo consists of 4 files:
- DermAI.ipynb -- a Jupyter notebook for training a neural network on the dataset and generating a CSV file of predicted results.
- DermAI_TwoParamHpBandSter.ipynb -- a Jupyter notebook for optimizing CNN parameters using the HpBandSter framework.
- DermAI_TwoParamHpBandSter.py -- same as above but in Python script format.
- Visualizing HpBandSter Results.ipynb -- a Jupyter notebook for visualizing the results obtained using HpBandSter.

# Running environment
These files have been tested:
 - On an Amazon AWS p2.xlarge instance
 - Using the following AMI: Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-1197bd74)
 - Using the conda environment listed in dermAIenv.yml
