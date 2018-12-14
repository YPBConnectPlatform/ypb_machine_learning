# Training and hyperparamter optimization for neural networks on the ISIC 2017 dataset

Training neural networks on the ISIC 2017 skin cancer dataset using convolutional neural network (CNN)-based transfer learning in PyTorch and the HpBandSter hyperparameter optimization model. 

The repo consists of 4 files:
- DermAI.ipynb -- a Jupyter notebook for training a neural network on the dataset and generating a CSV file of predicted results.
- DermAI_TwoParamHpBandSter.ipynb -- a Jupyter notebook for optimizing CNN parameters using the HpBandSter framework.
- DermAI_TwoParamHpBandSter.py -- same as above but in Python script format.
- Visualizing HpBandSter Results.ipynb -- a Jupyter notebook for visualizing the results obtained using HpBandSter.

# Running environment
These files have been tested:
 - On an Amazon AWS p2.xlarge instance
 - Using the following AMI: Deep Learning AMI (Ubuntu) Version 20.0 (ami-0c9ae74667b049f59)
 - Using the conda environment listed in dermAIenv.yml
 
 # Install
 - Start an AWS p2.xlarge instance using the AMI mentioned. Make sure that you have at least 100 GB of storage and that port 8888 is accessible in your security group (Jupyter notebooks use port 8888 by default). 
 - Run the following code to get everything started up.
 ```
cd /home/ubuntu
git clone https://github.com/paulbisso/ISIC2017_with_optimization
mv ISIC2017_with_optimization/startup.sh startup.sh
chmod +x startup.sh
./startup.sh
tmux
```
- In the new terminal window, run:
```
conda activate derm-ai
cd /home/ubuntu/src/derm-ai
cp ISIC2017_with_optimization/DermaAI_TwoParamHpBandSter.ipynb DermaAI_TwoParamHpBandSter.ipynb
cp ISIC2017_with_optimization/DermaAI.ipynb DermaAI.ipynb
cp "ISIC2017_with_optimization/Visualizing HpBandSter Results.ipynb" "Visualizing HpBandSter Results.ipynb"
jupyter notebook --ip=0.0.0.0 --no-browser
```
- Next, copy everything starting from ":8888" to a browser address bar on your local machine.
- In front of the ":8888" paste the public IPv4 address of the instance and open, and Jupyter will open and be ready to go!
- If you like, from that point, you can use the tmux command "CTRL + b, d" to detach from that terminal window and keep working (or close your ssh / PuTTY session without deactivating the Jupyter notebook session)
