# Training and hyperparamter optimization for neural networks on the ISIC 2017 dataset

Training neural networks on the ISIC 2017 skin cancer dataset using convolutional neural network (CNN)-based transfer learning in PyTorch and the HpBandSter hyperparameter optimization model. 

The repo consists of 4 files:
- DermAI.ipynb -- a Jupyter notebook for training a neural network on the dataset and generating a CSV file of predicted results.
- DermAI_TwoParamHpBandSter.ipynb -- a Jupyter notebook for optimizing CNN parameters using the HpBandSter framework.
- DermAI_TwoParamHpBandSter.py -- same as above but in Python script format.
- Visualizing HpBandSter Results.ipynb -- a Jupyter notebook for visualizing the results obtained using HpBandSter.

# Running environment
These files have been tested:
 - On Amazon AWS p2.xlarge(8xlarge;16xlarge) instances
 - Using the following AMI: Deep Learning AMI (Ubuntu) Version 20.0 (ami-0c9ae74667b049f59)
 - Using the conda environment listed in dermAIenv.yml
 
 # "Cold" Install
 Here I follow along with https://aws.amazon.com/blogs/machine-learning/scalable-multi-node-training-with-tensorflow/
 - Create an S3 bucket to store the image sets we'll be using (I call mine derm-ai-dataset). 
 - Start an AWS p2.xlarge (or .8xlarge or .16xlarge) instance using the AMI mentioned. Make sure that port 8888 is accessible in your security group (Jupyter notebooks use port 8888 by default). This instance will be used to download the image sets and will be your lead instance.
- Make sure that the instance you start has at least 200 GB of space and 10,000 IOPS -- you should use the Provisioned IOPS SSD (io1) choice for drive. 
- Make sure that you have an IAM role for EC2 with policy AmazonS3FullAccess, and that it is attached to the running EC2 instance. 
 - Run the following code to get everything started up.
 ```
cd /home/ubuntu
git clone https://github.com/paulbisso/ISIC2017_with_optimization
mv ISIC2017_with_optimization/startup_cold.sh startup_cold.sh
chmod +x startup_cold.sh
./startup_cold.sh
./startup_cool.sh
tmux
```
- In the new terminal window, run:
```
cd /home/ubuntu/src
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev
```
- Note that you may need to sign up for the NVIDIA developer program to download NCCL (the second line above).

- Then run: 
```
conda activate derm-ai
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
cd /home/ubuntu/src/derm-ai
cp ISIC2017_with_optimization/DermaAI_TwoParamHpBandSter.ipynb DermaAI_TwoParamHpBandSter.ipynb
cp ISIC2017_with_optimization/DermaAI.ipynb DermaAI.ipynb
cp "ISIC2017_with_optimization/Visualizing HpBandSter Results.ipynb" "Visualizing HpBandSter Results.ipynb"
```
- Next, run the following:
```
vim hosts
```
- Next, copy everything starting from ":8888" to a browser address bar on your local machine.
- In front of the ":8888" paste the public IPv4 address of the instance and open, and Jupyter will open and be ready to go!
- If you like, from that point, you can use the tmux command "CTRL + b, d" to detach from that terminal window and keep working (or close your ssh / PuTTY session without deactivating the Jupyter notebook session)
