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
 - You need to create a security group that will be replicated across all of the nodes -- key points are that SSH needs to be accessible from any IP and that all TCP ports should be open to the entire security group.
- Make sure that the instance you start has at least 200 GB of space and 10,000 IOPS -- you should use the Provisioned IOPS SSD (io1) choice for drive. 
- Make sure that you have an IAM role for EC2 with policy AmazonS3FullAccess, and that it is attached to the running EC2 instance. 
 - Run the following code to get everything started up.
 ```
cd /home/ubuntu
git clone -b multi-node-horovod https://github.com/paulbisso/ISIC2017_with_optimization
mv ISIC2017_with_optimization/startup_cold.sh startup_cold.sh
mv ISIC2017_with_optimization/startup_cool.sh startup_cool.sh
chmod +x startup_cold.sh
chmod +x startup_cool.sh
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
cp ISIC2017_with_optimization/DermAI_train_horovod.py DermAI_train_horovod.py
```
- Next, run the following:
```
vim hosts
```
- Add the following line to hosts -- ```localhost ports=<# GPUs>``` where # GPUs is the number of GPUs on the lead AMI -- and then save.
- For each instance you want to use, add a line to hosts that looks like ```<private IP> ports=<# GPUs>```

- Run the following on your local machineto copy over the private key file you use to SSH into the lead instance. 
```scp -i /path/to/yourkey.pem  ubuntu@lead.instance.IP.address:~/.ssh```

- Then, on the lead instance, run the following to put the private key in the appropriate format and copy it to all other instances.
```
mv ~/.ssh/yourkey.pem ~/.ssh/id_rsa
chmod 400 ~/.ssh/id_rsa
function runclust(){ while read -u 10 host; do host=${host%% slots*}; scp -i ~/.ssh/id_rsa ~/.ssh/id_rsa ubuntu@$host:~/.ssh && ssh $host chmod 400 ~/.ssh/id_rsa; done 10<$1; };
```

- Then, run the following: 
```
function runclust(){ while read -u 10 host; do host=${host%% slots*}; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };
runclust hosts "echo \"StrictHostKeyChecking no\" >> ~/.ssh/config"
```
You will see an "scp: permission denied" error wherever the key is already present (like localhost, in this case).

- On the lead node, run the following to run the training script:
```
mpirun -np <total # GPUs> -hostfile ~/src/derm-ai/hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore ~/src/derm-ai/DermAI_train_horovod.py
