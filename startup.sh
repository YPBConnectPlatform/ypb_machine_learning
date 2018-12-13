#!/bin/bash -i
cd src
mkdir derm-ai
cd derm-ai
git clone https://github.com/udacity/dermatologist-ai.git
mkdir data
cd data
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
unzip train.zip
rm train.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
unzip valid.zip
rm valid.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip
unzip test.zip
rm test.zip
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh -b
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate" >> ~/.bashrc
. ~/.bashrc
source deactivate
export PYTHONPATH=/home/ubuntu/anaconda3
cd /home/ubuntu/src/derm-ai
git clone https://github.com/paulbisso/ISIC2017_with_optimization
conda install -y python=3.5
conda env create -f /home/ubuntu/src/derm-ai/ISIC2017_with_optimization/dermAIenv.yml
python -m ipykernel install --user --name=derm-ai
