#!/bin/bash
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
cd /home/ubuntu/src/derm-ai
git clone https://github.com/paulbisso/ISIC2017_with_optimization
conda env create -f /home/ubuntu/src/derm-ai/ISIC2017_with_optimization/dermAIenv.yml
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
