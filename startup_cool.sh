#!/bin/bash
mkdir -p /home/ubuntu/src/derm-ai
cd /home/ubuntu/src/derm-ai
git clone https://github.com/paulbisso/ISIC2017_with_optimization
conda env create -f /home/ubuntu/src/derm-ai/ISIC2017_with_optimization/dermAIenv.yml
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc