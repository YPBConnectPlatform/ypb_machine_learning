#!/usr/bin/env bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Specify hosts in the file `hosts`, ensure that the number of slots is equal to the number of GPUs on that host

# Use train_more_aug.sh when training with large number of GPUs (128, 256, etc). That script uses more augmentations and layer wise adaptive rate control (LARC) to help with convergence at large batch sizes. 

# This script has been tested on DLAMI v17 and above

# In Linux, the [ -z "expr" ] statement returns true if "expr" is not an empty string. Also, it's worthwhile to note that in Linux shell scripts
# square brackets is equivalent to the test command. So the below line is equivalent to "test -z "$1"". It just checks to see 
# whether or not there are any arguments given to the script.
# So if the script ran as ./train.sh we would echo usage information and exit. If we ran it as ./train.sh 8 it would set the
# gpus variable to be 8.
if [ -z "$1" ]
  then
    echo "Usage: "$0" <num_gpus>"
    exit 1
  else
    gpus=$1
fi
# Line-by-line of this function.
# "while read -u 10 host" -- read -u 10 will read one line from file descriptor 10. Linux has 3 standard file descriptors (0, 1, 2)
# and anything above that (up to 1024 I think) can be assigned arbitrarily. At the very last line of the function, the "10<$1" command
# tells the function to assign file descriptor 10 to the first argument given to the function (in this case, it's intended to be our
# file named 'hosts'). 
#
# "do host = ${host%% slots*}" -- since our 'hosts' file has the syntax "<IP> slots=<numSlots>", hosts=${host%% slots*} will assign the value in
# <IP> to the variable 'host'.
#
# 'ssh -o "StrictHostKeyChecking no" $host ""$2"" just uses ssh to connect to the IP in each line of the 'hosts' file and then run the second
# input to the function as a shell command.
function runclust(){ while read -u 10 host; do host=${host%% slots*}; if [ ""$3"" == "verbose" ]; then echo "On $host"; fi; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };

# Activating derm-ai on each machine.
# 'runclust hosts "<commands>" will run <commands> on each machine.
# tmux new-session -s activation_pytorch will start a new tmux session named "activation_pytorch" on each machine.
# The -d \"conda activate derm-ai > activation_log.txt;\" part of the command will tell it not to attach to the current terminal session,
# and to run the command within escaped quotes, writing the output to a file called activation_log.txt in the current working directory.
runclust hosts "echo 'Activating derm-ai'; tmux new-session -s activation_pytorch -d \"conda activate derm-ai > activation_log.txt;\"" verbose; 
# Waiting for activation to finish.
# Any errors in stderr get written to file in the /dev/null directory. At the end any output (errors) from trying to activate your 
# environment, stored in activation_log.txt, get printed out to the screen.
# So basically these two runclust commands just check to make sure you can ssh into every node from the master and activate the 
# appropriate conda environment.
runclust hosts "while tmux has-session -t activation_pytorch 2>/dev/null; do :; done; cat activation_log.txt"
# You can comment out the above two runclust commands if you have activated the environment on all machines at least once

# Activate locally for the mpirun command to use
source activate derm-ai

echo "Launching training job using $gpus GPUs"
# Set -ex means that the script will print all commands and their arguments to stdout (-x) and the script will
# exit immediately if any command yields a non-zero exit status (-e).
set -ex

# use ens3 interface for DLAMI Ubuntu and eth0 interface for DLAMI AmazonLinux
if [  -n "$(uname -a | grep Ubuntu)" ]; then INTERFACE=ens3 ; else INTERFACE=eth0; fi
# nvidia-smi -L prints out info for each GPU, one GPU to a line. wc -l counts the number of lines in a command's output. 
NUM_GPUS_MASTER=`nvidia-smi -L | wc -l`

# p3 instances have larger GPU memory, so a higher batch size can be used.
# nvidia-smi --query-gpu=memory.total --format=csv,noheader will print out the memory capacity on each GPU, one per line. The
# -i 0 flag tells it just to take the first GPU as indicative of all the rest. the | awk '{print $1}' separates the number of 
# MB of memory from the "MB" portion of the string. 
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`

############## NOTE #########################
# If the GPU memory is greater than 15000, then set the batch size to 256, otherwise set it to 128. Will need to fiddle with this 
# depending on the details of my image set.
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi
############# /NOTE #########################

# Training
# Run mpirun using one process per GPU and the hostfile specified in ~/src/derm-ai/hosts.
# No idea what MCA parameters, bind-to socket or most of the other parameters do. Fiddle with them if things aren't working well.

# Then run the python script for training, ignoring warnings. 
~/anaconda3/envs/derm-ai/bin/mpirun -np $gpus -hostfile ~/src/derm-ai/hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore ~/src/derm-ai/DermAI_train_horovod.py
