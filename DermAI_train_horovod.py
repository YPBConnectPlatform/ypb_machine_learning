from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models, transforms, datasets
import torch.nn as nn
import horovod.torch as hvd
import timeit
import numpy as np
import time
from random import shuffle
import os, sys
import csv
from tqdm import tqdm
import math

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch DermAI Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--train-dir', default='data/train/',
                    help='location of training images')

parser.add_argument('--valid-dir', default='data/valid/',
                    help='location of validation images')

parser.add_argument('--test-dir', default='data/test/',
                    help='location of test images')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')

parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.000125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce
torch.manual_seed(args.seed)


hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break
        
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0


# Data loading function -- for the moment, uses native PyTorch data-loader alone. 
def setup_data_loader(train_batch, valid_batch, test_batch):
    random_transforms = [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomAffine(degrees=45,translate=(0.1,0.3),scale=(0.5,2))]

    train_transforms = transforms.Compose([transforms.Resize(size=224),transforms.CenterCrop(224),transforms.RandomChoice(random_transforms), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    # No data augmentation for the validation and test datasets, because we're using those as is for evaluation.
    train_data = datasets.ImageFolder(args.train_dir,transform=train_transforms)
    validation_data = datasets.ImageFolder(args.valid_dir,transform=valid_test_transforms)
    test_data = datasets.ImageFolder(args.test_dir,transform=valid_test_transforms)
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=hvd.size(), rank=hvd.rank())
    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_batch, sampler=train_sampler, drop_last=True, num_workers=4, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size = valid_batch, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, pin_memory=True, batch_size = test_batch, sampler=test_sampler)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, {'train': train_sampler, 'valid': val_sampler, 'test': test_loader}


# Set up data loader and samplers. 
loader_dict, sampler_dict = setup_data_loader(args.batch_size, args.batch_size, args.batch_size)

# Set up standard model.
model = models.alexnet(pretrained=True)

# Train last layer of classifier.
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(in_features=4096, out_features=3, bias=True)

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce * hvd.size()),
                      momentum=args.momentum, weight_decay=args.wd)
    
# When using DALI, it is essential to set the reduction='sum' flag when using CrossEntropyLoss.
# Because of the way that DALI queues up images into the processing pipeline, failing to use reduction='sum' will yield different
# losses on the same dataset using the same model, even when not shuffling the images in the dataset! This is due to an adverse
# interaction between the way the DALI iterator works and how PyTorch calculates Cross Entropy Loss when _reduction_ is not set to 'sum'
#
# It is not necessary to set reduction='sum' when using the PyTorch iterators, but it won't hurt either.
criterion = nn.CrossEntropyLoss(reduction='sum')


# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce)


# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

def train(epoch,loaders,samplers):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    samplers['train'].set_epoch(epoch)
    
    train_loss = Metric('train_loss')
    
    num_imgs_sofar = 0
    with tqdm(total=len(loaders['train']),
          desc='Train Epoch     #{}'.format(epoch),
          disable=not verbose) as t:

        for batch_idx, (data, target) in enumerate(loaders['train']):
            adjust_learning_rate(epoch, batch_idx, loaders)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                loss = criterion(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            
            optimizer.step()
            num_imgs_sofar += len(target)
            t.set_postfix({'loss': train_loss.sum.item()/num_imgs_sofar})
            t.update(1)
           
            
def validate(loaders):
    model.eval()
    val_loss = torch.tensor(0.)
    
    correct_imgs = 0.
    num_imgs = 0

    with tqdm(total=len(loaders['valid']),
              desc='Validation',
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in loaders['valid']:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss += torch.tensor(criterion(output, target).item())
                pred = output.data.max(1, keepdim=True)[1]
                correct_imgs += pred.eq(target.data.view_as(pred)).cpu().float().sum()
                num_imgs += len(target)
                t.update(1)
        correct_imgs = hvd.allreduce(correct_imgs,name='val_accuracy',average=False).item()
        val_loss = hvd.allreduce(val_loss,name='val_loss',average=False).item()
        num_imgs = hvd.allreduce(torch.tensor(num_imgs),name='num_imgs_seen',average=False).item()
        
    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            val_loss/num_imgs, correct_imgs/num_imgs*100))        

def test(loaders):
    model.eval()
    test_loss = torch.tensor(0.)
    correct_imgs = 0.
    num_imgs = 0
    for data, target in loaders['test']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += torch.tensor(criterion(output, target).item())
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_imgs += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        num_imgs += len(target)
   
    correct_imgs = hvd.allreduce(correct_imgs,name='test_accuracy',average=False).item()
    test_loss = hvd.allreduce(test_loss,name='test_loss',average=False).item()
    num_imgs = hvd.allreduce(torch.tensor(num_imgs),name='num_imgs_seen',average=False).item()

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss/num_imgs, correct_imgs/num_imgs*100))          
    
# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, loaders):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(loaders['train'])
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj
        
def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train(epoch,loader_dict, sampler_dict)
    total_time = time.time() - start
    log("Epoch {} training time took {} seconds".format(epoch,total_time))
    log("Average speed of {} images / second".format(2000/(total_time)))
    validate(loader_dict)
    #test(loader_dict)
    save_checkpoint(epoch)
    
