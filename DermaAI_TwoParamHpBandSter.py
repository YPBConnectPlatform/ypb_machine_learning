# This script uses the HpBandSter model (https://github.com/automl/HpBandSter) to perform hyperparameter optimization of the learning rate, # stochastic gradient descent (SGD) momentum, and dropout probability for a modified version of AlexNet trained on the ISIC 2017 skin      # cancer dataset. 

# The data is assumed to be in the following directories and structured in the following way:
# Train: data/train/
# Validation: data/valid/
# Test: data/test/

# Each of train, validation and test directories should have 3 folders: melanoma/, nevus/ and seborrheic_keratosis/

# The current version runs well:
# - On an Amazon AWS p2.xlarge instance
# - Using the following AMI: Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-1197bd74)
# - Using the conda environment listed in dermAIenv.yml

# General imports
import numpy as np
from glob import glob
import torch
from PIL import Image 
import matplotlib.pyplot as plt   
# Pytorch imports
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
# Imports used for HPBANDSTER.
import os
import time
import pickle
import argparse
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging

# Function definitions
        
# ISICNetAlex
# Defines the pretrained AlexNet-based convolutional neural network we will use to train on the ISIC 2017 dataset.
# The function in general can add new fully connected layers, specify the number of layer elements in each of those layers, specify the dropout probability in the dropout layers, and denote whether or not ot freeze weights in the 'original' fully connected layers in AlexNet. The default behavior is to add no new fully connected layers and freeze none of the old weights in the fully connected layers. 

def ISICNetAlex(num_new_fc_layers=0, new_layer_elements=[], dropout_rate=0.5, old_fclayers_tofreeze=[0,1]):
    ''' num_new_fc_layers -- There are 3 FC layers in Alexnet to begin with. The last layer gets replaced by default. 
                          A value of 1 here would add a fourth layer, whereas a value of 0 would just retool the last layer.
     dropout_rate --      0<= dropout_rate < 1. Sets the dropout rate for each FC layer to this .value.
     old_fclayers_to_freeze --  A set of indices with length <= 2. Must be a subset of [0,1]
    '''
    
    # Grab the pretrained PyTorch model. This model is pretrained on ImageNet, and expects 224 x 224 pixel images as input.
    model = models.alexnet(pretrained=True)
    
    # Define the number of output nodes. Here, on the ISIC 2017 database, there are 3 possible output classes: melanoma, nevus and seborrheic keratosis. 
    
    num_output_nodes = 3
    
    # IMPORTANT -- these 3 lines override the values passed in to the function. I've done this because I wanted to simplify the hyperparameter search by not messing around with the overall architecture of the network. But it can pretty easily be done. 
    num_new_fc_layers = 0
    new_layer_elements = []
    old_fclayers_tofreeze = []
    
    # Input error checking
    
    # Can't have negative number of new layers.
    assert num_new_fc_layers >= 0
    # Dropout rate must be between 0 and 1
    assert 0 <= dropout_rate < 1
    # You have to specify the number of output elements for each additional layer you want.
    assert num_new_fc_layers == len(new_layer_elements)
    # Old layers to freeze must be picked from the following: [0,1]. All new layers and the last layer will be trained by necessity.
    checklayers = set(old_fclayers_tofreeze)
    assert checklayers.issubset([0,1])
    
    # Add any new layers. This will have the same structure as the existing fc layers.
    for i in range(num_new_fc_layers+1):
        # No dropout before the last layer.
        if i == num_new_fc_layers:
            model.classifier.add_module(str(6+i*3), nn.ReLU(inplace=True))
            model.classifier.add_module(str(6+i*3), nn.Linear(model.classifier[4+i*3].out_features,num_output_nodes,bias=True))
        else:
            model.classifier.add_module(str(6+i*3), nn.Dropout(dropout_rate))
            model.classifier.add_module(str(7+i*3), nn.Linear(model.classifier[7+(i-1)*3].out_features,new_layer_elements[i],bias=True))
            model.classifier.add_module(str(8+i*3), nn.ReLU(inplace=True))
            
             
    # Set requires_grad appropriately for the model.
    # First, set all requires_grad for all parameters in the 'features' (convolutional) portion of the model to False.
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Now deal with the 'classifier' (the fully connected layers).
    for i in range(len(old_fclayers_tofreeze)):
        if old_fclayers_tofreeze[i] == 0:
            for param in model.classifier[0].parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = False
            for param in model.classifier[2].parameters():
                param.requires_grad = False
        if old_fclayers_tofreeze[i] == 1:
            for param in model.classifier[3].parameters():
                param.requires_grad = False
            for param in model.classifier[4].parameters():
                param.requires_grad = False
            for param in model.classifier[5].parameters():
                param.requires_grad = False
    
    #set dropout rate for all relevant layers.
    for i in range(len(model.classifier)):
        if type(model.classifier[i]) is torch.nn.modules.dropout.Dropout:
            model.classifier[i] = nn.Dropout(dropout_rate)
        
    
    
    # Print the model
    print(model)
        
    return model
    
    

# number_of_parameters
# Tells the number of weights / parameters to be varied by the model -- only looks at the ones for which requires_grad is True. 
def number_of_parameters(model):
        return(sum(p.numel() for p in model.parameters() if p.requires_grad))


#HpBandSter class definitions.    

# Define the worker
class worker(Worker):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
            ''' Define transforms, datasets and data loaders
             Note that AlexNet expects normalized images (as below). 
             Validation / test transforms do not have random transformations applied. 
             '''
            
            random_transforms = [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomAffine(degrees=45,translate=(0.1,0.3),scale=(0.5,2))]

            train_transforms = transforms.Compose([transforms.Resize(size=256),transforms.CenterCrop(224),transforms.RandomChoice(random_transforms), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

            valid_test_transforms = transforms.Compose([transforms.Resize(size=256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
            
            # No data augmentation for the validation and test datasets, because we're using those as is for evaluation.
            train_data = datasets.ImageFolder('data/train',transform=train_transforms)

            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, drop_last=True, shuffle=True)
            
            validation_data = datasets.ImageFolder('data/valid',transform=valid_test_transforms)
            test_data = datasets.ImageFolder('data/test',transform=valid_test_transforms)
            

            self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 128)
            
            self.test_loader = torch.utils.data.DataLoader(test_data, batch_size = 128)
            
            
    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        # Useful website -- https://aws.amazon.com/blogs/machine-learning/scalable-multi-node-deep-learning-training-using-gpus-in-the-aws-cloud/

        ''' The below is commented out because I don't want to mess with the CNN's architecture. If you want to use hyperparameter optimization to alter the architecture of the fully connected layers as well, you can use the below. '''
        
        #new_layer_elements = np.array([config['num_els_new_1'] if config['num_new_fc_layers'] >= 1 else None, 
        #                      config['num_els_new_2'] if config['num_new_fc_layers'] >= 2 else None, 
        #                      config['num_els_new_3'] if config['num_new_fc_layers'] >= 3 else None])
        
        #new_layer_elements = list(new_layer_elements[new_layer_elements != None])
        
        #old_fclayers_tofreeze = np.array([0 if config['freeze0_cat'] == 1 else None,
        #                        1 if config['freeze1_cat'] == 1 else None])
        
        #old_fclayers_tofreeze = list(old_fclayers_tofreeze[old_fclayers_tofreeze != None])
        
        # Generate the model
        model = ISICNetAlex(num_new_fc_layers=0,
                                                new_layer_elements=[],
                                                dropout_rate=config['dropout_rate'],
                                                old_fclayers_tofreeze=[],
        )

        # Use GPU processing if available. 
        if torch.cuda.is_available():
            model.cuda()
            
        # Build criterion and optimizer.
        criterion = torch.nn.CrossEntropyLoss()
        
        ''' The below is commented out because I don't want to mess with the optimizer. '''
        #if config['optimizer'] == 'Adam':
        #    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        #else:
        #    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])
        
        
        # Run training loop.
        # IMPORTANT -- note that the budget parameter used in setting up HpBandSter refers to the number of epochs. It can be made to refer to other parameters, but here we chose to have it refer to epochs. 
        for epoch in range(int(budget)):
            start = time.time()
            # initialize variables to monitor training and validation loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # move to GPU if available
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += 1/(batch_idx+1)*(loss.data-train_loss)

            print("Epoch {} training time took {} seconds".format(epoch,time.time()-start))

        train_accuracy = self.evaluate_accuracy(model, self.train_loader)
        validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
        test_accuracy = self.evaluate_accuracy(model, self.test_loader)

        return ({
                'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
                'info': {       'test accuracy': test_accuracy,
                                        'train accuracy': train_accuracy,
                                        'validation accuracy': validation_accuracy,
                                        'number of parameters': number_of_parameters(model),
                                }

        })

    def evaluate_accuracy(self, model, data_loader):
        correct = 0
        total = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(data_loader):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model.forward(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
        #import pdb; pdb.set_trace()
        return(correct/total)


    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        

        # Learning rate hyperparameter
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        

        # Stochastic gradient descent momentum as parameter.
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

        cs.add_hyperparameters([lr, sgd_momentum])
        
        # Optimizer hyperparameters.
        #optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        #cs.add_hyperparameters([optimizer])
        
        # Only add the sgd_momentum hyperparameter if the optimizer is stochastic gradient descent. Otherwise, it doesn't make sense.
        #cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        #cs.add_condition(cond)

        ''' The below is commented out because we're not fiddling with architecture in this optimization.'''
        #num_new_fc_layers =  CSH.UniformIntegerHyperparameter('num_new_fc_layers', lower=0, upper=3, default_value=0, log=False)
        #num_els_new_1 = CSH.UniformIntegerHyperparameter('num_els_new_1', lower=128, upper=4096, default_value = 1000, log=True)
        #num_els_new_2 = CSH.UniformIntegerHyperparameter('num_els_new_2', lower=128, upper=4096, default_value = 1000, log=True)
        #num_els_new_3 = CSH.UniformIntegerHyperparameter('num_els_new_3', lower=128, upper=4096, default_value = 1000, log=True)

        #freeze0_old = CSH.UniformIntegerHyperparameter('freeze0_cat', lower = 0, upper = 1, default_value = 1, log=False)
        #freeze1_old = CSH.UniformIntegerHyperparameter('freeze1_cat', lower=0, upper=1, default_value=1, log=False)

        #cs.add_hyperparameters([num_new_fc_layers, num_els_new_1, num_els_new_2, num_els_new_3, freeze0_old, freeze1_old, batchsize])

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)

        cs.add_hyperparameters([dropout_rate])

        return cs


# Main function
    
def main():
    # Check quantities of train, validation and test images
    train_images = np.array(glob("data/train/*/*"))
    valid_images = np.array(glob("data/valid/*/*"))
    test_images = np.array(glob("data/test/*/*"))

    # Check relative percentages of image types
    train_images_mel = np.array(glob("data/train/melanoma/*"))
    train_images_nev = np.array(glob("data/train/nevus/*"))
    train_images_seb = np.array(glob("data/train/seborrheic_keratosis/*"))

    valid_images_mel = np.array(glob("data/valid/melanoma/*"))
    valid_images_nev = np.array(glob("data/valid/nevus/*"))
    valid_images_seb = np.array(glob("data/valid/seborrheic_keratosis/*"))

    test_images_mel = np.array(glob("data/test/melanoma/*"))
    test_images_nev = np.array(glob("data/test/nevus/*"))
    test_images_seb = np.array(glob("data/test/seborrheic_keratosis/*"))

    print("There are {} training images, {} validation images and {} test images.".format(len(train_images),len(valid_images),len(test_images)))
    print("For the training images, {mel:=.1f}% ({mel2}) are of melanoma, {nev:=.1f}% ({nev2}) are of nevus and {seb:=.1f}% ({seb2}) are for seborrheic keratosis.".format(mel=len(train_images_mel)/len(train_images)*100, mel2=len(train_images_mel),nev=len(train_images_nev)/len(train_images)*100, nev2=len(train_images_nev), seb=len(train_images_seb)/len(train_images)*100, seb2=len(train_images_seb)))
    print("For the validation images, {mel:=.1f}% ({mel2}) are of melanoma, {nev:=.1f}% ({nev2}) are of nevus and {seb:=.1f}% ({seb2}) are for seborrheic keratosis.".format(mel=len(valid_images_mel)/len(valid_images)*100, mel2=len(valid_images_mel),nev=len(valid_images_nev)/len(valid_images)*100, nev2=len(valid_images_nev), seb=len(valid_images_seb)/len(valid_images)*100, seb2=len(valid_images_seb)))
    print("For the test images, {mel:=.1f}% ({mel2}) are of melanoma, {nev:=.1f}% ({nev2}) are of nevus and {seb:=.1f}% ({seb2}) are for seborrheic keratosis.".format(mel=len(test_images_mel)/len(test_images)*100, mel2=len(test_images_mel),nev=len(test_images_nev)/len(test_images)*100, nev2=len(test_images_nev), seb=len(test_images_seb)/len(test_images)*100, seb2=len(test_images_seb)))

    # Set HpBandSter logging
    logging.basicConfig(level=logging.DEBUG)

    # Define the parser. Note that key parametres are the min_budget, max_budget, shared_directory and n_iterations. 
    parser = argparse.ArgumentParser(description='ISIC2017 - CNN on Derm Dataset')
    parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
    parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=3)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
    parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='/home/ubuntu/src/derm-ai/data')
    parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='pytorch')
    args = parser.parse_args([])


    host = hpns.nic_name_to_host(args.nic_name)
    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.
    result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)
    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
    ns_host, ns_port = NS.start()


    # Start local worker
    w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
    w.run(background=True)


    bohb = BOHB(  configspace = w.get_configspace(),
                          run_id = args.run_id,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          result_logger=result_logger,
                          min_budget=args.min_budget, max_budget=args.max_budget,
                   )

    # Run an optimizer

    res = bohb.run(n_iterations=args.n_iterations)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

if __name__ == '__main__':
    main()