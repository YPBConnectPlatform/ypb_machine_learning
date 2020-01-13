import matplotlib.pyplot as _plt
from PIL import Image as _Image
import numpy as _np
import matplotlib.gridspec as _gridspec
import os as _os
import sys as _sys
from random import shuffle as _shuffle

def plot_some_imgs(num_imgs_to_plot,dataset,dataset_name):  
    print("Plotting {} images from dataset {}".format(num_imgs_to_plot,dataset_name))
    fig = _plt.figure(figsize = (10,num_imgs_to_plot*2))
    axes = []
    for i in range(num_imgs_to_plot):
        img = _Image.open(dataset[_np.random.randint(0,len(dataset)-1)])
        axes.append(fig.add_subplot(num_imgs_to_plot/2,2,i+1))
        axes[-1].imshow(img)
        
def view_DALIiter_images(batch_data, batch_size, normalized):
    # image_batch - the direct image output of running a DALIGenericIterator
    # batch_size - the number of images in each batch per GPU
    # normalized - Boolean flag denoting whether the image has undergone normalization (of the kind used for all pretrained Pytorch models)
    num_gpus = len(batch_data)
    columns = 4
    rows = (batch_size + 1) // (columns)

    gs = _gridspec.GridSpec(rows, columns)
    # Account for the fact that the image coming in may be sized as HWC vs CHW.
    if batch_data[0]['data'][0].size()[0] == 3:
        CHW_flag = 1
    else:
        CHW_flag = 0
    for i in range(num_gpus):
        titlestr = "GPU # {}".format(i)
        fig = _plt.figure(figsize = (32,(32 // columns) * rows))
        _plt.title(titlestr)
        for j in range(batch_size):
            _plt.subplot(gs[j])
            _plt.axis("off")
            # If the image is coming in as CHW, it has to be reshaped to HWC
            thisimg = batch_data[i]['data'][j].cpu().numpy()
            if CHW_flag:
                thisimg = _np.moveaxis(thisimg,0,-1)
            # If the image has been normalized (assuming Alexnet-type normalization on uint8s ranging from 0 to 255 originally),
            # then de-normalize it.
            if normalized:
                thisimg[:,:,0] = thisimg[:,:,0]*58.395+123.675
                thisimg[:,:,1] = thisimg[:,:,1]*57.12+116.28
                thisimg[:,:,2] = thisimg[:,:,2]*57.375+103.53
                thisimg = _np.uint8(thisimg)
            # Plot.
            _plt.imshow(thisimg)
            
def view_PTIter_images(batch_data,batch_size,normalized):
    columns = 4
    rows = (batch_size + 1) // (columns)

    gs = _gridspec.GridSpec(rows, columns)
    # Account for the fact that the image coming in may be sized as HWC vs CHW.
    CHW_flag = 1
    fig = _plt.figure(figsize = (32,(32 // columns) * rows))
    for j in range(batch_size):
        _plt.subplot(gs[j])
        _plt.axis("off")
        # If the image is coming in as CHW, it has to be reshaped to HWC
        thisimg = batch_data[j].cpu().numpy()
        if CHW_flag:
            thisimg = _np.moveaxis(thisimg,0,-1)
        # If the image has been normalized (assuming Alexnet-type normalization on uint8s ranging from 0 to 255 originally),
        # then de-normalize it.
        if normalized:
            thisimg[:,:,0] = thisimg[:,:,0]*58.395+123.675
            thisimg[:,:,1] = thisimg[:,:,1]*57.12+116.28
            thisimg[:,:,2] = thisimg[:,:,2]*57.375+103.53
            thisimg = _np.uint8(thisimg)
        # Plot.
        _plt.imshow(thisimg)
            
# Define the NVIDIA DALI iterator that will feed into the data loading pipeline. 
# The 'image_dir' passed into __init__ is assumed to be a top-level image directory. In other words, there should be no image
# files immediately within that directory. The algorithm below will ignore all image files that are directly within the
# top-level image directory. Rather, the top-level directory should contain exclusively other directories. 

# Each of those sub-directories should contain exclusively .JPEG image files.
# Files within each sub-directory are assumed to belong to the same image class. 
class ExternalInputIterator(object):
    def __init__(self, batch_size, image_dir, shuffle_flag):
        self.images_dir = image_dir
        self.files = []
        self.shuffle_flag = shuffle_flag
        self.num_files = 0
        assert isinstance(shuffle_flag, bool)
        # Check which of the entries in the top directory are themselves directories. 
        dirfiles = _os.listdir(image_dir)
        num_imgdirs = 0
        self.img_label_dict = dict() # storage denoting which label number corresponds with which label name.

        # NOTE: the below will only grab JPG/JPEG files, and would need to be modified to grab other types of image files.
        # NOTE: even if the below were modified to grab other types of image files, the DALI image processing pipeline
        # uses a JPEG decoder to load in the files, so just keep that in mind.
        for file in dirfiles:
            # Grab the name of every entry in the top directory.
            fullname = _os.path.join(image_dir, file)
            # If an entry is a directory, grab the filenames for all the JPG/JPEG files in that directory.
            if _os.path.isdir(fullname):
                imgfiles = _os.listdir(fullname)
                for imgfile in imgfiles:
                    thisimgpath = _os.path.join(fullname,imgfile)
                    if _os.path.isfile(thisimgpath) and (thisimgpath.lower().endswith(".jpg") or thisimgpath.lower().endswith(".jpeg")):
                        self.img_label_dict[thisimgpath] = num_imgdirs
                        self.files.append(thisimgpath)
                num_imgdirs += 1
        assert num_imgdirs > 0
        # Store the batch size.
        self.batch_size = batch_size
        # Shuffle the input filenames. 
        if self.shuffle_flag:
            _shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i], self.img_label_dict[self.files[self.i]]
            self.num_files += 1
            f = open(jpeg_filename, 'rb')
            # Later steps in the DALI image processing pipeline (the resize operation, in particular) expect uint8s. 
            batch.append(_np.frombuffer(f.read(), dtype = _np.uint8))
            labels.append(_np.array([label], dtype = _np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__

