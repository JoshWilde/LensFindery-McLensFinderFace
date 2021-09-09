import os
import numpy as np
import torch
from torch.optim import SGD
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import copy


import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import LogNorm
from astropy.io import fits
import matplotlib.pyplot as plt

from skimage.transform import resize
from scipy.ndimage import gaussian_filter

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]
    image = (pil_im-np.amin(pil_im))/(np.amax(pil_im)-np.amin(pil_im))
    # Resize image
    #if resize_im:
    #    pil_im.thumbnail((224, 224))
    #im_as_arr = np.float32(pil_im)
    #im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    #for channel, _ in enumerate(im_as_arr):
    #    im_as_arr[channel] /= 255
    #    im_as_arr[channel] -= mean[channel]
    #    im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(image).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224

    if len(im_as_ten.size()) < 4:
        im_as_ten.unsqueeze_(0)
    #im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    #if isinstance(im, (np.ndarray, np.generic)):
    #    im = format_np_output(im)
    #    im = Image.fromarray(im)
    #print(im)
    #print(type(im))
    #print(im.size)
    #im = Image.fromarray(im)
    #im.save(str(path))
    fig = plt.imshow(im[0]) #, norm=LogNorm())
    #fig = plt.imshow(im)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(str(path), bbox_inches='tight',dpi=300)
    plt.clf()
    #plt.savefig(str(path)+'.pdf', bbox_inches='tight',dpi=300)
    
def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    #reverse_mean = [-0.485, -0.456, -0.406]
    #reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    #for c in range(3):
    #    recreated_im[c] /= reverse_std[c]
    #    recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    #recreated_im = np.round(recreated_im * 255)
    #recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im
    


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class, lr, loc, input_img, range_Images):
        #self.mean = [-0.485, -0.456, -0.406]
        #self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.lr = lr
        self.range_Images = range_Images
        # Generate a random image
        #img_dir = '../../RandomImages/maxresdefault-59e8d857396e5a001012e50b.jpg'
        #R = plt.imread(img_dir)
        #R = resize(R[:,:,0], (200,200))
        #R = torch.from_numpy(R)
        #R = torch.unsqueeze(R,dim=0)
        self.created_image = input_img #R.numpy()#np.random.random((1,1,200,200))
        self.loc = loc
        #self.created_image = np.random.random((1,4,200,200)) #np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')
        
        im_path = str(self.loc)+'c_specific_iteration_0'#+'.jpg'
        self.created_image = self.created_image[0]
        save_image(self.created_image, im_path+'.jpg')
        np.save(im_path,self.created_image)

    def generate(self):
        #if self.target_class == 0:
        #    initial_learning_rate = 0.1#6
        #else:
        #    initial_learning_rate = 0.1
        for i in range(1, self.range_Images):
            # Process image and return variable
            #print(self.created_image)
            #print(self.created_image.size())
            self.processed_image = preprocess_image(self.created_image, False)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=self.lr)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            #print(output)
            #print(output[0][1])
            #class_loss = -output[0, self.target_class]
            class_loss = -output[0][self.target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()), 'Acc:', (output[1].data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                #print(i)
                # Save image
                #if self.target_class == 0:
                #    # loc = '../generated/SGL2-Data/Non_Lens_Gifs/non_lenses_1_jpg_No1/'
                #    loc = '../generated/SGL2-Data/Images/Non_lenses_0.1_lens2_No1/'
                #else:
                #    loc = '../generated/SGL2-Data/Images/Lenses_0.1_lens2_No1/'
                im_path = str(self.loc)+'c_specific_iteration_'+str(i)#+'.jpg'
                save_image(self.created_image, im_path+'.jpg')
                np.save(im_path,self.created_image)
        return self.processed_image