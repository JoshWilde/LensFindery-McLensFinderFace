import os
import numpy as np
import torch
from torch.optim import SGD
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import copy

from torch.optim import Adam
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
    
    
class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,lr, steps,path,input_Size):
        self.model = model.layer
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.lr = lr
        self.steps = steps
        self.path = path
        self.input_Size = input_Size
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = (np.random.uniform(0, 1, (self.input_Size)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        im_path = self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter0' + '.jpg'
        save_image(processed_image.detach().numpy()[0], im_path)
        np.save(self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter0', processed_image.detach().numpy())
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=self.lr, weight_decay=1e-6)
        for i in range(1, self.steps+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            
            # Save image
            if i % 5 == 0:
                im_path = self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                np.save(self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i), self.created_image)
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = (np.random.uniform(0, 1, (self.input_Size)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        im_path = self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter0' + '.jpg'
        save_image(processed_image.detach().numpy(), im_path)
        np.save(self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter0', processed_image.detach().numpy())
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=self.lr, weight_decay=1e-6)
        for i in range(1, self.steps+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            
            # Save image
            if i % 5 == 0:
                im_path = self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                np.save(self.path+'/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i), self.created_image)
                save_image(self.created_image, im_path)