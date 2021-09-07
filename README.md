# Detecting gravitational lenses using machine learning: exploring interpretability and sensitivity to rare lensing configurations
This repository contains the CNN architectures and weights, and the code used to train and test these CNN models [1]. The code used to generate gravitational compound lenses are also included [1]. The interpretability code in this paper is adapted from [2]. 


## CNN Models
This paper uses 7 CNN models, these models use two basic architectures (OU-66 and OU-200) with varing number of input bands. The models OU-J, OU-Y, OU-H, OU-JYH, and OU-66 use the OU-66 architecture, and OU-VIS and OU-200 use the OU-200 architecture. 

In the figures below, yellow squares represent 2D Convolutional layers, teal squares represent ReLU layers, orange squares represent 2D MaxPool layers, and red squares represent Dropout layers.

![OU-66](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/CNN%20Models/OU-66_3-1.png)


![OU-200](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/CNN%20Models/OU-200-4BANDS-CLEAR_3-1.png)



## Gravitational Compound Lenses
To investigate if these models can identify gravitational compound lenses, we needed to simulate these images. This was achieved using the Lenstronomy [3] and SkyPy [4] Python packages. In the paper [1], We generated two different compound lens datasets, one was mainly compound arcs and the other was mainly double Einstein rings.

## Interpretability
Here we describe the interpretability methods we have used in this paper to understand how our CNNs may be identifying gravitational lenses and non-lenses. We present gifs which illustrate these methods at the expense of a smaller colour pallete. 

### Occlusion Maps
In our paper we use create an occlusion square that contains only zero values (1x1 pixels for OU-66 and 4x4 pixels for OU-200). The image is passed through the CNN for each position of the occlusion square which can be seen in middle of figure 1C. The output of these positions are recorded. The change in output is shown in the right section of figure 1C. Blue pixels indicate a feature associated with gravitational lensing and red pixels indicate a feature associated with non-lensing.

![OccMap](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/Interpretability/OccMap_GitHub.gif)


### Class Generated Images
We generate images which highly activated both classes (lens and non-lens) to understand what features in the image the CNN identifies with each class. This is done by freezing the model weights of an already trained CNN and all output classes are set to 0 apart from the target class which is set to 1 [5,6]. The input image contains random uniform noise and the same random image is used for each target class. Using back-propagation the input image is updated rather than the model weights. This results in an updated image that activates the target class more strongly than the unmodified input image. 

This process is shown in figure 2C, which shows the generation of class generated images for both classes in OU-66. This shows the generation of new features which cause a strong response in each of the target classes.


ClassAct66 gif

This process is shown in figure 3C, which shows the generation of class generated images for both classes in OU-66.

### Deep Dream
The deep dream process in the same as the class generated image except that the original input image is an image from the data set instead of random noise. Examples of this process are shown in figures 4C and 5C for OU-66 and OU-200. The input image to the deep dream process is classified as non-lens in the dataset.

![DD-0-OU66](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/Interpretability/DeepDreamImagesGif_252473_OU66.gif)

![DD-0-OU200](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/Interpretability/DeepDreamImagesGif_252473_OU200.gif)

Examples of this process are shown in figures 4C and 5C for OU-66 and OU-200. The input image to the deep dream process is classified as a gravitational lens in the dataset.

![DD-1-OU66](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/Interpretability/DeepDreamImagesGif_250952_OU66.gif)

![DD-1-OU200](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/Interpretability/DeepDreamImagesGif_250952_OU200.gif)

## To Do List
### CNN Models
model code

load images code

training code

testing code

Add Weights

### Gravitational Compound Lenses
Generating Simulated Compound Lens Code

testing code

### Interpretability
Add Occlusion Map Code

Images that activate kernels code

Class generated images code

class generated images gif

Deep Dream Code

generating images to highly activate kernels code

Guided Grad-CAM code



## Citation


## References:
[1] Detecting gravitational lenses using machine learning: exploring interpretability and sensitivity to rare lensing configurations

[2] Ozbulak U., 2019, PyTorch CNN Visualizations, https://github.com/utkuozbulak/pytorch-cnn-visualizations 

[3] Birrer S., Amara A., 2018, Physics of the Dark Universe, 22, 189, https://github.com/sibirrer/lenstronomy

[4] SkyPy Collaboration et al., 2021, SkyPy, https://github.com/skypyproject/skypy

[5] Yosinski J., Clune J., Nguyen A., Fuchs T., Lipson H., 2015, arXiv preprintarXiv:1506.06579

[6] Simonyan  K.,  Vedaldi  A.,  Zisserman  A.,  2013,  arXiv  preprintarXiv:1312.6034
