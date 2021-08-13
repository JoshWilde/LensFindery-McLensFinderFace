# Detecting gravitational lenses using machine learning: exploring interpretability and sensitivity to rare lensing configurations
This repository contains the CNN architectures and weights, and the code used to train and test these CNN models [1]. The code used to generate gravitational compound lenses are also included [1]. The interpretability code in this paper is adapted from [2]. 


## CNN Models
This paper uses 7 CNN models, these models use two basic architectures (OU-66 and OU-200) with varing number of input bands. The models OU-J, OU-Y, OU-H, OU-JYH, and OU-66 use the OU-66 architecture, and OU-VIS and OU-200 use the OU-200 architecture. 

![OU-66](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/OU-66_3-1.png)


![OU-200](https://github.com/JoshWilde/LensFindery-McLensFinderFace/blob/main/OU-200-4BANDS-CLEAR_3-1.png)



## Gravitational Compound Lenses
To investigate if these models can identify gravitational compound lenses, we needed to simulate these images. This was achieved using the Lenstronomy [3] and SkyPy [4] Python packages. In the paper [1], We generated two different compound lens datasets, one was mainly compound arcs and the other was mainly double Einstein rings.



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



## Citation


## References:
[1] Detecting gravitational lenses using machine learning: exploring interpretability and sensitivity to rare lensing configurations

[2] Ozbulak U., 2019, PyTorch CNN Visualizations, https://github.com/utkuozbulak/pytorch-cnn-visualizations 

[3] Birrer S., Amara A., 2018, Physics of the Dark Universe, 22, 189, https://github.com/sibirrer/lenstronomy

[4] SkyPy Collaboration et al., 2021, SkyPy, https://github.com/skypyproject/skypy
