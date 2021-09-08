mport numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import glob
import torch.utils.data as data_utils
import math 
import time
import datetime
from astropy.io import fits
from sklearn import metrics


def J_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    SourceJ = glob.glob('EUC_J/*fits')
    hdu_listJ = fits.open(SourceJ[0])
    Images = np.zeros((Output.shape[0],1,hdu_listJ[0].data.shape[0],hdu_listJ[0].data.shape[1]))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_listJ = fits.open(str(SourceJ[B][:-11])+str(int(Output[B,0]))+'.fits')
            
            #print(str(SourceJ[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = hdu_listJ[0].data
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))           
            #Close the FITS file
            hdu_listJ.close()
            NewOutput.append([int(Output[B,0]),Output[B,1]])

                    
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


Images, NewOutput = LoadImages(Output)


def Y_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    SourceY = glob.glob('EUC_Y/*fits')
    hdu_listY = fits.open(SourceY[0])
    Images = np.zeros((Output.shape[0],1,hdu_listY[0].data.shape[0],hdu_listY[0].data.shape[1]))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_listY = fits.open(str(SourceY[B][:-11])+str(int(Output[B,0]))+'.fits')
            
            #print(str(SourceY[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = hdu_listY[0].data
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))           
            #Close the FITS file
            hdu_listY.close()
            NewOutput.append([int(Output[B,0]),Output[B,1]])

                    
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


Images, NewOutput = LoadImages(Output)



def H_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    SourceH = glob.glob('EUC_H/*fits')
    hdu_listH = fits.open(SourceH[0])
    Images = np.zeros((Output.shape[0],1,hdu_listH[0].data.shape[0],hdu_listH[0].data.shape[1]))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_listH = fits.open(str(SourceH[B][:-11])+str(int(Output[B,0]))+'.fits')
            
            #print(str(SourceH[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = hdu_listH[0].data
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))           
            #Close the FITS file
            hdu_listH.close()
            NewOutput.append([int(Output[B,0]),Output[B,1]])

                    
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


Images, NewOutput = LoadImages(Output)


def JYH_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    SourceJ = glob.glob('EUC_J/*fits')
    hdu_listJ = fits.open(SourceJ[0])
    SourceY = glob.glob('EUC_Y/*fits')
    hdu_listY = fits.open(SourceY[0])
    SourceH = glob.glob('EUC_H/*fits')
    hdu_listH = fits.open(SourceH[0])
    Images = np.zeros((Output.shape[0],3,hdu_listJ[0].data.shape[0],hdu_listJ[0].data.shape[1]))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_listJ = fits.open(str(SourceJ[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_listY = fits.open(str(SourceY[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_listH = fits.open(str(SourceH[B][:-11])+str(int(Output[B,0]))+'.fits')
            
            #print(str(SourceJ[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = hdu_listJ[0].data
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))
            Images[a,1,:,:] = hdu_listY[0].data
            Images[a,1,:,:] = (Images[a,1,:,:]-np.amin(Images[a,1,:,:]))/(np.amax(Images[a,1,:,:])-np.amin(Images[a,1,:,:]))
            Images[a,2,:,:] = hdu_listH[0].data
            Images[a,2,:,:] = (Images[a,2,:,:]-np.amin(Images[a,2,:,:]))/(np.amax(Images[a,2,:,:])-np.amin(Images[a,2,:,:]))
            
            
            #Close the FITS file
            hdu_listJ.close()
            hdu_listY.close()
            hdu_listH.close()
            NewOutput.append([int(Output[B,0]),Output[B,1]])

                    
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


Images, NewOutput = LoadImages(Output)



def VIS_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    Source_VIS = glob.glob('EUC_VIS/*fits')
    hdu_list_VIS = fits.open(Source_VIS[0])

    Images = np.zeros((Output.shape[0],1,hdu_list_VIS[0].data.shape[0],hdu_list_VIS[0].data.shape[1]))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_list_VIS = fits.open(str(Source_VIS[B][:-11])+str(int(Output[B,0]))+'.fits')
            #print(str(Source_VIS[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,:,:,:] = hdu_list_VIS[0].data
            Images[a,:,:,:] = (Images[a,:,:,:]-np.amin(Images[a,:,:,:]))/(np.amax(Images[a,:,:,:])-np.amin(Images[a,:,:,:]))
            hdu_list_VIS.close()   
            
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


VIS_Images, NewOutput = Load_VIS_Images(Output)


def OU66_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    Source_J = glob.glob('EUC_J/*fits')
    hdu_list_J = fits.open(Source_J[0])
    Source_Y = glob.glob('EUC_Y/*fits')
    hdu_list_Y = fits.open(Source_Y[0])
    Source_H = glob.glob('EUC_H/*fits')
    hdu_list_H = fits.open(Source_H[0])
    Source_VIS = glob.glob('EUC_VIS/*fits')
    hdu_list_VIS = fits.open(Source_VIS[0])

    Images = np.zeros((Output.shape[0],4,66,66))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_list_J = fits.open(str(Source_J[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_Y = fits.open(str(Source_Y[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_H = fits.open(str(Source_H[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_VIS = fits.open(str(Source_VIS[B][:-11])+str(int(Output[B,0]))+'.fits')


            #print(str(Source_J[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))

            Images[a,1,:,:] = (Images[a,1,:,:]-np.amin(Images[a,1,:,:]))/(np.amax(Images[a,1,:,:])-np.amin(Images[a,1,:,:]))

            Images[a,2,:,:] = (Images[a,2,:,:]-np.amin(Images[a,2,:,:]))/(np.amax(Images[a,2,:,:])-np.amin(Images[a,2,:,:]))

            Images[a,0,:,:] = resize(hdu_list_VIS[0].data, (66,66))
            Images[a,3,:,:] = (Images[a,3,:,:]-np.amin(Images[a,3,:,:]))/(np.amax(Images[a,3,:,:])-np.amin(Images[a,3,:,:]))

            hdu_list_J.close()   
            hdu_list_Y.close()   
            hdu_list_H.close()   
            hdu_list_VIS.close()   
            NewOutput.append([int(Output[B,0]),Output[B,1]]) 
            
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput


def OU200_LoadImages(Output):
    NewOutput = []
    badValues = [213913, 226305, 233597, 244071, 259871, 261145, 270458, 288731, 294173]
    Source_J = glob.glob('EUC_J/*fits')
    hdu_list_J = fits.open(Source_J[0])
    Source_Y = glob.glob('EUC_Y/*fits')
    hdu_list_Y = fits.open(Source_Y[0])
    Source_H = glob.glob('EUC_H/*fits')
    hdu_list_H = fits.open(Source_H[0])
    Source_VIS = glob.glob('EUC_VIS/*fits')
    hdu_list_VIS = fits.open(Source_VIS[0])

    Images = np.zeros((Output.shape[0],4,200,200))
    a = 0
    T = 0
    
    for B in range(0,Output.shape[0]):
      #print(Output.shape[0])
      if Output[B,0] in badValues:
              print('Bad Value')
      else:
            if a == 0 :
                a = B
            #print(B)
            #Open the FITS file
            hdu_list_J = fits.open(str(Source_J[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_Y = fits.open(str(Source_Y[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_H = fits.open(str(Source_H[B][:-11])+str(int(Output[B,0]))+'.fits')
            hdu_list_VIS = fits.open(str(Source_VIS[B][:-11])+str(int(Output[B,0]))+'.fits')


            #print(str(Source_J[B][:-11])+str(int(Output[B,0]))+'.fits')
            #Extract image data
            Images[a,0,:,:] = resize(hdu_list_J[0].data, (200,200))
            Images[a,0,:,:] = (Images[a,0,:,:]-np.amin(Images[a,0,:,:]))/(np.amax(Images[a,0,:,:])-np.amin(Images[a,0,:,:]))

            Images[a,1,:,:] = resize(hdu_list_Y[0].data, (200,200))
            Images[a,1,:,:] = (Images[a,1,:,:]-np.amin(Images[a,1,:,:]))/(np.amax(Images[a,1,:,:])-np.amin(Images[a,1,:,:]))

            Images[a,2,:,:] = resize(hdu_list_H[0].data, (200,200))
            Images[a,2,:,:] = (Images[a,2,:,:]-np.amin(Images[a,2,:,:]))/(np.amax(Images[a,2,:,:])-np.amin(Images[a,2,:,:]))

            Images[a,3,:,:] = hdu_list_VIS[0].data
            Images[a,3,:,:] = (Images[a,3,:,:]-np.amin(Images[a,3,:,:]))/(np.amax(Images[a,3,:,:])-np.amin(Images[a,3,:,:]))

            hdu_list_J.close()   
            hdu_list_Y.close()   
            hdu_list_H.close()   
            hdu_list_VIS.close()   
            NewOutput.append([int(Output[B,0]),Output[B,1]]) 
            
            a = a+1
    NewOutput = np.array(NewOutput)
    return Images, NewOutput










