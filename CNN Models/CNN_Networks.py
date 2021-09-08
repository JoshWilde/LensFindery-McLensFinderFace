import numpy as np
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

class J_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(J_CNN,self).__init__()
        #self.num_feature=num_feature
        
        self.layer = nn.Sequential(            
             nn.Conv2d(1,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50, 40, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),  
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),          
             nn.Conv2d(40, 30, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),

        )
        self.fc_layer = nn.Sequential(
          #nn.Linear(65,65),
          #nn.Linear(65,36),
          nn.Linear(120,500),  
          nn.ReLU(),
          nn.Linear(500, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        #out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        out = self.fc_layer(out)
        softmaxProbabilities = torch.softmax((out),dim=1)

        return out, softmaxProbabilities



class Y_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(Y_CNN,self).__init__()
        #self.num_feature=num_feature
        
        self.layer = nn.Sequential(            
             nn.Conv2d(1,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50, 40, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),  
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),          
             nn.Conv2d(40, 30, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),

        )
        self.fc_layer = nn.Sequential(
          #nn.Linear(65,65),
          #nn.Linear(65,36),
          nn.Linear(120,500),  
          nn.ReLU(),
          nn.Linear(500, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        #out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        out = self.fc_layer(out)
        softmaxProbabilities = torch.softmax((out),dim=1)

        return out, softmaxProbabilities




class H_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(H_CNN,self).__init__()
        #self.num_feature=num_feature
        
        self.layer = nn.Sequential(            
             nn.Conv2d(1,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50, 40, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),  
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),          
             nn.Conv2d(40, 30, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),

        )
        self.fc_layer = nn.Sequential(
          #nn.Linear(65,65),
          #nn.Linear(65,36),
          nn.Linear(120,500),  
          nn.ReLU(),
          nn.Linear(500, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        #out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        out = self.fc_layer(out)
        softmaxProbabilities = torch.softmax((out),dim=1)

        return out, softmaxProbabilities




class VIS_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(VIS_CNN,self).__init__()
        #self.num_feature=num_feature
                
        self.VIS_layer = nn.Sequential(            
             nn.Conv2d(1,50,kernel_size=5,padding=0),
             nn.ReLU(),            
             nn.MaxPool2d(2, 2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(50,40,kernel_size=5,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
 	     nn.Dropout(p=0.2),
	     nn.Conv2d(40,30,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(30,20,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(20,20,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(20,20,kernel_size=3,padding=0),
	     nn.ReLU(),
 	     nn.Dropout(p=0.2),
        )
        
        
        
        self.fc_layer = nn.Sequential(
          nn.Linear(720,350),  
          nn.ReLU(),
          nn.Linear(350, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,VIS):
        

        
        out = self.VIS_layer(VIS)
        out = out.view(out.size()[0],-1)
        out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        #VIS_out = VIS_out.view(VIS_out.size()[0],-1)
        #out = out.flatten(start_dim=1)
        #VIS_out = torch.squeeze(VIS_out)
        #VIS_out = self.VIS_To_FC(VIS_out)
        #print(JYH_out.size())

        #print(VIS_out.size())
        
        
        #combined = torch.cat((J,
        #                     VIS_out),dim=1)
        #print(combined.size())
        out = self.fc_layer(out)
        #print(out.size())
        softmaxProbabilities = torch.softmax((out),dim=0)

        return out, softmaxProbabilities









class JYH_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(JYH_CNN,self).__init__()
        #self.num_feature=num_feature
        
        self.layer = nn.Sequential(            
             nn.Conv2d(3,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50, 40, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),  
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),          
             nn.Conv2d(40, 30, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),

        )
        self.fc_layer = nn.Sequential(
          #nn.Linear(65,65),
          #nn.Linear(65,36),
          nn.Linear(120,500),  
          nn.ReLU(),
          nn.Linear(500, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        #out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        out = self.fc_layer(out)
        softmaxProbabilities = torch.softmax((out),dim=1)

        return out, softmaxProbabilities
        
class OU66_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(OU66_CNN,self).__init__()
        #self.num_feature=num_feature
        
        self.layer = nn.Sequential(            
             nn.Conv2d(4,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50,50,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(50, 40, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),  
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),
             nn.Conv2d(40,40,kernel_size=3,padding=0),
             nn.ReLU(),
             nn.Dropout(p=0.2),          
             nn.Conv2d(40, 30, kernel_size=3, padding=0),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.Dropout(p=0.2),

        )
        self.fc_layer = nn.Sequential(
          #nn.Linear(65,65),
          #nn.Linear(65,36),
          nn.Linear(120,500),  
          nn.ReLU(),
          nn.Linear(500, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        #out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        out = self.fc_layer(out)
        softmaxProbabilities = torch.softmax((out),dim=1)

        return out, softmaxProbabilities
        
        
class OU200_CNN(nn.Module):
    def __init__(self):#,num_feature=30):
        super(OU200_CNN,self).__init__()
        #self.num_feature=num_feature
                
        self.layer = nn.Sequential(            
             nn.Conv2d(4,50,kernel_size=5,padding=0),
             nn.ReLU(),            
             nn.MaxPool2d(2, 2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(50,40,kernel_size=5,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
 	     nn.Dropout(p=0.2),
	     nn.Conv2d(40,30,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(30,20,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.MaxPool2d(2,2),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(20,20,kernel_size=3,padding=0),
	     nn.ReLU(),
	     nn.Dropout(p=0.2),
	     nn.Conv2d(20,20,kernel_size=3,padding=0),
	     nn.ReLU(),
 	     nn.Dropout(p=0.2),
        )
        
        
        
        self.fc_layer = nn.Sequential(
          nn.Linear(720,350),  
          nn.ReLU(),
          nn.Linear(350, 2)
        )       
        
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
                
            elif isinstance(m, nn.Linear):
        
                # Kaming Initialization
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        
    def forward(self,VIS):
        

        
        out = self.layer(VIS)
        out = out.view(out.size()[0],-1)
        out = out.flatten(start_dim=1)
        out = torch.squeeze(out)
        #VIS_out = VIS_out.view(VIS_out.size()[0],-1)
        #out = out.flatten(start_dim=1)
        #VIS_out = torch.squeeze(VIS_out)
        #VIS_out = self.VIS_To_FC(VIS_out)
        #print(JYH_out.size())

        #print(VIS_out.size())
        
        
        #combined = torch.cat((J,
        #                     VIS_out),dim=1)
        #print(combined.size())
        out = self.fc_layer(out)
        #print(out.size())
        softmaxProbabilities = torch.softmax((out),dim=0)

        return out, softmaxProbabilities
