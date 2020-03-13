import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        # Load weight values from file
        self.vgg = {} 
        self.vgg_conv1_1 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.vgg_conv1_2 = nn.Conv2d(64, 64, 3, bias=False, padding=1)
        self.vgg_conv1_3 = nn.Conv2d(64, 64, 3, bias=False, padding=1)
        self.vgg_conv1_4 = nn.Conv2d(64, 64, 3, bias=False, padding=1)
        
        self.vgg_conv2_1 = nn.Conv2d(64, 128, 3, bias=False, padding=1)
        self.vgg_conv2_2 = nn.Conv2d(128, 128, 3, bias=False, padding=1)
        self.vgg_conv2_3 = nn.Conv2d(128, 128, 3, bias=False, padding=1)
        self.vgg_conv2_4 = nn.Conv2d(128, 128, 3, bias=False, padding=1)
        
        self.vgg_conv3_1 = nn.Conv2d(128, 256, 3, bias=False, padding=1)
        self.vgg_conv3_2 = nn.Conv2d(256, 256, 3, bias=False, padding=1)
        self.vgg_conv3_3 = nn.Conv2d(256, 256, 3, bias=False, padding=1)
        self.vgg_conv3_4 = nn.Conv2d(256, 256, 3, bias=False, padding=1)
        
        self.vgg_conv4_1 = nn.Conv2d(256, 512, 3, bias=False, padding=1)
        self.vgg_conv4_2 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv4_3 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv4_4 = nn.Conv2d(512, 512, 3, bias=False, padding=1)

        self.vgg_conv5_1 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv5_2 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv5_3 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv5_4 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        
        self.vgg_conv6_1 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv6_2 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv6_3 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.vgg_conv6_4 = nn.Conv2d(512, 512, 3, bias=False, padding=1)

        self.vgg_fc6 = nn.Linear(8*8*512, 4096, bias=False)
        self.vgg_fc7 = nn.Linear(4096, 4096, bias=False)
        self.vgg_fc8 = nn.Linear(4096, 1, bias=False)
       
        self.vgg_pool = nn.MaxPool2d(2,2)
        self.dropout01 = nn.Dropout(0.1)
        self.dropout02 = nn.Dropout(0.2)
        self.dropout03 = nn.Dropout(0.3)
        self.dropout04 = nn.Dropout(0.4)
        self.dropout05 = nn.Dropout(0.5)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x): 
        # x = self.dropout01(x)
        #x = x/9.0
        x = self.leakyrelu(self.vgg_conv1_1(x))	
        x = self.leakyrelu(self.vgg_conv1_2(x))
        # x = self.dropout01(x)
        x = self.leakyrelu(self.vgg_conv1_3(x))
        x = self.leakyrelu(self.vgg_conv1_4(x))
        x = self.vgg_pool(x)
        
	# x = self.dropout05(x)  
        x = self.leakyrelu(self.vgg_conv2_1(x))
        x = self.leakyrelu(self.vgg_conv2_2(x))
        # x = self.dropout05(x)
        x = self.leakyrelu(self.vgg_conv2_3(x))
        x = self.leakyrelu(self.vgg_conv2_4(x))
        x = self.vgg_pool(x)

	# x = self.dropout05(x)
        x = self.leakyrelu(self.vgg_conv3_1(x))
        x = self.leakyrelu(self.vgg_conv3_2(x))
        x = self.leakyrelu(self.vgg_conv3_3(x))
        x = self.leakyrelu(self.vgg_conv3_4(x))
        x = self.vgg_pool(x)
       
	# x = self.dropout05(x)
        x = self.leakyrelu(self.vgg_conv4_1(x))
        x = self.leakyrelu(self.vgg_conv4_2(x))
        x = self.leakyrelu(self.vgg_conv4_3(x))
        x = self.leakyrelu(self.vgg_conv4_4(x))
        x = self.vgg_pool(x)

	# x = self.dropout05(x)
        x = self.leakyrelu(self.vgg_conv5_1(x))
        x = self.leakyrelu(self.vgg_conv5_2(x))
        x = self.leakyrelu(self.vgg_conv5_3(x))
        x = self.leakyrelu(self.vgg_conv5_4(x))
        x = self.vgg_pool(x)
        
	# x = self.dropout05(x)
        x = self.leakyrelu(self.vgg_conv6_1(x))
        x = self.leakyrelu(self.vgg_conv6_2(x))
        x = self.leakyrelu(self.vgg_conv6_3(x))
        x = self.leakyrelu(self.vgg_conv6_4(x))
        x = self.vgg_pool(x)
	
        x = x.view(-1, 8*8*512)
        x = self.dropout05(x)
        x = self.vgg_fc6(x)
        x = self.dropout05(x)
        x = self.vgg_fc7(x) 
        x = self.vgg_fc8(x)
        
        return x 

def grad_clip(net, max_grad = 0.1):
    params = [p for p in list(net.parameters()) if p.requires_grad==True]
    for p in params:
        p_grad = p.grad 

        if (type(p_grad) == type(None)):
            pdb.set_trace()
            here = 1 
        else:
            magnitude = torch.sqrt(torch.sum(p_grad**2)) 
            if (magnitude.data[0] > max_grad):
                #pdb.set_trace()
                p_grad.data = (max_grad*p_grad/magnitude.data[0]).data

