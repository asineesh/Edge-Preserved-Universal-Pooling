import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

    
def gauss_kernel(device,channels,size):
    if size == 6:
      kernel = torch.tensor([[0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010],
        [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
        [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
        [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
        [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
        [0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010]])  
      
    if size ==2:
      kernel = torch.tensor([[0.2500, 0.2500],
        [0.2500, 0.2500]])
      
    if size == 3:
      kernel = torch.tensor([[0.0625, 0.1250, 0.0625],
        [0.1250, 0.2500, 0.1250],
        [0.0625, 0.1250, 0.0625]])
    
    if size == 5:
      kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    t = x[:, :, ::2, ::2]
    return t

def upsample(x,size,pad):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(device=x.device,channels=x.shape[1],size=size),size,pad)

def conv_gauss(img, kernel,filt_size,pad_off):
  if(img.shape[3]%2==0):
    pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
    pad_sizes = [pad_size+pad_off for pad_size in pad_sizes]
  else:
    pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
    pad_sizes = [pad_size+pad_off for pad_size in pad_sizes]
  img = torch.nn.functional.pad(img, pad_sizes , mode='reflect')
  out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
  return out

def laplacian_pyramid(img,size,device,channels,pad):
    current = img
    kernel = gauss_kernel(device,channels,size)
    filtered = conv_gauss(current, kernel,size,pad)    
    diff = current-filtered
    return diff, filtered

class SE(nn.Module):
    def __init__(self, channel, reduction_ratio =16):
        super(SE, self).__init__()
        ### Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class ConAvg(nn.Module):
    def __init__(self,inc,device,size):
        super(ConAvg,self).__init__()
        self.device = device
        self.inc = inc
        self.size = size
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3,padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)            
        self.attn = SE(2*inc)
                      

    def forward(self, x):
        x1 , x2 = laplacian_pyramid(img=x,size=self.size,device=self.device,channels=self.inc,pad=0)
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x.to(device=self.device))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x.to(self.device)
    
class ConMax(nn.Module):
    def __init__(self,inc,device,size):
        super(ConMax,self).__init__()
        self.device = device
        self.inc = inc
        self.size = size
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)        
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.pool1(x)
        x1 , x2 = laplacian_pyramid(img=x,size=self.size,device=self.device,channels=self.inc,pad=0)
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x.to(device=self.device))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x.to(self.device)
    
class ConConv(nn.Module):
    def __init__(self,inc,device,size):
        super(ConConv,self).__init__()
        self.device = device
        self.inc = inc
        self.size = size
        self.relu6 = nn.ReLU6()
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3, stride =2,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)        
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.relu6(x) 
        x1 , x2 = laplacian_pyramid(img=x,size=self.size,device=self.device,channels=self.inc,pad=0)
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x.to(device=self.device))
        x = self.relu(self.conv2(x))
        
        return x.to(self.device)
    
class ConConvb(nn.Module):
    def __init__(self,inc,device,size):
        super(ConConvb,self).__init__()
        self.device = device
        self.inc = inc
        self.size = size
        self.bn = nn.BatchNorm2d(inc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3, stride =2,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)        
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.bn(x) 
        x1 , x2 = laplacian_pyramid(img=x,size=self.size,device=self.device,channels=self.inc,pad=0)     
        x = torch.cat((x1,x2),1)
        x = self.attn(x.to(device=self.device))
        x = self.relu(self.conv2(x))
        return x.to(self.device)
        
class ConMaxA(nn.Module):
    def __init__(self,inc,device,size):
        super(ConMaxA,self).__init__()
        self.device = device
        self.inc = inc
        self.size = size
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)        
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.pool1(x)
        x1 , x2 = laplacian_pyramid(img=x,size=self.size,device=self.device,channels=self.inc,pad=0)
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x.to(device=self.device))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x.to(self.device)
    
class ConConv1(nn.Module):
    def __init__(self,inc,device):
        super(ConConv1,self).__init__()
        self.device = device
        self.inc = inc
        self.bn = nn.BatchNorm2d(inc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             

    def forward(self, x):
        x = self.bn(x) 
        x = x[:,:,::2,::2]
        return x.to(self.device)

