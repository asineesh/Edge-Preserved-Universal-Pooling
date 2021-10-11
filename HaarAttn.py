import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

def downsample(x):
    t = x[:, :, ::2, ::2]
    return t
   
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
    
class HaarAvg(nn.Module):
    def __init__(self,inc,device):
        super(HaarAvg,self).__init__()
        self.device = device
        self.inc = inc
        
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3,padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)          
        self.attn = SE(2*inc)
                      

    def forward(self, x):
        if self.device.type == 'cuda':
            xfm = DWTForward(J=1, mode='symmetric', wave='haar').cuda()
            yfm = DWTInverse(mode='symmetric', wave='haar').cuda()
        else:            
            xfm = DWTForward(J=1, mode='symmetric', wave='haar')
            yfm = DWTInverse(mode='symmetric', wave='haar')
            
        al,ah = xfm(x)  
        
        c=torch.zeros_like(al)
        x1 = yfm((c,ah)) 
        
        d=torch.zeros_like(ah[0])
        r= [d]  
        x2 = yfm((al,r))
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x
    
class HaarMax(nn.Module):
    def __init__(self,inc,device):
        super(HaarMax,self).__init__()
        self.device = device
        self.inc = inc
        
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)
        self.attn = SE(2*inc)
              

    def forward(self, x):
        if self.device.type == 'cuda':
            xfm = DWTForward(J=1, mode='symmetric', wave='haar').cuda()
            yfm = DWTInverse(mode='symmetric', wave='haar').cuda()
        else:            
            xfm = DWTForward(J=1, mode='symmetric', wave='haar')
            yfm = DWTInverse(mode='symmetric', wave='haar')
            
        al,ah = xfm(x)  
        
        c=torch.zeros_like(al)
        x1 = yfm((c,ah)) 
        
        d=torch.zeros_like(ah[0])
        r= [d]  
        x2 = yfm((al,r))
        
        x = torch.cat((x1,x2),1)
        x = self.attn(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x
    
class HaarConv(nn.Module):
    def __init__(self,inc,device):
        super(HaarConv,self).__init__()
        self.device = device
        self.inc = inc
        
        self.relu6 = nn.ReLU6()
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3, stride =2,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.relu6(x) 
        if self.device.type == 'cuda':
            xfm = DWTForward(J=1, mode='symmetric', wave='haar').cuda()
            yfm = DWTInverse(mode='symmetric', wave='haar').cuda()
        else:            
            xfm = DWTForward(J=1, mode='symmetric', wave='haar')
            yfm = DWTInverse(mode='symmetric', wave='haar')
            
        al,ah = xfm(x)  
        
        c=torch.zeros_like(al)
        x1 = yfm((c,ah)) 
        
        d=torch.zeros_like(ah[0])
        r= [d]  
        x2 = yfm((al,r))
        x = torch.cat((x1,x2),1)
        x = self.attn(x)
        x = self.relu(self.conv2(x))
        
        return x
    
class HaarConvb(nn.Module):
    def __init__(self,inc,device):
        super(HaarConvb,self).__init__()
        self.device = device
        self.inc = inc
        
        self.bn = nn.BatchNorm2d(inc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2 = nn.Conv2d (2*inc,inc, kernel_size=3, stride =2,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu = nn.ReLU(inplace=False)
        self.attn = SE(2*inc)
              

    def forward(self, x):
        x = self.bn(x) 
                   
        if self.device.type == 'cuda':
            xfm = DWTForward(J=1, mode='symmetric', wave='haar').cuda()
            yfm = DWTInverse(mode='symmetric', wave='haar').cuda()
        else:            
            xfm = DWTForward(J=1, mode='symmetric', wave='haar')
            yfm = DWTInverse(mode='symmetric', wave='haar')
            
        al,ah = xfm(x) 
        
        c=torch.zeros_like(al)
        x1 = yfm((c,ah)) 
        
        d=torch.zeros_like(ah[0])
        r= [d]  
        x2 = yfm((al,r))
        x = torch.cat((x1,x2),1)
        x = self.attn(x)
        x = self.relu(self.conv2(x))
        
        return x

class HaarConv1(nn.Module):
    def __init__(self,inc,device):
        super(HaarConv1,self).__init__()
        self.device = device
        self.inc = inc
        self.bn = nn.BatchNorm2d(inc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
             

    def forward(self, x):
        x = self.bn(x) 
        x = downsample(x)
        return x
            
       