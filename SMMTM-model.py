
import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)
import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
from utils.SincConv_util import SincConv2d
from utils.MHSA_util import MultiHeadSelfAttention
from utils.TCN_util import TemporalConvNet
from utils.EEGNet_util import EEGNet_util
from utils.util import Conv2dWithConstraint,LinearWithConstraint
import torch.nn.functional as F
import numpy as np

class TemporalInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1,3), kerSize_2=(1,5), kerSize_3=(1,7),
                 kerStr=1, out_chan=4, pool_ker=(1,3), pool_str=1, bias=False, max_norm=1.):

        super(TemporalInception, self).__init__()   
        self.conv1 = nn.Sequential(
        nn.Conv2d(
                in_channels=in_chan, 
                out_channels=out_chan*2,
                kernel_size=kerSize_2,
                stride=1,
                padding='same',
                groups=out_chan,
                bias=bias
            ),
        nn.Conv2d(
                in_channels=out_chan*2,
                out_channels=out_chan*2,
                kernel_size=(1,1),
                padding='same',
                stride=1,
                bias=bias
            ),
        nn.BatchNorm2d(num_features=out_chan*2),
        nn.ELU(), 
            nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(round(pool_ker[0]/2+0.1)-1,round(pool_ker[1]/2+0.1)-1)
        ),
        )     
        self.conv3 = nn.Sequential(
        nn.Conv2d(
                in_channels=in_chan, 
                out_channels=out_chan*2,
                kernel_size=kerSize_2,
                stride=1,
                padding='same',
                groups=out_chan,
                bias=bias
            ),
        nn.Conv2d(
                in_channels=out_chan*2,
                out_channels=out_chan*2,
                kernel_size=(1,1),
                padding='same',
                stride=1,
                bias=bias
            ),
        nn.BatchNorm2d(num_features=out_chan*2),
        nn.ELU(),
        )
        self.conv4 =  nn.Sequential(
        nn.Conv2d(
                in_channels=in_chan, 
                out_channels=out_chan*4,
                kernel_size=kerSize_2,
                stride=1,
                padding='same',
                groups=out_chan,
                bias=bias
            ),
        nn.Conv2d(
                in_channels=out_chan*4,
                out_channels=out_chan*4,
                kernel_size=(1,1),
                padding='same',
                stride=1,
                bias=bias
            ),
        nn.BatchNorm2d(num_features=out_chan*4),
        nn.ELU(),
        )
    def forward(self, x):
        p1 = self.conv1(x)      
        p3 = self.conv3(x)
        p4 = self.conv4(x)
        out = torch.cat((p1,p3,p4), dim=1)
        return out


#%%
class My_Model(nn.Module):
    def __init__(self, F1=32, D=2, F2=64, kerSize=16, eeg_chans=22, poolSize=8, kerSize_Tem=4, dropout_dep=0.5, dropout_temp=0.5, 
                 dropout_atten=0.3, tcn_filters=64, tcn_kernelSize=4, tcn_dropout=0.5,n_classes=4):
        super(My_Model, self).__init__()

        self.sincConv =  nn.Conv2d(
            in_channels = 1, 
            out_channels= F1, 
            kernel_size = (1,kerSize), 
            stride      = 1,
            padding     = 'same',
            bias        = False
        )

        self.bn_sinc = nn.BatchNorm2d(num_features=F1)
 
        self.conv_depth = Conv2dWithConstraint(
            in_channels = F1,
            out_channels= F1*D,
            kernel_size = (eeg_chans,1),
            groups      = F1,
            bias        = False,
            max_norm    = 1.
        )

        self.bn_depth = nn.BatchNorm2d(num_features=64)
        self.act_depth = nn.ELU()
        self.avgpool_depth = nn.AvgPool2d(
            kernel_size=(1,8),
            stride=(1,8)
        )
        self.drop_depth = nn.Dropout(p=dropout_dep)

        self.incept_temp = TemporalInception(
            in_chan     = 64,
            kerSize_1   = (1,8),
            kerSize_2   = (1,8),
            kerSize_3   = (1,32),
            kerStr      = 1,
            out_chan    = 8,
            pool_ker    = (3,3),
            pool_str    = 1,
            bias        = False,
            max_norm    = .5
        )

        self.bn_temp = nn.BatchNorm2d(num_features=64)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(
            kernel_size=(1,poolSize),
            stride=(1,poolSize)
        )
        self.drop_temp = nn.Dropout(p=dropout_temp)

        self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(
            in_features  = 960,
            out_features = n_classes,
            max_norm     = .25,
            bias         = True
        )

        self.layerNorm = nn.LayerNorm(
            normalized_shape=64,
            eps=1e-6
        )
      
        self.multihead_attn = MultiHeadSelfAttention(
            embed_dim = 64,
            heads     = 8,
            dropout   = dropout_atten,
            norm      = .25
        )
     

        self.tcn_block = TemporalConvNet(
            num_inputs   = 64,
            num_channels = [192, 192],
            kernel_size  = 8,    
            dropout      = tcn_dropout,
            bias         = False,
            WeightNorm   = True,
            max_norm     = .5
        )

        self.tcn_block1 = TemporalConvNet(
            num_inputs   = 192,
            num_channels = [192, 192],
            kernel_size  = 8,     
            dropout      = tcn_dropout,
            bias         = False,
            WeightNorm   = True,
            max_norm     = .5
        )

        self.flatten_tcn = nn.Flatten()
        self.liner_tcn = LinearWithConstraint(
            in_features  = 192,
            out_features = n_classes,
            max_norm     = .25,
            bias         = True
        )

        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta_sigmoid = nn.Sigmoid()       

        self.softmax = nn.Softmax(dim=-1)
       
        self.dense = LinearWithConstraint(
            in_features=4,
            out_features=n_classes,
            max_norm=.25
        )


    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        
        x = self.sincConv(x)
        x = self.bn_sinc(x)

        x = self.conv_depth(x)
        x = self.drop_depth(self.avgpool_depth(self.act_depth(self.bn_depth(x))))
       
        x = self.incept_temp(x)
        x = self.drop_temp(self.avgpool_temp(self.act_temp(self.bn_temp(x)))) # (batch, F1*D, 1, 15)
        x = torch.squeeze(x, dim=2) # (batch, F1*D, 15)
        x = self.tcn_block(x)
           
        x = x[:, :, -1]
        tcn_out = self.liner_tcn(self.flatten_tcn(x))
        x = self.dense(tcn_out)
        
        out = self.softmax(x)

        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = My_Model(eeg_chans=22, n_classes=4)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()

