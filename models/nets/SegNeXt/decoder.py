#%%
import yaml
with open('models/nets/SegNeXt/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

from .hamburger import HamBurger
from .bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize


class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, outChannels, config=config, enc_embed_dims=[32,64,460,256]):
        super().__init__()

        ham_channels = config['ham_channels']

        # squeeze는 논문에서 head 이전 concat 역핳을 하는 듯 함.
        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        
        # Check parameters
        '''
        p_num = 0
        for name, p in self.ham_attn.named_parameters():
            p_num += p.numel()
        print(f"ham_attn: {round(p_num/1000000, 1)}M")
        '''
        
        # self.align = ConvRelu(ham_channels, outChannels) # 내가 주석한 부분
        # MLP인데, ReLU를 쓸 필요는 없어 보임.
        self.align = nn.Conv2d(ham_channels, outChannels, kernel_size=1)
       
    def forward(self, features):
        
        features = features[1:] # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)

        return x


#%%

# import torch.nn.functional as F

# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):

#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# inputs = [resize(
#         level,
#         size=x[0].shape[2:],
#         mode='bilinear',
#         align_corners=False
#     ) for level in x]

# for i in range(4):
#     print(x[i].shape)
# for i in range(4):
#     print(inputs[i].shape)



# inputs = torch.cat(inputs, dim=1)
# print(inputs.shape)
