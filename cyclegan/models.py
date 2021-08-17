import torch 
import torch.nn as nn
import torch.nn.functional as F

# According to paper weights are initialized from Gaussian distribution with 0 mean
# and 0.02 standard deviation

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
        if hasattr(m,'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0.0)

    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
        torch.nn.init.constant_(m.bias.data,0.0)


## Paper Uses Residual blocks 

class ResidualBlocks(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlocks,self).__init__()

        self.main = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels,in_channels,3),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace = True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels,in_channels,3),
                nn.InstanceNorm2d(in_channels),
        )
    def forward(self,x):
        return x + self.main(x) # skip connection is made this way
 


## Generator Architecture has Resdiual Blocks

class GeneratorResNet(nn.Module):
    def __init__(self,img_dims,num_res_blocks):
        super(GeneratorResNet,self).__init__()

        channels = img_dims[0]

        # First conv block

        model = [
                nn.ReflectionPad2d(channels),
                nn.Conv2d(channels,64,7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace = True)
        ]

        out_features = 64
        in_features = out_features

        # Downsampling

        for _ in range(2):
            out_features *= 2
            model += [
                    nn.Conv2d(in_features,out_features,3,2,1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace = True),
            ]
            in_features = out_features

        # Residual Blocks

        for _ in range(num_res_blocks):
            model += [ResidualBlocks(out_features)]

        # Upsampling 

        for _ in range(2):
            out_features //= 2
            model += [
                    nn.Upsample(scale_factor = 2),
                    nn.Conv2d(in_features,out_features,3,1,1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace = True)
            ]
            in_features = out_features


        # Output layer 

        model += [
                nn.ReflectionPad2d(channels),
                nn.Conv2d(out_features,channels,7),
                nn.Tanh()
        ]

        self.model = nn.Sequential(*model)


    def forward(self,x):
        return self.model(x)
    
            
## Discriminator Architecture

class Discriminator(nn.Module):
    def __init__(self,img_dims):
        super(Discriminator,self).__init__()

        channels,h,w = img_dims

        # output shape of image discriminator using PatchGAN

        self.output_shape = (1,h//2 ** 4, w//2  ** 4)

        def discriminatorBlock(in_channels,out_channels,normalize = True):

            layers = [nn.Conv2d(in_channels,out_channels,4,2,1)]

            if normalize : 
                layers += [nn.InstanceNorm2d(out_channels)]
            layers.append(nn.LeakyReLU(0.2,inplace = True))

            return layers

        self.model = nn.Sequential(

                *discriminatorBlock(channels,64,False),
                *discriminatorBlock(64,128),
                *discriminatorBlock(128,256),
                *discriminatorBlock(256,512),
                nn.ZeroPad2d((1,0,1,0)),
                nn.Conv2d(512,1,4,padding = 1)

        )

    def forward(self,x):
        return self.model(x)

