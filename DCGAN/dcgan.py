import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## Setting random seed 
manualSeed = 0
print("Random Seed : ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Setting up valuable vars

dataroot = 'data'
workers = 2 # number of worker threads for loading data with DataLoader
batch_size = 64
image_size = 64
nc = 3 # number of channels
nz = 100 # size of latent vector space
ngf = 64 # size of feature maps (intermediate layer output) in generator
ndf = 64 # size of feature maps in discriminator 
num_epochs = 1
lr = 0.0002 
beta1 = 0.5 # for Adam optimizer
ngpu = 0 # 0 for CPU mode



## Dataset

dataset = datasets.ImageFolder(root = dataroot,transform = 
                                transforms.Compose([transforms.Resize(image_size),
                                                    transforms.CenterCrop(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,))]))

dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers =  2)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)  else  "cpu")


# Plotting some training images

real_batch = next(iter(dataloader))

plt.figure(figsize = (8,8))
plt.axis('off')

plt.title("training images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64],padding=2,
            normalize = True).cpu(),(1,2,0)))

plt.show()



# weight initialization for our generator and discriminator 

def weights_init(m) : 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 : 
        nn.init.normal_(m.weight.data,0.0,0.02)
    if classname.find('BatchNorm') != -1 : 
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.normal_(m.bias.data,0)


##### Generator #####

class Generator(nn.Module) : 
    def __init__(self,ngpu) :
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is the latent vector Z
                nn.ConvTranspose2d(nz,ngf * 8,4,1,0,bias = False),
                nn.BatchNorm2d(ngf*8),
                nn.ReLU(True),

                # state size 512 x 4 x 4
                nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias = False),
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),

                # state size 256 x 8 x 8 
                nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias = False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),

                # state size 128 x 16 x 16
                nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias = False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                # state size 64 x 32 x 32
                nn.ConvTranspose2d(ngf,nc,4,2,1,bias = False),
                nn.Tanh() 
                 
                # state size nc x 64 x 64
        )

    def forward(self,input):
        return self.main(input)



genNet = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weight_init function to randomly initialize weights to mean 0 and std 0.2
genNet.apply(weights_init)

print(genNet)



##### Discriminator #####


class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is nc x 64 x 64
                nn.Conv2d(nc,ndf,4,2,1,bias = False),
                nn.LeakyReLU(0.2,inplace = True),

                # state size 64 x 32 x 32
                nn.Conv2d(ndf,ndf*2,4,2,1,bias = False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2,inplace = True),

                # state size 128 x 16 x 16
                nn.Conv2d(ndf*2,ndf*4,4,2,1,bias = False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2,inplace = True),

                # state size 256 x 8 x 8 
                nn.Conv2d(ndf*4,ndf*8,4,2,1,bias = False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2,inplace = True),

                # state size 128 x 4 x 4
                nn.Conv2d(ndf*8,1, 4,1,0,bias = False),
                nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input)


discNet = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weight_init function to randomly initialize weights to mean 0 and std 0.2
discNet.apply(weights_init)

print(discNet)


#### Loss Function and Optimizers ####

criterion = nn.BCELoss()

# Creating a Batch of latent vectors that we will use to visualize the progression of 
# generator, as our generator stars progressing it will learn to generate nice images 
# from this fixed noise vector space

fixed_noise = torch.randn(64,nz,1,1,device = device)

# Label convention to help us calculate loss of D and G 
real_label = 1.
fake_label = 0.

# Setting up Adam Optimizer for both G and D
optimizerD = optim.Adam(genNet.parameters(),lr = lr,betas = (beta1,0.999))
optimizerG = optim.Adam(discNet.parameters(),lr = lr,betas = (beta1,0.999))


### Training Time ###

# Part 1 -- Training the Discriminator

# we want to maximize the loss function for discriminator which is : 
# (log(D(x)) + log(1-D(G(z)))
# we will use two seperate mini batches one with real images, and one with fake ones

# Firstly we pass the real mini batch to D and calculate first half of the loss function
# i.e log(D(x)), then calculate the gradients in a backward pass

# Secondly, we will then pass the fake batch to current generator and then forward pass
# this through D, calcluate loss's 2nd part i.e. log(1-D(G(z))) and accumulate grads
# in backward pass, now we have both the grads of fake and real batches so we step the 
# optimizer of Discriminator


# Part 2 -- Trainiing the Generator 

# Original stated in paper, we should mininmize the log(1-D(G(z))) in effort to make 
# the G better but Goodfellow said it doesn't provide sufficient gradients, especially early
# in the learning process. As a fix, we instead wish to maximize log(D(G(z)))

# For this we classify the G's output from Part 1 and compute the G's loss using real_labels
# which are all 1 as GT (this makes the (1-y)log(1-x) part become 0 which is what we want)
# so now we have log(x) part which want to maximize


#### Let's Start ####

img_list = []
G_losses = []
D_losses =  []

iters = 0

print("Starting the Players...")

inital_time = time.time()
for epoch in range(num_epochs):
    start_time_epoch = time.time()

    # Each batch
    for i,data in enumerate(dataloader,0):
            
        # Part 1 #
        ### real batch ###
        #### Update the Discriminator ####
        start_iter = time.time()

        discNet.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label, dtype = torch.float,device = device)

        # Forward pass  
        output = discNet(real_cpu).view(-1)

        # Loss on real batch
        loss_D_real = criterion(output,label)

        # Calculate gradients for D in backward pass
        loss_D_real.backward()
        D_x = output.mean().item()

        ### fake batch ###

        # Generating batch of latent vectors
        noise = torch.randn(b_size,nz,1,1,device = device) 

        fake = genNet(noise)
        label.fill_(fake_label)

        #Classifying the fake ones by D
        output = discNet(fake.detach()).view(-1) # we detach gradient tracker for noise

        # Loss on fake batch
        loss_D_fake = criterion(output,label)

        # Calculate gradients for D in backward pass now for fake batch
        loss_D_fake.backward()
        D_G_z = output.mean().item()

        ## We got all grads from real and fake batch now we step the optimizer of D
        loss_D = loss_D_real + loss_D_fake
        optimizerD.step()

        # Part 2 # 
        #### Update the Generator #### 
        genNet.zero_grad()

        # for generator fake labels are real 
        label.fill_(real_label)
# Since we just updated D, perform another foward pass for all fake batch output = discNet(fake).view(-1)
        
        # Calculate G's loss now 
        loss_G = criterion(output,label)

        # Calcluate G's grads
        loss_G.backward()
        D_G_z2 = output.mean().item()

        # Step the optimizer for G
        optimizerG.step()


        ## Stats
        if i % 50 == 0: 
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %( epoch,num_epochs,
                                    i,len(dataloader),loss_D.item(),loss_G.item(),
                                    D_x,D_G_z,D_G_z2))
            time_elapsed = time.time()-start_time_epoch
            print("time : {:.1f} min {:.2f} sec".format(time_elapsed//60,time_elapsed%60))

        # Saving losses 
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Checking how generator is doing on fixed noise
        if(iters % 500 == 0) or ((epoch == num_epochs -1 ) and (i == len(dataloader) -1)):
            with torch.no_grad():
                fake = genNet(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake,padding = 2,normallize = True))


        iters += 1
elapsed_time = time.time() - inital_time

print("Total elapsed time : {:.1f} hr {:.1f} min {:.2f} sec".format(
                elapsed_time // 3600, elapsed_time//60, elapsed_time%60))


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

### Visualization of G's progression

fig = plt.figure(figsize = (8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated = True)] for i in img_list]
ani = animation.ArtistAnimation(fig,ims,interval = 1000,repeat_delay = 1000,blit = True)

plt.show()


### Let's look at real vs fake images

real_batch = next(iter(dataloader))
plt.figure(figsize = (15,15))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake images")
# we use transpose to permute the indexes as the img is nc x 64 x 64, but to show
# we want 64 x 64 x n thus we changed index 0 to last and 1,2 at first
plt.imshow(np.transpose(img_list[-1],(1,2,0))) 
plt.show()

