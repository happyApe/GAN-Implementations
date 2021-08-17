import argparse
import os
import time
import math
import numpy as np
import itertools
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import * 
from datasets import  * 
from utils import * 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparams and arguments

start_epoch = 0 # starting epoch
num_epochs = 200
dataroot = 'data/monet2photo/'
batch_size = 1
# Adam
lr = 0.0002
b1 = 0.5 
b2 = 0.999 
decay_epoch = 100 # from which epoch to start lr decay
img_height = 256
img_width = 256
channels = 3
sample_interval = 100 # interval between saving output
checkpoint_interval = -1 # interval between saving model checkpoint 
num_res_blocks = 9 # res blocks in generator
lambda_cyc = 10.0 # cycle loss weight
lambda_id = 5.0 # identity loss weight


# Creating sample and checkpoint dirs

dataset_name =  'monet2photo'
os.makedirs("images/%s" % dataset_name, exist_ok = True)
os.makedirs('saved_models/%s' % dataset_name,exist_ok = True)

# Loss function

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


cuda = torch.cuda.is_available()

img_dims = (channels,img_height,img_width)

# model init

# two generators, two discriminators
# G : A -> B and  G : B -> A
# D : A      and  D : B
G_AB =  GeneratorResNet(img_dims,num_res_blocks)
G_BA = GeneratorResNet(img_dims,num_res_blocks)
D_A = Discriminator(img_dims)
D_B = Discriminator(img_dims)

if cuda :
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if start_epoch != 0 :
    # Load pretrained models 
    G_AB.load_state_dict(torch.load('saved_models/%s/%G_AB_%d.pth' %(dataset_name,start_epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/%G_BA_%d.pth' %(dataset_name,start_epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/%D_A_%d.pth' %(dataset_name,start_epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/%D_B_%d.pth' %(dataset_name,start_epoch)))

else : 
    # weights initialization
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

# Optimizers 

optim_G = torch.optim.Adam(itertools.chain(G_AB.parameters(),G_BA.parameters()),
        lr = lr,betas = (b1,b2))

optim_D_A = torch.optim.Adam(D_A.parameters(),lr,betas = (b1,b2))
optim_D_B = torch.optim.Adam(D_B.parameters(),lr,betas = (b1,b2))

# Learning rate scheduler

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G,
                lr_lambda = LambdaLR(num_epochs,start_epoch,decay_epoch).step)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_D_A,
                lr_lambda = LambdaLR(num_epochs,start_epoch,decay_epoch).step)

lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_D_B,
                lr_lambda = LambdaLR(num_epochs,start_epoch,decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# Buffers of previous generated samples 

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


transforms_ = [transforms.Resize(int(img_height* 1.12),Image.BICUBIC),
        transforms.RandomCrop((img_height,img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5 ,0.5,0.5),(0.5,0.5,0.5))
        ]

dataloader = DataLoader(
        ImageDataset(dataroot,transforms_ = transforms_,unaligned = True),
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
        )

val_dataloader = DataLoader(
        ImageDataset(dataroot,transforms_ = transforms_,unaligned = True,mode = 'test'),
        batch_size = 5,
        shuffle = True,
        num_workers = 1
        )

def sample_images(batches_done):
    'saving generated images from test batch'
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)

    real_A = make_grid(real_A,5,normalize = True)
    real_B = make_grid(real_B,5,normalize = True)
    fake_A = make_grid(fake_A,5,normalize = True)
    fake_B = make_grid(fake_B,5,normalize = True)

    image_grid = torch.cat((real_A,fake_B,real_B,fake_A),1)
    save_image(image_grid,'images/%s/%s.png' % (dataset_name,batches_done),normalize = False)




##################
#### Training ####
##################



start_time = time.time()

for epoch in range(start_epoch,num_epochs):

    for i,batch in enumerate(dataloader):

        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0),*D_A.output_shape))),requires_grad = False)
        fake = Variable(Tensor(np.zeros((real_A.size(0),*D_A.output_shape))),requires_grad = False)


        #==> Generator Training

        G_AB.train()
        G_BA.train()

        optim_G.zero_grad()

        # 3 Losses : Identity Loss, cycle Loss, GAN Loss

        # Identity Loss

        loss_id_A = criterion_identity(G_BA(real_A),real_A)
        loss_id_B = criterion_identity(G_AB(real_B),real_B)

        loss_id = (loss_id_A + loss_id_B) / 2

        # GAN Loss

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B),valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A),valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2

        # Cycle Loss

        reconstructed_A = G_BA(fake_B)
        # L1 loss
        Loss_cycle_A = criterion_cycle(reconstructed_A,real_A)
        reconstructed_B = G_AB(fake_A)
        Loss_cycle_B = criterion_cycle(reconstructed_B,real_B)

        loss_cycle = (Loss_cycle_A + Loss_cycle_B)/2

        ## Total Loss

        loss_G = loss_GAN + lambda_id * loss_id + lambda_cyc * loss_cycle

        loss_G.backward()
        optim_G.step()



        #==> Discriminator Training

        #### D_A

        optim_D_A.zero_grad()
        
        # real loss
        loss_real = criterion_GAN(D_A(real_A),valid)

        # fake loss from previous generatoed samples
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()),fake)

        # Total Loss
        loss_D_A  = (loss_real + loss_fake)/2
        loss_D_A.backward()
        optim_D_A.step()


        #### D_B

        optim_D_B.zero_grad()

        # real_loss 
        loss_real = criterion_GAN(D_B(real_B),valid)

        # fake loss
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()),fake)

        # Total Loss
        loss_D_B = (loss_real + loss_fake)/2
        loss_D_B.backward()
        optim_D_B.step()

        
        ## Loss of both Discriminators
        loss_D = (loss_D_A + loss_D_B) / 2


        ### Stats : 

        batches_done = epoch * len(dataloader) + i
        batches_left = num_epochs * len(dataloader) - batches_done

        # approx time left
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - start_time))
        start_time = time.time()

        ## Printing stats

        print(f'''[Epoch {epoch}/{num_epochs}] [Batch {i} /{len(dataloader)}] [Loss D : {loss_D.item():.1f}] 
            [Loss G : {loss_G.item():.1f} Adv : {loss_GAN.item():.1f} Cycle : {loss_cycle.item():1f}
            identity : {loss_id.item():1f}] ETA : {time_left}''')

        if batches_done % sample_interval == 0:
            sample_images(batches_done)


    ## Update Learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(),'saved_models/%s/G_AB_%d.pth'%(dataset_name,epoch))
        torch.save(G_BA.state_dict(),'saved_models/%s/G_BA_%d.pth'%(dataset_name,epoch))
        torch.save(D_A.state_dict(),'saved_models/%s/D_A_%d.pth'%(dataset_name,epoch))
        torch.save(D_B.state_dict(),'saved_models/%s/D_B_%d.pth'%(dataset_name,epoch))
    

                
                



        
        
