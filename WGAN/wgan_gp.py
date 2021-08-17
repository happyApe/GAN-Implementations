import time
import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# HyperParameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
batch_size = 64 
image_size = 64
channels_img = 3
z_dims = 128
num_epochs = 5
critic_features = 64
gen_features = 64
critic_iterations = 5
lambda_gp = 10


## Gradient Penalty ##

def gradient_penalty(critic,real,fake,device):

    bs,c,h,w = real.shape
    epsilon = torch.rand((bs,1,1,1)).repeat(1,c,h,w).to(device)
    interpolated_images = real * epsilon  + fake * (1-epsilon)

    # calculate the critic scores
    mixed_scores = critic(interpolated_images)

    # Take gradient of scores with respect to images
    gradient = torch.autograd.grad(
            inputs = interpolated_images,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph = True,
            retain_graph = True)[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penalty = torch.mean((gradient_norm -1)**2)

    return gradient_penalty

def save_checkpoint(state,filename = 'celeba_wgan_gp.pth.tar'):
    print("---> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,gen,critic):
    print("---> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic'])




# Architecture of DCGAN modified for WGAN
class Critic(nn.Module):
    def __init__(self,channels_img,critic_features):
        super(Critic,self).__init__()

        self.main = nn.Sequential(

                # input : channels_img X 64 X 64 
                nn.Conv2d(channels_img,critic_features, 4, 2, 1,bias = False), 
                nn.LeakyReLU(0.2),

                # input : critic_features X 32 X 32
                nn.Conv2d(critic_features,critic_features * 2, 4, 2, 1,bias = False),
                nn.InstanceNorm2d(critic_features * 2,affine = True),
                nn.LeakyReLU(0.2),

                # input : critic_features*2 X 16 X 16
                nn.Conv2d(critic_features*2,critic_features * 4, 4, 2, 1,bias = False),
                nn.InstanceNorm2d(critic_features * 4,affine = True),
                nn.LeakyReLU(0.2),

                # input : critic_features*4 X 8 X 8
                nn.Conv2d(critic_features*4,critic_features * 8, 4, 2, 1,bias = False),
                nn.InstanceNorm2d(critic_features * 8,affine = True),
                nn.LeakyReLU(0.2),

                # state size critic_features*8 x 4 x 4
                # convolving to 1x1  
                nn.Conv2d(critic_features*8,1,4,2,0,bias = False)
        )

    def forward(self,x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self,z_dims,channels_img,gen_features):
        super(Generator,self).__init__()

        self.main = nn.Sequential(

                # input is a vector of size z_dims x 1 x 1
                nn.ConvTranspose2d(z_dims,gen_features * 16,4,1,0),
                nn.BatchNorm2d(gen_features * 16),
                nn.ReLU(),

                # input : gen_features * 16 x 4 x 4
                nn.ConvTranspose2d(gen_features*16,gen_features * 8,4,2,1),
                nn.BatchNorm2d(gen_features * 8),
                nn.ReLU(),

                # input : gen_features  * 8 x 8 x 8 
                nn.ConvTranspose2d(gen_features*8,gen_features * 4,4,2,1),
                nn.BatchNorm2d(gen_features * 4),
                nn.ReLU(),

                # input : gen_features  * 4 x 16 x 16
                nn.ConvTranspose2d(gen_features*4,gen_features * 2,4,2,1),
                nn.BatchNorm2d(gen_features * 2),
                nn.ReLU(),

                # input : gen_features  * 4 x 16 x 16
                nn.ConvTranspose2d(gen_features*2,channels_img,4,2,1),
                nn.Tanh()
        )

    def forward(self,x):
        return self.main(x)

# Acc to DCGAN Paper
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
            


dataroot = 'data'
dataset = datasets.ImageFolder(root = dataroot,transform = 
                                transforms.Compose([transforms.Resize(image_size),
                                                    transforms.CenterCrop(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,),(0.5,))]))

dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers =  2)


# initializing the adversaries

gen = Generator(z_dims,channels_img,gen_features).to(device)
critic = Critic(channels_img,critic_features).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializing the optimizers

optim_gen = optim.Adam(gen.parameters(),lr = lr, betas = (0.0,0.9))
optim_critic = optim.Adam(critic.parameters(),lr = lr, betas = (0.0,0.9))

# tensorboard
fixed_noise = torch.randn(64,z_dims,1,1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')

step = 0

gen.train()
critic.train()



## Training time 

start_time = time.time()
for epoch in range(num_epochs):

    epoch_time = time.time()
    for batch_idx,(data,_) in enumerate(dataloader):

        batch_time = time.time()

        data = data.to(device)
        bs = data.shape[0]


        # Training critic 
        # max E[critic(real)] - E[critic(fake)]


        for _ in range(critic_iterations):
            noise = torch.randn(bs,z_dims,1,1).to(device)
            fake = gen(noise)

            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic,data,fake,device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            critic.zero_grad()

            loss_critic.backward(retain_graph=True)
            optim_critic.step()

        # Training generator
        # max E [critic(gen(fake))] <-> min - E [critic(gen(fake))] 
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        ## After a batch ends, print stats ##
        ## And write to tensor board ## 


        if batch_idx % 100 == 0 and batch_idx > 0: 
            gen.eval()
            critic.eval()
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                    Loss C : {loss_critic:.4f}, Loss G : {loss_gen:.4f}")


            with torch.no_grad():
                fake = gen(noise)
                # taking upto 32 examples
                img_grid_real = torchvision.utils.make_grid(data[:32],normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize = True)

                writer_real.add_image("Real", img_grid_real,global_step = step)
                writer_fake.add_image("Fake", img_grid_fake,global_step = step)

            step += 1
            gen.train()
            critic.train()
 
    elapsed_time = time.time() - epoch_time
    print(f'Time per epoch : {elapsed_time:.2f} sec')


end_time = time.time() - start_time
print(f"Training Done!, Elapsed time : {(end_time//60):.1f} mins {end_time%60 :.2f} seconds")
  

