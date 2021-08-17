import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

### Designing Architecture of Discriminator and Generator  

## Discriminator ## 
class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(img_dim,256),
                nn.LeakyReLU(0.01),
                nn.Linear(256,128),
                nn.LeakyReLU(0.01),
                nn.Linear(128,1),
                nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x)


## Generator ##
class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(z_dim,128),
                nn.LeakyReLU(0.01),
                nn.Linear(128,256),
                nn.LeakyReLU(0.01),
                nn.Linear(256,img_dim),
                nn.Tanh(), # normalizing inputs to [-1,1] so to make ouptuts in [-1,1]
        )
    def forward(self,x):
        return self.main(x)


## Hyperparameters ## 

img_dim = 28 * 28 * 1
z_dim = 64
lr = 3e-4 # andrej karpathy said it's the best
num_epochs =  50
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator(z_dim,img_dim)
disc = Discriminator(img_dim)

## Data and optimizer and other stuff ## 

fixed_noise = torch.randn((batch_size,z_dim)).to(device)
transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

dataset = datasets.MNIST(root = 'data', transform = transforms, download = True)
dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = True)

optim_gen = optim.Adam(gen.parameters(),lr = lr)
optim_disc = optim.Adam(disc.parameters(),lr = lr)

writer_fake = SummaryWriter(log_dir = 'logs/fake')
writer_real = SummaryWriter(log_dir = 'logs/real')

step = 0

criterion = nn.BCELoss() # ylog(pred) + (1-y)log(pred)


## Training ## 

start_time = time.time()
for epoch in range(num_epochs):

    epoch_time = time.time()
    for batch_idx,(real,_) in enumerate(dataloader):

        batch_time = time.time()

        real = real.view(-1,784).to(device)
        batch_size = real.shape[0] # 32
        # print(batch_size)

        ### Training the discriminator ### 
        # goal is to maximize log(D(real)) + log(1-D(fake)) where real = x, fake = G(z)

        noise = torch.randn((batch_size,z_dim)).to(device)
        fake = gen(noise) # we need this for generator part so we will retain graph 

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real,torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph = True)
        optim_disc.step()


        ### Training the generator ###
        # goal is to minimize log(1-D(G(z))) which means maximizing log(D(G(z)))
        # Latter one is better for bettter gradient descent and updation as earlier 
        # one saturates early in training 

        decision = disc(fake).view(-1) # decision of discriminator for generated image
        lossG = criterion(decision, torch.ones_like(decision))

        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        ## After a batch ends, print stats ##
        ## And write to tensor board ## 
 

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch Length : {len(dataloader)}  \
                    Loss D : {lossD:.4f}, Loss G : {lossG:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize = True)
                img_grid_real = torchvision.utils.make_grid(data,normalize = True)

                writer_fake.add_image(
                        "MNIST Fake generated images ", img_grid_fake,global_step = step)
                writer_real.add_image(
                        "MNIST Real data images ", img_grid_real, global_step = step)

                step+=1
    elapsed_time = time.time() - epoch_time
    print(f'Time per epoch : {elapsed_time:.2f} sec')


end_time = time.time() - start_time

print(f"Training Done!, Elapsed time : {(end_time//60):.1f} mins {end_time%60 :.2f} seconds")
