import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from magic.utils import BaseClass
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self,out_chan,img_dim):
        super(Generator, self).__init__()
        
        self.init_size = img_dim//4
        self.latent_dim = img_dim
        self.out_chan = out_chan
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_chan, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2,inplace=True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.Sigmoid()
            )

    '''
    Function for completing a forward pass of the discriminator: Given an image tensor, 
    returns a 1-dimension tensor representing fake/real.
    Parameters:
        image: a flattened image tensor with dimension (im_dim)
    '''
    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

class DCGANTrain(BaseClass):
    def __init__(self,in_chan,img_dim):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_chan = in_chan
        self.img_dim = img_dim
        self.generator = nn.DataParallel(Generator(self.in_chan,
                                   self.img_dim).to(self.device))
        self.discriminator = nn.DataParallel(Discriminator(self.in_chan,self.img_dim).to(self.device))
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.mean_generator_loss = []
        self.mean_discriminator_loss = []
    def create_dir(self,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        else:
            print(f"Folder already exists: {folder_name}")

    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),
              gen_name='gen.pth',disc_name="disc.pth"):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, 
                                       betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, 
                                       betas=betas)
        adversarial_loss = torch.nn.BCELoss()
        self.mean_discriminator_loss = []
        self.mean_generator_loss = []
        self.create_dir("model_weights")
        for e in range(epoch):
            disc_loss_track = []
            gen_loss_track = []
            for imgs in tqdm(dataloader):
                cur_batch_size = imgs.shape[0]
                imgs = imgs.to(self.device)
                # Train Discriminator
                optimizer_D.zero_grad()
                noise = self.get_noise(cur_batch_size,self.img_dim,device=self.device)
                fake_noise = self.generator(noise)
                fake_pred = self.discriminator(fake_noise.detach())
                fake_disc_loss = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
                real_pred = self.discriminator(imgs)
                real_disc_loss = adversarial_loss(real_pred,torch.ones_like(real_pred))
                disc_loss = (fake_disc_loss + real_disc_loss)/2
                disc_loss_track.append(disc_loss.item())
                # Updata the gradients
                disc_loss.backward(retain_graph=True)
                # Updata optimizer
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad() # Clear gradient value
                fake_s = self.generator(noise) # Generator create fake example
                fake_p = self.discriminator(fake_s) # Discriminator Predict on fake or real

                gen_loss = adversarial_loss(fake_p,torch.ones_like(fake_p)) # Calculate Generator loss
                gen_loss_track.append(gen_loss.item())
                gen_loss.backward(retain_graph=True)
                optimizer_G.step()
            self.mean_discriminator_loss.append(sum(disc_loss_track)/len(dataloader))
            self.mean_generator_loss.append(sum(gen_loss_track)/len(dataloader))
            super().loss_plot(e+1,self.mean_generator_loss[e],self.mean_discriminator_loss[e],title="Model Tracking",
                           label_loss="Loss",last_epoch=epoch,
                           sign="o",gcolor="green",dcolor='red')
            torch.save(self.generator.state_dict(),f="model_weights/"+gen_name)
            torch.save(self.discriminator.state_dict(),f="model_weights/"+disc_name)
    def get_noise(self,n_samples,z_dim,device='cpu'):
        noise = torch.randn(n_samples,z_dim,device=device)
        return noise
    
    def load_model(self,gpath,dpath):
        gen_state = torch.load(gpath)
        self.generator.load_state_dict(gen_state)
        disc_state = torch.load(dpath)
        self.discriminator.load_state_dict(disc_state)
        return self.generator,self.discriminator
    




