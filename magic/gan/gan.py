
import torch.nn as nn
import torch
import os
from magic.utils import BaseClass
from tqdm import tqdm
import numpy as np

class Generator(nn.Module):
    def __init__(self,laten_dim=10,im_dim=784):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(laten_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024,im_dim),
            nn.Tanh()
        )
    

    def forward(self, z):
        img = self.model(z)
        return img
    
class Discriminator(nn.Module):
    def __init__(self,img_dim=768,out_dim=1):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, out_dim),
            nn.Sigmoid(),
            )
    def forward(self,latent):
         validity = self.model(latent)
         return validity
  
class GANTrain(BaseClass):
    def __init__(self,laten_dim,img_shape=(1,28,28),batch_size=1):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_shape = img_shape
        self.laten_dim = laten_dim
        self.out_latendim = batch_size*img_shape[0]*img_shape[1]*img_shape[2]
        self.generator = Generator(self.laten_dim,
                                   self.out_latendim).to(self.device)
        self.discriminator = Discriminator(self.out_latendim,1).to(self.device)
        self.mean_generator_loss = []
        self.mean_discriminator_loss = []
    def create_dir(self,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        else:
            print(f"Folder already exists: {folder_name}")

    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999)):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, 
                                       betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, 
                                       betas=betas)
        adversarial_loss = torch.nn.BCELoss()
        self.mean_discriminator_loss = []
        self.mean_generator_loss = []
        self.create_dir("model")
        for e in range(epoch):
            disc_loss_track = []
            gen_loss_track = []
            for imgs in tqdm(dataloader):
                cur_batch_size = len(imgs.view(-1))
                imgs = imgs.view(-1).to(self.device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                noise = self.get_noise(cur_batch_size,self.laten_dim,device=self.device)
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
            self.loss_plot(e+1,self.mean_discriminator_loss[e],title="Discriminator Model Tracking",
                           label_loss="Loss",last_epoch=epoch,
                           sign="o")
            self.loss_plot(e+1,self.mean_generator_loss[e],title="Generator Model Tracking",
                           label_loss="Loss",last_epoch=epoch,sign="o")
            torch.save(self.generator.state_dict(),f="model_weights/generator.pth")
            torch.save(self.discriminator.state_dict(),f="model_weights/discriminator.pth")
        
    def get_noise(self,n_samples,z_dim,device='cpu'):
        noise = torch.randn(n_samples,z_dim,device=device)
        return noise
    
    def load_model(self,gpath,dpath):
        gen_state = torch.load(gpath)
        self.generator.load_state_dict(gen_state)
        disc_state = torch.load(dpath)
        self.discriminator.load_state_dict(disc_state)
        return self.generator,self.discriminator
    





