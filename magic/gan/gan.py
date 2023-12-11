
import torch.nn as nn
import torch
import os
from magic.utils import BaseClass
from tqdm import tqdm
from  torch.nn import functional as F

class Generator(nn.Module):
  def __init__(self,z_dim=10,im_shape=(1,64,64),hidden_dim=128,batch_size=1):
    super(Generator,self).__init__()
    self.img_shape = im_shape
    self.img_dim = im_shape[0]*im_shape[1]*im_shape[2]
    self.batch_size = batch_size
    self.gen = nn.Sequential(
        self.get_generator_block(z_dim,hidden_dim),
        self.get_generator_block(hidden_dim,hidden_dim*2),
        self.get_generator_block(hidden_dim*2,hidden_dim*4),
        self.get_generator_block(hidden_dim*4,hidden_dim*8),
        nn.Linear(hidden_dim*8,self.img_dim),
        nn.Tanh()
    )  
  def get_generator_block(self,input_dim,output_dim):
    return nn.Sequential(nn.Linear(input_dim,output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))
  
  def forward(self,x):
    x = self.gen(x)
    x = x.view((-1,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
    return x

    
class Discriminator(nn.Module):
  def __init__(self,img_shape=(1,64,64),hidden_dim=128,batch_size=32):
    super(Discriminator,self).__init__()
    self.img_shape = batch_size*img_shape[0]*img_shape[1]*img_shape[2]
    self.disc = nn.Sequential(
        self.get_discriminator_block(self.img_shape,hidden_dim*4),
        self.get_discriminator_block(hidden_dim*4,hidden_dim*2),
        self.get_discriminator_block(hidden_dim*2,hidden_dim),
        nn.Linear(hidden_dim,1)
        )
  def get_discriminator_block(self,input_dim,output_dim):
    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )

  def forward(self,x):
    x = x.view(-1)
    return F.sigmoid(self.disc(x))
  
class GANTrain(BaseClass):
    def __init__(self,laten_dim,img_shape=(1,28,28),batch_size=1):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_shape = img_shape
        self.laten_dim = laten_dim
        self.batch_size = batch_size
        self.generator = nn.DataParallel(Generator(self.laten_dim,
                                   self.img_shape,
                                   batch_size=batch_size).to(self.device))
        self.discriminator = nn.DataParallel(Discriminator(self.img_shape,batch_size=batch_size).to(self.device))
        self.mean_generator_loss = []
        self.mean_discriminator_loss = []
   
    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),
              gen_name="gen.pth",disc_name="disc.pth"):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, 
                                       betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, 
                                       betas=betas)
        adversarial_loss = torch.nn.BCELoss()
        self.mean_discriminator_loss = []
        self.mean_generator_loss = []
        super().create_dir("model_weights")
        for e in range(epoch):
            disc_loss_track = []
            gen_loss_track = []
            for imgs in tqdm(dataloader):
                imgs = imgs.view(-1).to(self.device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                noise = self.get_noise(self.batch_size,self.laten_dim,device=self.device)
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
                           sign="o",gcolor='green',dcolor='red')
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
    
    def sample(self):
       noise = self.get_noise(4,self.laten_dim,device=self.device)
       fake_img = self.generator(noise)
       super().show_tensor_images(fake_img)
    
    
    
