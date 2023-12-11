import torch
from torch import nn
from tqdm import tqdm 
from magic.utils import BaseClass
from torch.optim import Adam

class Generator(nn.Module):
  def __init__(self,in_chan):
    super(Generator,self).__init__()
    self.e1 = self.define_encoder_block(in_chan,64,batchnorm=False)
    self.e2 = self.define_encoder_block(64,128)
    self.e3 = self.define_encoder_block(128,256)
    self.e4 = self.define_encoder_block(256,512)
    self.e5 = self.define_encoder_block(512,512)
    self.e6 = self.define_encoder_block(512,512)
    self.e7 = self.define_encoder_block(512,512)

    # bottlenech, no batch norm and relu
    self.b = nn.Conv2d(512,512,(4,4),stride=(2,2),padding=1)
    nn.init.normal_(self.b.weight, mean=0.0, std=0.02)
    self.actb = nn.ReLU(inplace=True)

    # Decoder model : CD512-CD512-CD512-C512-C256-C128-C64
    self.d1 = self.define_decoder_block(512,512)
    self.act1 = nn.ReLU(inplace=True)
    self.d2 = self.define_decoder_block(1024,512)
    self.act2 = nn.ReLU(inplace=True)
    self.d3 = self.define_decoder_block(1024,512)
    self.act3 = nn.ReLU(inplace=True)
    self.d4 = self.define_decoder_block(1024,512,dropout=False)
    self.act4 = nn.ReLU(inplace=True)
    self.d5 = self.define_decoder_block(1024,256,dropout=False)
    self.act5 = nn.ReLU(inplace=True)
    self.d6 = self.define_decoder_block(512,128,dropout=False)
    self.act6 = nn.ReLU(inplace=True)
    self.d7 = self.define_decoder_block(256,64,dropout=False)
    self.act7 = nn.ReLU(inplace=True)

    self.d8 = nn.ConvTranspose2d(128,in_chan,(4,4),stride=(2,2),padding=1,bias=False)
    nn.init.normal_(self.d8.weight, mean=0.0,std=0.02)
    self.act8 = nn.Tanh()

  def forward(self,x):
    xe1 = self.e1(x)
    xe2 = self.e2(xe1)
    xe3 = self.e3(xe2)
    xe4 = self.e4(xe3)
    xe5 = self.e5(xe4)
    xe6 = self.e6(xe5)
    xe7 = self.e7(xe6)
    b1 = self.actb(self.b(xe7))

    xd8 = self.act1(torch.cat((self.d1(b1),xe7),dim=1))
    xd9 = self.act2(torch.cat((self.d2(xd8),xe6),dim=1))
    xd10 = self.act3(torch.cat((self.d3(xd9),xe5),dim=1))
    xd11 = self.act4(torch.cat((self.d4(xd10),xe4),dim=1))
    xd12 = self.act5(torch.cat((self.d5(xd11),xe3),dim=1))
    xd13 = self.act6(torch.cat((self.d6(xd12),xe2),dim=1))
    xd14 = self.act7(torch.cat((self.d7(xd13),xe1),dim=1))

    xd15 = self.d8(xd14)
    out_image = self.act8(xd15)
    return out_image




  def define_encoder_block(self,in_chan, n_filters, batchnorm=True):
    # Create a list to store the layers of the encoder block
    layers = []

    # Add the convolutional layer with the specified number of filters
    conv_layer = nn.Conv2d(in_chan, n_filters, kernel_size=4, stride=2, padding=1, bias=False)
    nn.init.normal_(conv_layer.weight, mean=0.0, std=0.02)
    layers.append(conv_layer)

    # Conditionally add batch normalization
    if batchnorm:
        layers.append(nn.BatchNorm2d(n_filters))

    # Add the LeakyReLU activation
    layers.append(nn.LeakyReLU(0.2))

    # Create a sequential block using the list of layers
    encoder_block = nn.Sequential(*layers)

    return encoder_block

  def define_decoder_block(self,in_chan,out_chan,dropout=True):
    layers = []
    g = nn.ConvTranspose2d(in_chan,out_chan,(4,4),stride=(2,2),padding=1,bias=False)
    nn.init.normal_(g.weight, mean=0.0,std=0.02)
    layers.append(g)
    g = nn.BatchNorm2d(out_chan)
    layers.append(g)
    if dropout:
      g = nn.Dropout(0.5)
      layers.append(g)
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
  def __init__(self,in_chan):
    super(Discriminator,self).__init__()
    # Weight initialization
    self.conv1 =  nn.Conv2d(in_chan,64,(4,4),stride=(2,2),padding=1,bias=False)
    self.act1 = nn.LeakyReLU(negative_slope=0.2)

    self.conv2 = nn.Conv2d(64,128,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm1 = nn.BatchNorm2d(128)
    self.act2 = nn.LeakyReLU(negative_slope=0.2)

    self.conv3 = nn.Conv2d(128,256,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm2 = nn.BatchNorm2d(256)
    self.act3 = nn.LeakyReLU(negative_slope=0.2)

    self.conv4 = nn.Conv2d(256,512,(4,4),stride=(2,2),padding=1,bias=False)
    self.bnorm3 = nn.BatchNorm2d(512)
    self.act4 = nn.LeakyReLU(negative_slope=0.2)

    self.conv5 = nn.Conv2d(512,512,(4,4),padding=1,bias=False)
    self.bnorm4 = nn.BatchNorm2d(512)
    self.act5 = nn.LeakyReLU(negative_slope=0.2)

    self.conv6 = nn.Conv2d(512,1,(4,4),padding=1,bias=False)
    self.patch_out = nn.Sigmoid()

    # weight initializer all conv2d layer
    self._initialize_weights()
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

  def forward(self,s_img, t_img):

    # Concatenate source image and target image
    m_img = torch.cat((s_img,t_img),dim=1)
    # C64: 4x4 kernel stride 2x2
    x = self.act1(self.conv1(m_img))
    # C128: 4x4 kernel stride 2x2
    x = self.act2(self.bnorm1(self.conv2(x)))
    # C256: 4x4 kernel stride 2x2
    x = self.act3(self.bnorm2(self.conv3(x)))
    # C512: 4x4 kernel stride 2x2
    x = self.act4(self.bnorm3(self.conv4(x)))
    # C512: 4x4 kernel stride 2x2
    x = self.act5(self.bnorm4(self.conv5(x)))
    # Patch Output
    x = self.patch_out(self.conv6(x))
    return x



class Pix2Pix:
    def __init__(self,in_chan):
        super().__init__()
        self.base = BaseClass()
        self.in_chan = in_chan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = nn.DataParallel(Generator(in_chan).to(self.device))
        self.disc = nn.DataParallel(Discriminator(in_chan+in_chan).to(self.device))
        self.mean_generator_loss = []
        self.mean_discriminator_loss = []
    
    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),
              gen_name='pix2pix.pth',crit_name="pix_disc.pth"):
        optimizer_G = Adam(self.generator.parameters(),lr=lr,betas=betas)
        optimizer_D = Adam(self.disc.parameters(),lr=lr,betas=betas)
        bc_loss = nn.BCELoss()
        self.mean_discriminator_loss = []
        self.mean_generator_loss = []
        self.base.create_dir("model_weights")
        for e in range(epoch):
           gen_loss_track = []
           disc_loss_track = []
           for src_img, tar_img in tqdm(dataloader):
              src_img = src_img.to(self.device)
              tar_img = tar_img.to(self.device)

              # Update Discriminator model parameters
              optimizer_D.zero_grad()
              real_pred = self.disc(src_img,tar_img)
              rb_loss = bc_loss(real_pred,torch.ones_like(real_pred))
              
              fake_img = self.generator(src_img)
              fake_pred = self.disc(src_img,fake_img.detach())
              fb_loss = bc_loss(fake_pred,torch.zeros_like(fake_pred))
              d_loss = rb_loss + fb_loss
              d_loss.backward()
              optimizer_D.step()
              disc_loss_track.append(d_loss.item())

              # Update Generator model parameters
              optimizer_G.zero_grad()
              fake_pred2 = self.disc(src_img,fake_img)
              g_loss = bc_loss(fake_pred2, torch.ones_like(fake_pred2))
              g_loss.backward()
              optimizer_G.step()
              gen_loss_track.append(g_loss.item())
        
           self.mean_generator_loss.append(sum(gen_loss_track)/len(dataloader))
           self.mean_discriminator_loss.append(sum(disc_loss_track)/len(dataloader))
           self.base.loss_plot(e+1,self.mean_generator_loss[e],self.mean_discriminator_loss[e],title="Model Tracking",
                           label_loss="Loss",last_epoch=epoch,
                           sign="o",gcolor='green',dcolor='red')
           torch.save(self.generator.state_dict(),f="model_weights/"+gen_name)
           torch.save(self.disc.state_dict(),f="model_weights/"+crit_name)
    
    def load_model(self,gpath,dpath):
        gen_state = torch.load(gpath)
        self.generator.load_state_dict(gen_state)
        disc_state = torch.load(dpath)
        self.disc.load_state_dict(disc_state)
        return self.generator,self.disc
    
    def sample(self,src_img):
       fake_img = self.generator(src_img)
       super().show_tensor_images(fake_img,num_images=1,nrows=1)


      
      
  

