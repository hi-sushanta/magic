import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm 
from magic.utils import BaseClass

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
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
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
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
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class CGANTrain(BaseClass):
    def __init__(self,in_chan,img_dim,num_classes,batch_size):
        super().__init__()
        self.in_chan = in_chan 
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_shape = (in_chan,img_dim,img_dim)
        _genrator_dim, _critc_dim = self.get_input_dimensions(self.img_dim,self.img_shape,self.num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = nn.DataParallel(Generator(input_dim=_genrator_dim,im_chan=self.in_chan).to(self.device))
        self.generator.apply(weights_init)
        self.critic = nn.DataParallel(Critic(im_chan=_critc_dim).to(self.device))
        self.critic.apply(weights_init)
        self.mean_generator_loss = []
        self.mean_critic_loss = []
    
    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),
              gen_name='cgen.pth',crit_name="ccritic.pth"):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, 
                                       betas=betas)
        optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr, 
                                       betas=betas)
        criterion = nn.BCEWithLogitsLoss()
        self.mean_critic_loss = []
        self.mean_generator_loss = []
        super().create_dir("model_weights")
        for e in range(epoch):
            gen_loss_track = []
            crit_loss_track = []
            for img,labels in tqdm(dataloader):
                img = img.to(self.device)
                one_hot_labels = self.get_one_hot_labels(labels.to(self.device), self.num_classes)
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.img_shape[1], self.img_shape[2])

                # Update Discriminator Parameters
                optimizer_C.zero_grad()
                fake_noise = self.get_noise(self.batch_size, self.img_dim, device=self.device)
                noise_and_labels = self.combine_vectors(fake_noise,one_hot_labels)
                fake =  self.generator(noise_and_labels)
                fake_image_and_labels = self.combine_vectors(fake,image_one_hot_labels)
                real_image_and_labels = self.combine_vectors(img, image_one_hot_labels)
                disc_fake_pred = self.critic(fake_image_and_labels.detach())
                disc_real_pred = self.critic(real_image_and_labels)
                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                optimizer_C.step() 
                crit_loss_track.append(disc_loss.item())

                # Update Generator Parameters
                optimizer_G.zero_grad()
                fake_image_and_labels = self.combine_vectors(fake, image_one_hot_labels)
                # This will error if you didn't concatenate your labels to your image correctly
                disc_fake_pred = self.critic(fake_image_and_labels)
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
                optimizer_G.step()
                gen_loss_track.append(gen_loss.item())
            self.mean_generator_loss.append(sum(gen_loss_track)/len(dataloader))
            self.mean_critic_loss.append(sum(crit_loss_track)/len(dataloader))
            super().loss_plot(e+1,self.mean_generator_loss[e],self.mean_critic_loss[e],title="Model Tracking",
                           label_loss="Loss",last_epoch=epoch,
                           sign="o",gcolor='green',dcolor='red')
            torch.save(self.generator.state_dict(),f="model_weights/"+gen_name)
            torch.save(self.critic.state_dict(),f="model_weights/"+crit_name)



    def get_one_hot_labels(self,labels, n_classes):
        '''
        Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
        Parameters:
            labels: tensor of labels from the dataloader, size (?)
            n_classes: the total number of classes in the dataset, an integer scalar
        '''
        return F.one_hot(labels,n_classes)
    
    def combine_vectors(self,x, y):
        '''
        Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
        Parameters:
          x: (n_samples, ?) the first vector. 
            In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
            but you shouldn't need to know the second dimension's size.
          y: (n_samples, ?) the second vector.
            Once again, in this assignment this will be the one-hot class vector 
            with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
        '''
        # Note: Make sure this function outputs a float no matter what inputs it receives
        combined = torch.cat((x.float(),y.float()),1)
        return combined
    
    def get_input_dimensions(self,z_dim, mnist_shape, n_classes):
        '''
        Function for getting the size of the conditional input dimensions 
        from z_dim, the image shape, and number of classes.
        Parameters:
            z_dim: the dimension of the noise vector, a scalar
            mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
            n_classes: the total number of classes in the dataset, an integer scalar
                    (10 for MNIST)
        Returns: 
            generator_input_dim: the input dimensionality of the conditional generator, 
                              which takes the noise and class vectors
            discriminator_im_chan: the number of input channels to the discriminator
                                (e.g. C x 28 x 28 for MNIST)
        '''
        generator_input_dim = z_dim + n_classes
        discriminator_im_chan = mnist_shape[0] + n_classes
        return generator_input_dim, discriminator_im_chan

    
    def get_noise(self,n_samples,z_dim,device='cpu'):
        noise = torch.randn(n_samples,z_dim,device=device)
        return noise
    
    def load_model(self,gpath,dpath):
        gen_state = torch.load(gpath)
        self.generator.load_state_dict(gen_state)
        disc_state = torch.load(dpath)
        self.critic.load_state_dict(disc_state)
        return self.generator,self.critic
    
    def sample(self,labels):
       one_hot_labels = self.get_one_hot_labels(labels.to(self.device), self.num_classes)
       image_one_hot_labels = one_hot_labels[:, :, None, None]
       image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.img_shape[1], self.img_shape[2])

       noise = self.get_noise(4,self.img_dim,device=self.device)
       noise_and_labels = self.combine_vectors(noise,one_hot_labels)

       fake_img = self.generator(noise_and_labels)
       super().show_tensor_images(fake_img)

    




