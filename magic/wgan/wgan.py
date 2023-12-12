import torch
from torch import nn
from tqdm import tqdm 
from magic.utils import BaseClass

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
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
        x = noise.view(len(noise), self.z_dim, 1, 1)
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


class WGanTrain(BaseClass):
    def __init__(self,in_chan,img_dim,batch_size=32):
        super().__init__()
        self.in_chan = in_chan
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = nn.DataParallel(Generator(self.img_dim,self.in_chan,
                                   self.img_dim).to(self.device))
        self.generator.apply(weights_init)
        self.critic = nn.DataParallel(Critic(self.in_chan,self.img_dim).to(self.device))
        self.critic.apply(weights_init)

        self.mean_generator_loss = []
        self.mean_critic_loss = []

    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),crit_repeat=6,
              c_lambda=10,gen_name='wgen.pth',crit_name="wcritic.pth"):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, 
                                       betas=betas)
        optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr, 
                                       betas=betas)
        self.mean_critic_loss = []
        self.mean_generator_loss = []
        super().create_dir("model_weights")
        for e in range(epoch):
            crit_loss_track = []
            gen_loss_track = []
            for img in tqdm(dataloader):
                img = img.to(self.device)
                mean_iterate_critic_loss = 0
                for _ in range(crit_repeat):
                    # Update Critic Parameters
                    optimizer_C.zero_grad()
                    fake_noise = self.get_noise(self.batch_size,self.img_dim,device=self.device)
                    fake_img = self.generator(fake_noise)
                    crit_fake_pred = self.critic(fake_img.detach())
                    crit_real_pred = self.critic(img)
                    epsilon = torch.rand(len(img), 1,1,1, device=self.device, requires_grad=True)
                    gradient = self.get_gradient(self.critic, img, fake_img.detach(), epsilon)
                    gp = self.gradient_penalty(gradient)
                    crit_loss = self.get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                    # Keep track of the average critic loss in this batch
                    mean_iterate_critic_loss += crit_loss.item() / crit_repeat
                    # Update gradients
                    crit_loss.backward(retain_graph=True)
                    # Update optimizer
                    optimizer_C.step()
                
                crit_loss_track.append(mean_iterate_critic_loss)

                # Update Generator
                optimizer_G.zero_grad()
                fake_noise = self.get_noise(self.batch_size,self.img_dim,self.device)
                fake_img = self.generator(fake_noise)
                crit_fake_pred = self.critic(fake_img)
                gen_loss = self.get_gen_loss(crit_fake_pred)
                gen_loss.backward()
                optimizer_G.step()
                gen_loss_track.append(gen_loss.item())
            
            self.mean_generator_loss.append(sum(gen_loss_track)/len(dataloader))
            self.mean_critic_loss.append(sum(crit_loss_track)/len(dataloader))
            super().print_loss(e+1,self.mean_generator_loss[e],self.mean_critic_loss[e],dlabel="Critic")

            torch.save(self.generator.state_dict(),f="model_weights/"+gen_name)
            torch.save(self.critic.state_dict(),f="model_weights/"+crit_name)
        
    def load_model(self,gpath,dpath):
        gen_state = torch.load(gpath)
        self.generator.load_state_dict(gen_state)
        disc_state = torch.load(dpath)
        self.critic.load_state_dict(disc_state)
        return self.generator,self.critic
    
    
    def get_crit_loss(self,crit_fake_pred, crit_real_pred, gp, c_lambda):
        '''
        Return the loss of a critic given the critic's scores for fake and real images,
        the gradient penalty, and gradient penalty weight.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
            crit_real_pred: the critic's scores of the real images
            gp: the unweighted gradient penalty
            c_lambda: the current weight of the gradient penalty 
        Returns:
            crit_loss: a scalar for the critic's loss, accounting for the relevant factors
        '''
        crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
        return crit_loss
    
    def get_gen_loss(self,crit_fake_pred):
        '''
        Return the loss of a generator given the critic's scores of the generator's fake images.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
        Returns:
            gen_loss: a scalar loss value for the current batch of the generator
        '''
        gen_loss = -1. * torch.mean(crit_fake_pred)
        return gen_loss
    
    def gradient_penalty(self,gradient):
        '''
        Return the gradient penalty, given a gradient.
        Given a batch of image gradients, you calculate the magnitude of each image's gradient
        and penalize the mean quadratic distance of each magnitude to 1.
        Parameters:
            gradient: the gradient of the critic's scores, with respect to the mixed image
        Returns:
            penalty: the gradient penalty
        '''
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)

        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)

        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty
    
    def get_gradient(self,crit, real, fake, epsilon):
        '''
        Return the gradient of the critic's scores with respect to mixes of real and fake images.
        Parameters:
            crit: the critic model
            real: a batch of real images
            fake: a batch of fake images
            epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        Returns:
            gradient: the gradient of the critic's scores, with respect to the mixed image
        '''
        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = crit(mixed_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores), 
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient
    
    def get_noise(self,n_samples,z_dim,device='cpu'):
        noise = torch.randn(n_samples,z_dim,device=device)
        return noise
    
    def sample(self):
       noise = self.get_noise(4,self.img_dim,device=self.device)
       fake_img = self.generator(noise)
       super().show_tensor_images(fake_img)
