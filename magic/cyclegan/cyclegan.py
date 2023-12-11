import torch
from torch import nn
from tqdm import tqdm 
from magic.utils import BaseClass
from torch.optim import Adam


class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class:
    Performs two convolutions and an instance normalization, the input is added
    to this output to form the residual block output.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ResidualBlock:
        Given an image tensor, completes a residual block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x
    
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to upsample,
        with an optional instance norm
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator -
    maps each the output to the desired number of output channels
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, input_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator:
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)
    

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn

class CycleGan(BaseClass):
    def __init__(self,input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generatorAB = nn.DataParallel(Generator(input_channels).to(self.device))
        self.generatorAB = self.generatorAB.apply(self.weights_init)
        self.generatorBA = nn.DataParallel(Generator(input_channels).to(self.device))
        self.generatorBA = self.generatorBA.apply(self.weights_init)
        self.discA = nn.DataParallel(Discriminator(input_channels).to(self.device))
        self.discA = self.discA.apply(self.weights_init)
        self.discB = nn.DataParallel(Discriminator(input_channels).to(self.device))
        self.discB.apply(self.weights_init)

        self.generator_loss = []
        self.discriminator_loss = []
    
    def train(self,dataloader,epoch=100,lr=0.0001,betas=(0.9,0.999),
              gen_name='cycleGAN.pth',crit_name="cycleDISC.pth"):
        optimizer_G = Adam(list(self.generatorAB.parameters())+list(self.generatorBA.parameters()),
                           lr=lr,betas=betas)
        optimizer_DA = Adam(self.discA.parameters(),lr=lr,betas=betas)
        optimizer_DB = Adam(self.discB.parameters(),lr=lr,betas=betas)
        self.generator_loss = []
        self.discriminator_loss = []
        adv_criterion = nn.MSELoss()
        recon_criterion = nn.L1Loss()
        for e in range(epoch):
            gen_loss_track = []
            disc_loss_track = []
            for realA, realB in tqdm(dataloader):
                realA = realA.to(self.device)
                realB = realB.to(self.device)

                # Update Discriminator A Model Parameters
                optimizer_DA.zero_grad()
                with torch.no_grad():
                    fake_A = self.generatorBA(realB)
                disc_A_loss = self.get_disc_loss(realA, fake_A, self.discA, adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                optimizer_DA.step() # Update optimizer
                disc_loss_track.append(disc_A_loss.item())
                # Update Discriminator B Model Parameters
                optimizer_DB.zero_grad()
                with torch.no_grad():
                    fake_B = self.generatorAB(realA)
                disc_B_loss = self.get_disc_loss(realB,fake_B,self.discB,adv_criterion)
                disc_B_loss.backward(retain_graph=True)
                optimizer_DB.step()

                # Update Generator Model Parameters
                optimizer_G.zero_grad()
                gen_loss ,fake_A,fake_B = self.get_gen_loss(
                    realA,realB,self.generatorAB,self.generatorBA,
                    self.discA,self.discB,adv_criterion,
                    recon_criterion,recon_criterion
                )
                gen_loss.backward()
                optimizer_G.step()
                gen_loss_track.append(gen_loss.item())
            
            self.generator_loss.append(sum(gen_loss_track)/len(dataloader))
            self.discriminator_loss.append(sum(disc_loss_track)/len(dataloader))
            super().loss_plot(e+1,self.generator_loss[e],self.discriminator_loss[e],title="Model Tracking",
                           label_loss="Loss",last_epoch=epoch,
                           sign="o",gcolor='green',dcolor='red')
            torch.save(self.generatorAB.state_dict(),f="model_weights/"+gen_name)
            torch.save(self.discA.state_dict(),f="model_weights/"+crit_name)




    def get_disc_loss(self,real_X, fake_X, disc_X, adv_criterion):
        '''
        Return the loss of the discriminator given inputs.
        Parameters:
            real_X: the real images from pile X
            fake_X: the generated images of class X
            disc_X: the discriminator for class X; takes images and returns real/fake class X
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator
                predictions and the target labels and returns a adversarial
                loss (which you aim to minimize)
        '''
        disc_fake_X_hat = disc_X(fake_X.detach()) # Detach generator
        disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
        disc_real_X_hat = disc_X(real_X)
        disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
        disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
        return disc_loss
    
    def get_gen_adversarial_loss(self,real_X, disc_Y, gen_XY, adv_criterion):
        '''
        Return the adversarial loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
                prediction matrices
            gen_XY: the generator for class X to Y; takes images and returns the images
                transformed to class Y
            adv_criterion: the adversarial loss function; takes the discriminator
                      predictions and the target labels and returns a adversarial
                      loss (which you aim to minimize)
        '''
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
        return adversarial_loss, fake_Y
    
    def get_identity_loss(self,real_X, gen_YX, identity_criterion):
        '''
        Return the identity loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            gen_YX: the generator for class Y to X; takes images and returns the images
                transformed to class X
            identity_criterion: the identity loss function; takes the real images from X and
                            those images put through a Y->X generator and returns the identity
                            loss (which you aim to minimize)
        '''
        identity_X = gen_YX(real_X)
        identity_loss = identity_criterion(identity_X, real_X)
        return identity_loss, identity_X
    
    def get_cycle_consistency_loss(self,real_X, fake_Y, gen_YX, cycle_criterion):
        '''
        Return the cycle consistency loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            fake_Y: the generated images of class Y
            gen_YX: the generator for class Y to X; takes images and returns the images
                transformed to class X
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                            those images put through a X->Y generator and then Y->X generator
                            and returns the cycle consistency loss (which you aim to minimize)
        '''
        cycle_X = gen_YX(fake_Y)
        cycle_loss = cycle_criterion(cycle_X, real_X)
        return cycle_loss, cycle_X
    
    def get_gen_loss(self,real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
        '''
        Return the loss of the generator given inputs.
        Parameters:
            real_A: the real images from pile A
            real_B: the real images from pile B
            gen_AB: the generator for class A to B; takes images and returns the images
                transformed to class B
            gen_BA: the generator for class B to A; takes images and returns the images
                transformed to class A
            disc_A: the discriminator for class A; takes images and returns real/fake class A
                prediction matrices
            disc_B: the discriminator for class B; takes images and returns real/fake class B
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator
                predictions and the true labels and returns a adversarial
                loss (which you aim to minimize)
            identity_criterion: the reconstruction loss function used for identity loss
                and cycle consistency loss; takes two sets of images and returns
                their pixel differences (which you aim to minimize)
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                those images put through a X->Y generator and then Y->X generator
                and returns the cycle consistency loss (which you aim to minimize).
                Note that in practice, cycle_criterion == identity_criterion == L1 loss
            lambda_identity: the weight of the identity loss
            lambda_cycle: the weight of the cycle-consistency loss
        '''
        # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
        adv_loss_BA, fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
        adv_loss_AB, fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
        gen_adversarial_loss = adv_loss_BA + adv_loss_AB

        # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
        identity_loss_A, identity_A = self.get_identity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_B, identity_B = self.get_identity_loss(real_B, gen_AB, identity_criterion)
        gen_identity_loss = identity_loss_A + identity_loss_B

        # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
        cycle_loss_BA, cycle_A = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss_AB, cycle_B = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
        gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

        # Total loss
        gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss
        return gen_loss, fake_A, fake_B

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
    def load_model(self,gan_path):
        generator_state = torch.load(gan_path)
        self.generatorAB.load_state_dict(generator_state)
        return self.generatorAB
    
    def sample(self,src_img):
        sample = self.generatorAB(src_img)
        super().show_tensor_images(sample,num_images=1,nrows=1)


