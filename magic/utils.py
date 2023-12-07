import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

class BaseClass:
    def __init__(self,):
        self.fig = None 
        self.ax = None 
        
    def loss_plot(self,epoch:int,gloss:float,dloss:float,title:str,label_loss:str,last_epoch:int,sign:str,gcolor="green",
                  dcolor='red'):
         # Create a new figure if needed
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        # Add the current loss to the plot
        self.fig.suptitle(title)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel(label_loss)
        if epoch == 1:
            self.ax.plot(epoch, gloss, sign,color=gcolor,label="Generator")
            self.ax.plot(epoch,dloss,sign,color=dcolor,label="Discriminator")
            plt.legend()

        else:
            self.ax.plot(epoch,gloss,sign,color=gcolor)
            self.ax.plot(epoch,dloss,sign,color=dcolor)

        # Update the plot
        # Add a legend

        plt.draw()
        plt.pause(0.01)
        if epoch+1 == last_epoch:
            plt.show()
    
    def show_tensor_images(self,image_tensor, num_images = 4,nrows=2,show=True):
      '''
      Function for visualizing images: Given a tensor of images, number of images, and
      size per image, plots and prints the images in an uniform grid.
      '''
      image_tensor = (image_tensor + 1) / 2
      image_unflat = image_tensor.detach().cpu()
      image_grid = make_grid(image_unflat[:num_images], nrow=nrows)
      plt.figure(figsize=(20,20))
      plt.imshow(image_grid.permute(1, 2, 0).squeeze())
      if show:
          plt.show()
          plt.axis('off')
    
    def create_dir(self,folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        else:
            print(f"Folder already exists: {folder_name}")
    


