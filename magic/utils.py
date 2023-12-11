import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

class BaseClass:
    def __init__(self,):
        pass
    
    def print_loss(self,epoch:int,gloss:float,dloss:float,glabel:str="Generator",dlabel:str="Discriminator"):
        print("\t========<>=============<>==============<>=========")
        print(f"\tEpoch:{epoch}")
        print(f"\t{glabel} Loss: {gloss}")
        print(f"\t{dlabel} Loss: {dloss}")
        print(f"\t========<>=============<>==============<>=========")
    
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
    


