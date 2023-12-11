
import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
import torch
from magic.pix2pix import Pix2Pix

data =  datasets.MNIST(
        "./data/",
        train=False,
        download=True, 
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        )


class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img = transforms.RandomResizedCrop((256,256),antialias=True)(self.data[idx].unsqueeze(dim=0))
        timg = transforms.GaussianBlur((3,3))(img)
        return img.to(torch.float),timg.to(torch.float)

# It's just demo purpose to using this dataset.
# but your case must be used src image with targated image.
cdata = CustomDataset(data.data[:32]) 
dataloader = DataLoader(
    cdata,
    batch_size=32,
    shuffle=True,
)

item = next(iter(dataloader))
pix_train = Pix2Pix(in_chan=1)
pix_train.train(dataloader)
# pix_train.sample(item[0][0].unsqueeze(dim=0))
