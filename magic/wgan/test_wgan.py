
import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
import torch
from magic.wgan import WGanTrain

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
        img = self.data[idx].unsqueeze(dim=0)
        return img.to(torch.float)

cdata = CustomDataset(data.data) # It's create becuse I only accept actual image not any label.
dataloader = DataLoader(
    cdata,
    batch_size=32,
    shuffle=True,
    drop_last=True
)


wgan_train = WGanTrain(in_chan=1,img_dim=28,batch_size=32)
wgan_train.train(dataloader)
# Pretrain Model weight load
# gen, critic = wgan_train.load_model(gpath="model_weights\wgen.pth",dpath="model_weights\wcritic.pth")

# wgan_train.sample()