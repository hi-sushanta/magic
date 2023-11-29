
import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
import torch
from magic.gan import GANTrain
data =  datasets.MNIST(
        "./data/",
        train=True,
        download=True, 
        transform=transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    )

x = data.data
# print(x.shape)

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img = self.data[idx]
        return img.to(torch.float)

cdata = CustomDataset(x) # It's create becuse I only accept actual image not any label.
dataloader = DataLoader(
    cdata,
    batch_size=32,
    shuffle=True,
)

item = next(iter(dataloader))
gan_train = GANTrain(laten_dim=64,batch_size=32)

gan_train.train(dataloader)