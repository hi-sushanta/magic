from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from magic.dcgan import GANTrain
import torch
data =  datasets.MNIST(
        "./data/",
        train=True,
        download=True, 
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    )

x = data.data[:50]

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img = self.data[idx].unsqueeze(dim=0)
        img = transforms.RandomResizedCrop((64,64),antialias=True)(img)
        return img.to(torch.float)

cdata = CustomDataset(x) # It's create becuse I only accept actual image not any label.
dataloader = DataLoader(
    cdata,
    batch_size=32,
    shuffle=True,
)


gantrain = GANTrain(1,64)
gantrain.train(dataloader)
