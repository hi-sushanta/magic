from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from magic.dcgan import DCGANTrain
import torch

data =  datasets.MNIST(
        "./data/",
        train=False,
        download=True, 
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
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
    batch_size=16,
    shuffle=True,
)


gantrain = DCGANTrain(1,28)
gantrain.train(dataloader)
# Down method using to when your model training is complete to get generate image.
# gantrain.sample()
