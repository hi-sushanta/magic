
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
    batch_size=32,
    shuffle=True,
)

# item = next(iter(dataloader))
# print(item.shape)
gan_train = GANTrain(laten_dim=10,img_shape=(1,64,64),batch_size=32)
gan_train.train(dataloader)
# Pretrain Model weight load
# gen, disc = gan_train.load_model(gpath="model_weights\gen.pth",dpath="model_weights\disc.pth")

# Sample method 
# gan_train.sample()