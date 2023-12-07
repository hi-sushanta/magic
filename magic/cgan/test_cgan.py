import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from magic.cgan import CGANTrain

data =  datasets.MNIST(
        "./data/",
        train=False,
        download=True, 
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        )

dataloader = DataLoader(
    data,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

item = next(iter(dataloader))
cgan_train = CGANTrain(in_chan=1,img_dim=28,num_classes=10,batch_size=32)
cgan_train.train(dataloader)
# Pretrain Model weight load
# gen, critic = wgan_train.load_model(gpath="model_weights\wgen.pth",dpath="model_weights\wcritic.pth")
# Only down code help when your model training complete.
# import torch
# cgan_train.sample(torch.tensor([8,8,3,2]))