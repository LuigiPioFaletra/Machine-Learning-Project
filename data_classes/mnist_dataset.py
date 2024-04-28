import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, train=True, root='data'):
        self.data = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        item = {
            'image': image,
            'label': label
        }
        return item
