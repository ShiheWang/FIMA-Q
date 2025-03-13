import torchvision.transforms as transforms
import torchvision 
import torch
from torch.utils.data import Dataset
import numpy as np
CIFAR_PATH = "~/data/cifar"
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
train_loader_kwargs = {
            'num_workers': 2,
            'pin_memory': True,
            'drop_last': False,
        }
class CacheDataset(Dataset):
    def __init__(self, datas) -> None:
        super().__init__()
        self.datas = datas
        
    def __getitem__(self,idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)
def cifar100_dataset(seed=3):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT期望的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=False, transform=transform)
    #print(len(cifar100_training))
    np.random.seed(seed)
    inds = np.random.permutation(len(cifar100_training))[:1024]
    preloaded_data = [cifar100_training[i] for i in inds]
    calib_set = CacheDataset(preloaded_data)
    trainloader = torch.utils.data.DataLoader(calib_set, batch_size=32, shuffle=True, **train_loader_kwargs)
    
    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader,testloader