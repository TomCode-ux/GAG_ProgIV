from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class dataset():
    def __init__():
        train_data = datasets.CIFAR10(
            root='data',                     # kaust, kuhu andmed laetakse alla
            train=True,                      # kas soovime treeningandmeid (True) või testimisandmeid (False)
            download=True,                   # kas laeme vajadusel andmed automaatselt alla internetist
            transform=transforms.ToTensor()  # teisendame pildid tensoriteks
        )
        
        test_data = datasets.CIFAR10(
            root='data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        x, y = next(iter(train_loader))  # võtame miniploki
        
        print(x.shape)  # torch.Size([16, 3, 32, 32]) - 16 pilti, 3 värvikanalit (RGB), 32x32 pikslit
        print(y.shape)  # torch.Size([16]) - 16 märgendit (klassi, kuhu miniploki pildid kuuluvad)
        
        return x,y