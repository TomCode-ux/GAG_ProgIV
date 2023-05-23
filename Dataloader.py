import torch
import torchvision

class dataloader():
    def __init__(self):
        
        data_loader = torch.utils.data.DataLoader(data,batch_size=10,shuffle=True)
        
        train_features, train_labels = next(iter(data_loader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        
        label = train_labels[0]
        self.data = next(iter(data_loader))
        print(f"Label: {label}")
    
    def get_data(self):
        return(self.data)
