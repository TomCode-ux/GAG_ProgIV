import torch
from torch import nn
import dataset


class NeuralNetwork(nn.Module):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self):
        super().__init__() 
        self.hidden_layer = nn.Linear(3 * 32 * 32, 30)  # 32x32x3 pikslit, 30 väljundit
        self.sigmoid = nn.Sigmoid()                     # aktivatsioonifunktsioon peidetud kihile
        self.output_layer = nn.Linear(30, 10)           # 30 sisendit, 10 väljundit (üks iga klassi kohta)
        self.softmax = nn.Softmax(dim=1)                # aktivatsioonifunktsioon väljundkihile
        
        data = torch.tensor([5.5, 3.5, 4.2])
        activation_fn = nn.Softmax()
        print(activation_fn(data))  # tensor([0.7103, 0.0961, 0.1936]) - 71%, 10%, 19%
        
    def forward(self, x):
        x = x.flatten(start_dim=1)    # 16x32x32x3 -> 16x3072
        z_1 = self.hidden_layer(x)    # 16x3072    -> 16x30
        a_1 = self.sigmoid(z_1)       # 16x30      -> 16x30
        z_2 = self.output_layer(a_1)  # 16x30      -> 16x10
        a_2 = self.softmax(z_2)       # 16x10      -> 16x10
        return a_2

  
    # -----Treenimine-----#
    def train():
        
        net = NeuralNetwork()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        max_epochs = 100

        for epoch in range(max_epochs):
            with torch.no_grad():         # me ei soovi arvutada gradiente
                for x, y in dataset:  # käime läbi kõik miniplokid
                    
                    y_hat = net(x)        # teeme ennustuse

                    y = nn.functional.one_hot(y, 10).to(torch.float)  # muudame y-i õigele kujule (10 klassi)
                    loss = loss_fn(y_hat, y)  # arvutame kahju statistika jaoks
        
    
    
    
    
#Lingid


                                        
                            