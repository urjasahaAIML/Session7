import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dropout_value = 0.05
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),padding=1, bias=False),
            nn.ReLU(),  
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)         
        ) # o/p = 32, RF = 5, cout=32     

        # TRANSITION BLOCK 1
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # o/p = 16, RF = 10, cout=16    
        

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # o/p = 16, RF = 12, cout=64

         # TRANSITION BLOCK 2
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # o/p = 8, RF = 28, cout=32  


       # CONVOLUTION BLOCK 3 : Atrous Convolution witrh dilation = 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3),  padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # o/p = 8, RF = 28+4, cout=128

        # TRANSITION BLOCK 3
        self.transition3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        )  # o/p = 4, RF = (28+4)*2, cout=64

        # CONVOLUTION BLOCK 4 : Depthwise Separable Convolution
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3),  dilation=2, padding=3, groups=64, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(256),   
            nn.Dropout(dropout_value)       
        ) # o/p = 4, RF = 80?, cout=256
       
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # o/p = 1

        self.linear1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=0, bias=False)           
        )
        self.linear2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)           
        )
        self.linear3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)           
        )
        self.dropout = nn.Dropout(dropout_value)     

    def forward(self, x):
        x = self.convblock1(x)
        x = self.transition1(x)

        x = self.convblock2(x)
        x = self.transition2(x)

        x = self.convblock3(x)
        x = self.transition3(x)

        x = self.convblock4(x)
        #print('before gap', x.shape)
        x = self.gap(x)  
       
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
criterion = nn.CrossEntropyLoss()
net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.1)