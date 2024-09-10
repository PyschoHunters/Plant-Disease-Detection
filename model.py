import torch
from torch import nn

class CNN(nn.Module):

    #tiny VGG

    def __init__(self, input_shape:int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1), 
                            nn.SiLU(),
                            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1),
                            nn.SiLU(),
                            nn.BatchNorm2d(40),
                            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1), 
                            nn.SiLU(),
                            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1),
                            nn.SiLU(),
                            nn.BatchNorm2d(40),
                            nn.MaxPool2d(kernel_size=2,stride=2)
            
        )
        self.conv_block_3=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1), 
                            nn.SiLU(),
                            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1),
                            nn.SiLU(),
                            nn.BatchNorm2d(40),
                            nn.MaxPool2d(kernel_size=2,stride=2)
            
        )
        self.conv_block_4=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1), 
                            nn.SiLU(),
                            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1),
                            nn.SiLU(),
                            nn.BatchNorm2d(40),
                            nn.MaxPool2d(kernel_size=2,stride=2)
            
        )
        self.conv_block_5=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1), 
                            nn.SiLU(),
                            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                            kernel_size=3,stride=1,padding=1),
                            nn.SiLU(),
                            nn.BatchNorm2d(40),
                            nn.MaxPool2d(kernel_size=2,stride=2)
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape),
            nn.Flatten(),        
        )
    def forward(self,x):
        x=self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.conv_block_3(x)
        x=self.conv_block_4(x)
        x=self.conv_block_5(x)
        x=self.classifier(x)
        return x
