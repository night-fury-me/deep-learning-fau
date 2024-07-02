import torch                        #type: ignore
import torch.nn as nn               #type: ignore
import torch.nn.functional as F     #type: ignore

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Down sampling for skip connection to match dimensions
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        skip_connection = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            skip_connection = self.downsample(skip_connection)
        
        out += skip_connection
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        # Initial convolution, batch norm, and relu
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResBlock(64,    64,   stride = 1),
            ResBlock(64,    128,  stride = 2),
            ResBlock(128,   256,  stride = 2),
            ResBlock(256,   512,  stride = 2)
        )
        
        self.avgpool = nn.AvgPool2d(kernel_size = 10)
        self.fc      = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.initial_block(x)
        x = self.residual_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x
