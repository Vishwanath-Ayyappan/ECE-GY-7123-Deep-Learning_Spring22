import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 32
        torch.manual_seed(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),           
            nn.ReLU(),
           
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
     
        self.layer2 = self.make_layer(ResidualBlock,128 , 1, stride=2)
    
        self.layer3 = self.make_layer(ResidualBlock,256 , 1, stride=2)
     
        self.layer4 = self.make_layer(ResidualBlock, 512 , 1, stride=2)
     
        self.fc = nn.Linear(512, num_classes)
       

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
  
        out = self.layer2(out)
      
        out = self.layer3(out)
   
        out = self.layer4(out)
     
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
 
        return out
