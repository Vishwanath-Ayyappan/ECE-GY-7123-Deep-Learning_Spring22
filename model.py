# Optimizer Code: LookAhead Technique with Adam
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer

torch.manual_seed(1)
class Lookahead(Optimizer):

    def __init__(self, optimizer, la_steps=6, la_alpha=0.5, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss



# Data Augmentation and Loading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
torch.manual_seed(1)
transforms1=torchvision.transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
transforms2=torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
trainingdata = torchvision.datasets.CIFAR10('./CIFAR10/',train=True,download=True,transform=transforms1)
testdata = torchvision.datasets.CIFAR10('./CIFAR10/',train=False,download=True,transform=transforms2)
trainDataLoader = torch.utils.data.DataLoader(trainingdata,batch_size=32,shuffle=True,num_workers=2, pin_memory=True,
                        drop_last=True)
testDataLoader = torch.utils.data.DataLoader(testdata, batch_size=32,shuffle=False,
                        num_workers=2, pin_memory=True,
                        drop_last=False)

# Model

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        torch.manual_seed(1)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
         
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
           
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            torch.manual_seed(1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



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
torch.manual_seed(1)
model=ResNet(ResidualBlock).cuda()
Loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)
Optimizer = Lookahead(optimizer)
device = torch.device("cuda:0")
model.to(device)

# Training

total_step = len(trainDataLoader)

Train_loss_history=[]
Test_loss_history=[]
Train_accuracy_history=[]
Test_accuracy_history=[]
torch.manual_seed(1)
for epoch in range(15):
  Train_loss=0
  Test_loss=0
  for i,data in enumerate(trainDataLoader):
    images_train,labels_train = data
    images_train=images_train.cuda()
    labels_train=labels_train.cuda()
    Optimizer.zero_grad()
    Predicted_output=model(images_train)
    fit=Loss(Predicted_output,labels_train)
    fit.backward()
    Optimizer.step()
    Train_loss+=fit.item()
    
  for i,data in enumerate(testDataLoader):
    with torch.no_grad():
      images,labels = data
      images=images.cuda()
      labels=labels.cuda()
      Predicted_output=model(images)
      fit=Loss(Predicted_output,labels)
      Test_loss+=fit.item()
  Train_loss=Train_loss/len(trainDataLoader)
  Test_loss=Test_loss/len(testDataLoader)
  Train_loss_history.append(Train_loss)
  Test_loss_history.append(Test_loss)

  Predicted_output=model(images_train)
  correct_classifications=0
  for i in range(len(Predicted_output)):
    Compare=torch.argmax(Predicted_output[i])==labels_train[i]
    if Compare == True:
      correct_classifications+=1
      
  Train_accuracy=(correct_classifications/len(Predicted_output))*100
  print("The train accuracy is %s %s "%(Train_accuracy,'%'))
  Train_accuracy_history.append(Train_accuracy)
  
  Predicted_output=model(images)
  correct_classifications=0
  for i in range(len(Predicted_output)):
    Compare=torch.argmax(Predicted_output[i])==labels[i]
    if Compare == True:
      correct_classifications+=1
      
  Test_accuracy=(correct_classifications/len(Predicted_output))*100
  print("The test accuracy is %s %s "%(Test_accuracy,'%'))
  Test_accuracy_history.append(Test_accuracy)

  print("the train loss and test loss in iteration %s is %s and %s"%(epoch+1,Train_loss,Test_loss))

# callback code for early stopping

  if epoch == 0:
    best_loss=3
  if Test_loss < best_loss:
    best_loss = Test_loss
    es = 0
  else:
    es += 1
    print("Counter {} of 5".format(es))

    if es > 4:
        print("Early stopping with best_loss: ", best_loss, "and test_loss for this epoch: ", Test_loss, "...")
        break

# Code for Accuracy Report
Predicted_output_test=model(images)
Correct_classifications=0
for i in range(len(Predicted_output_test)):
  Compare=torch.argmax(Predicted_output_test[i])==labels[i]
  if Compare == True:
    Correct_classifications+=1
    
Test_accuracy=(Correct_classifications/len(Predicted_output_test))*100
print("The test accuracy is %s %s "%(Test_accuracy,'%'))

Predicted_output_train=model(images_train)
Correct_classifications=0
for i in range(len(Predicted_output_train)):
  Compare=torch.argmax(Predicted_output_train[i])==labels_train[i]
  if Compare == True:
    Correct_classifications+=1
    
Train_accuracy=(Correct_classifications/len(Predicted_output_train))*100
print("The train accuracy is %s %s "%(Train_accuracy,'%'))

# Plotting the Train and test loss over epochs

import matplotlib.pyplot as pt
import numpy as np
A=np.arange(1,epoch+2)
pt.title("Train loss and Test loss curve")
pt.plot(A,Train_loss_history,color="blue",label="Train loss")
pt.plot(A,Test_loss_history,color="red",label="Test loss")
pt.legend()
pt.show()

# Code to find Number of parameters in the network

from prettytable import PrettyTable
net = ResNet(ResidualBlock)
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(net);