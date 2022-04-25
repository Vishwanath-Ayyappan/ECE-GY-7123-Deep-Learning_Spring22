import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

wb = pd.read_excel('Data_try (1).xlsx')
x=wb.values

X=[]
l=0
u=28
for i in range(0,1886):  #Data handling for NARX network
  A=[]
  for i in range(l,u):
    A.append(x[0][i])
  X.append(A)
  l=l+1
  u=u+1

Y=[]
l=28
u=56
for i in range(0,1886): 
  A=[]
  for i in range(l,u):
    A.append(x[0][i])
  Y.append(A)
  l=l+1
  u=u+1

X_train=[X[0:1500]]
X_test=[X[1500:np.shape(X)[0]]]
X_train=np.reshape(X_train,[1500,28])
X_train=torch.FloatTensor(X_train)
X_test=np.reshape(X_test,[386,28])
X_test=torch.FloatTensor(X_test)

Y_train=[Y[0:1500]]
Y_test=[Y[1500:np.shape(Y)[0]]]
Y_train=np.reshape(Y_train,[1500,28])
# Y_train=torch.Tensor([-1,0,1]).to(torch.long)
Y_train=torch.FloatTensor(Y_train)
Y_test=np.reshape(Y_test,[386,28])
Y_test=torch.LongTensor(Y_test)

class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super(NeuralNetwork,self).__init__()
    self.Network=torch.nn.Sequential(torch.nn.Linear(28,10), torch.nn.ReLU(),
                                    torch.nn.Linear(10,28))
    
  def forward(self,x):
    # x=x.view(-1,28)
    transformed_x=self.Network(x)
    return transformed_x

net=NeuralNetwork()
Loss=torch.nn.MSELoss()
Optimizer=torch.optim.Adam(net.parameters(),lr=0.01)

Train_loss_history=[]
Test_loss_history=[]
for epoch in range(100):
  Train_loss=0
  Test_loss=0
  for i in range(0,np.shape(X_train)[0]):
    input=X_train[i][:]
    labels = Y_train[i][:]
    Optimizer.zero_grad()
    Predicted_output=net(input)
    fit=Loss(Predicted_output,labels)
    fit.backward()
    Optimizer.step()
    Train_loss+=fit.item()
  for i in range(0,np.shape(X_test)[0]):
    with torch.no_grad():
      input=X_test[i][:]
      labels = Y_test[i][:]
      Predicted_output=net(input)
      fit=Loss(Predicted_output,labels)
      Test_loss+=fit.item()
  Train_loss=Train_loss/len(X_train)
  Test_loss=Test_loss/len(X_test)
  Train_loss_history.append(Train_loss)
  Test_loss_history.append(Test_loss)
  print("the train loss and test loss in iteration %s is %s and %s"%(epoch,Train_loss,Test_loss))
