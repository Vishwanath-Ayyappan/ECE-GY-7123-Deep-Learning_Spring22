def project1_model():
    return ResNet(ResidualBlock, [2, 1, 1, 1]).cuda()

torch.manual_seed(1)
model=project1_model()
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
