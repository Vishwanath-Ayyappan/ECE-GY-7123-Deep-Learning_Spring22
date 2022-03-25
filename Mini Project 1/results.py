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
