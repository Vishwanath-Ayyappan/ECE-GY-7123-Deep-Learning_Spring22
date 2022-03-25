# Mini Project-1
## Residual Network for CIFAR-10 Classification
This Github Repo folder named "Mini Project -1" contains all the files required to run the model and other files to show and plot the results.

## Files

- Lookahead.py = Optimization technique to boost Adam Performance
- ResidualBlock.py = Residual Skip Connections for the Neural Network
- ResNet.py = Neural Network
- train.py = To run and train the model
- results.py = Plots and accuracy reports for the model
- count_parameters.py = Displays the number of trainable parameters in each layer
- final.pt = Contains the model weights which can be resused to run the model again
- model.py = Contains the total code for this project

## Installation
This code requires Pytorch and other torchvision libraries to run.
```sh
torch version:'1.10.0+cu111'
```
## Running the model
Run all the files in the following order
- Lookahead.py
- ResidualBlock.py
- ResNet.py
- train.py
- results.py
```sh
python3 model.py
```
The model.py contains project1_model which can also be called to train the model. But to get results and plot, call
```sh
python3 results.py
```
If you are running in an interactive environment, you can simply run the model.py file.
This works only if you have downloaded the necessary libraries and have the same version of libraries as this code.

The final.pt file contains the weights downloaded from the model. These weights can be reproduced to check the accuracy or test the model with a custom held-out dataset.

This folder also includes a report paper which has detailed explanations of the model.
