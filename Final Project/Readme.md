# Automated Hyperparameter Tuning based on Bayesian Optimization for Time-Series Prediction using NAR and LSTM Networks

## Abstract
A Non-linear Autoregressive Neural Network (NARNN) is designed for time series prediction of Walmart sales data with an objective of maximizing the test accuracy. Bayesian Optimization  is utilized in this network for automated hyperparameter tuning to determine the optimum parameters that do not overfit, and this method is analyzed in comparison with grid search and manual search in order to realize to what extent it improves the overall hyperparameter tuning process. The NAR network will be compared with the Long Short-Term Memory (LSTM) Network  in order to obtain the most accurate model.

## Dataset

The M5 dataset, made available by Walmart, is used and this dataset was utilized in Kaggle’s M5 2020 time series prediction competition as well [9]. It contains information about the unit sales of different products sold across 10 stores of Walmart, located in three states in the USA: California, Texas and Wisconsin. The dataset has time series data of the products over a period of more than 5 years from 01/29/2011 to 06/19/2016 (1941 data points) given in Figure 2. The dataset is divided into three different categories namely hobbies, household and foods and the unit sales heat map of the three categories is given in Figure 1a, 1b and 1c. The dataset also contains information about other variables like special events (holidays, sporting events, promotion events etc.,) and sale price of each product.
## Installation
The follwoing libraries area required to run the model
```sh
!pip install optuna
```
```sh
!pip install dask[dataframe] --upgrade
```
## GPU Requirements

The project requires a GPU to run faster. We have taken advantage of Google Colab's GPU which uses NVIDIA Tesla K80 GPU.
To check for GPU device, run the following
```sh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
and to run the model in GPU, 

```sh
model.to('cuda')
```

## Running the Model



- [Data_try.xlxs](https://github.com/Vishwanath-Ayyappan/ECE-GY-7123-Deep-Learning_Spring22/blob/main/Final%20Project/Data_try.xlsx) - Contains the data for running NAR network
- [nar_with_bayes.py](https://github.com/Vishwanath-Ayyappan/ECE-GY-7123-Deep-Learning_Spring22/blob/main/Final%20Project/nar_with_bayes.py) - NAR network with Bayesian Optimization
- [nar_with_grid_search.py](https://github.com/Vishwanath-Ayyappan/ECE-GY-7123-Deep-Learning_Spring22/blob/main/Final%20Project/nar_with_grid_search.py) - NAR network with Grid Search Optimization
- [lstm_final.py](https://github.com/Vishwanath-Ayyappan/ECE-GY-7123-Deep-Learning_Spring22/blob/main/Final%20Project/lstm_final.py) - LSTM Netwrok
- [requirement.txt](https://github.com/Vishwanath-Ayyappan/ECE-GY-7123-Deep-Learning_Spring22/blob/main/Final%20Project/requirement.txt) - Contains all necessary libraries to be imported with specification

> Note: `data_try.xlxs` is to be loaded for running the NAR network
> Note: ` sell prices.csv`, `calendar.csv`, `sample_submission.csv`, `sales_train_validation.csv` is to be loaded for running the LSTM network
## Acknowledgement 
This project is done as a part of the academic course ECE-GY 7123 Deep Learning taught at NYU Tandon, Spring 22 by Prof. Siddharth Garg and Prof. Arsalan Mosenia.
###### Authors:  
ArunKumar Nachimuthu Palanichamy, Shoban Venugopal Palani, Viswanathan Babu Chidambaram Ayyappan
