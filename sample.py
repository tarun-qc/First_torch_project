# first import method
# # import opendatasets as od
# # od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")

#second import method
import kagglehub
# Download latest version
path = kagglehub.dataset_download("mssmartypants/rice-type-classification")
print("Path to dataset files:", path)

import torch #part of torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary #part of torchsummary
from sklearn.model_selection import train_test_split #part of scikit-learn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu' #check if gpu is there or cpu is there
print(device)

data_df = pd.read_csv(path +"\\riceClassification.csv")
# print(data_df)
print(data_df.shape)
print(data_df.dropna(inplace=True))
print(data_df.drop(['id'], axis=1, inplace=True)) #axis=1 is representation to drop column
print(data_df.shape)
print("These are unique values in column of Class:",data_df["Class"].unique()) #print unique values for column Class column
"""
Count value repeats in column Class, It tells if dataset is balanced
# or there is big difference between population of two values
"""
print(data_df["Class"].value_counts())

"""Normalize (change of unit to make max value 1) each column in dataset, 
because big values will require more storage and loss function will roundup
the value for big numbers, which may give wrong result"""
origina_df = data_df.copy() #we require this for inference
for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max() #divide by max value of that column
print(data_df.head())

X = np.array(data_df.iloc[:,:-1]) #take all rows and columns except the last one
Y = np.array(data_df.iloc[:,-1]) #take all rows and only last column
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) #30% for testing, only take 70% data for training
"""X_val: validation
Y_val: validation
""" 
X_test, X_val,Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5) #only take 50% data for test and validation
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)