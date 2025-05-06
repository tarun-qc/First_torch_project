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
print(data_df.shape) #know shape of data
print(data_df.dropna(inplace=True)) #drop empty values
print(data_df.drop(['id'], axis=1, inplace=True)) #axis=1 is representation to drop column, otherwise it will be row
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
original_df = data_df.copy() #we require this for inference
for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max() #divide by max value of that column
print(data_df.head())

X = np.array(data_df.iloc[:,:-1]) #take all rows and columns except the last one
Y = np.array(data_df.iloc[:,-1]) #take all rows and only last column
print(X)
print(Y)
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) #30% for testing, only take 70% data for training
"""X_val: validation
Y_val: validation
""" 
X_test, X_val,Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5) #only take 50% data for test and validation
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

class dataset(Dataset):
    def __init__(self, X, Y): #this is constructor, self, input,output
        self.X = torch.tensor(X, dtype = torch.float32).to(device) #convert our numpy or panda data to tensor
        self.Y = torch.tensor(Y, dtype = torch.float32).to(device) #torch can only understand tensor
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

training_data = dataset(X_train, Y_train)
validation_data = dataset(X_val, Y_val)
testing_data = dataset(X_test, Y_test)

train_dataloader = DataLoader(training_data, batch_size = 32, shuffle = True)
validation_dataloader = DataLoader(training_data, batch_size = 32, shuffle = True)
testing_dataloader = DataLoader(training_data, batch_size = 32, shuffle = True)

for x,y in train_dataloader:
    print(x)
    print("======")
    print(y)
    break

HIDDEN_NEURONS = 10
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = MyModel().to(device)

summary(model, (X.shape[1],))
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr =1e-3) #Adam function takes parameters and learning rate

total_loss_train_plot = [] #total loss
total_loss_validation_plot = [] #total loss
total_acc_train_plot = [] #total accuracy
total_acc_validation_plot = [] #total accuracy

epochs = 10
for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    for data in train_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1) #inorder to make prediction in same dimension as labels
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()
        acc = (prediction.round() == labels).sum().item()
        # print(acc)
        total_acc_train += acc
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels =data

            prediction = model(inputs).squeeze(1)
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()
            acc = ((prediction).round() == labels).sum().item()

            total_acc_val += acc
    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/training_data.__len__()*100, 4))
    total_acc_validation_plot.append(round(total_acc_val/validation_data.__len__()*100, 4))
    print(f'''Epoch no. {epoch+1} Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round(total_acc_train/training_data.__len__()*100, 4)} Validation Loss: {round(total_loss_val/1000, 4)} Validation Accuracy: {round(total_acc_val/validation_data.__len__()*100, 4)}''')
    print("="*25)
with torch.no_grad():
    total_loss_test=0
    total_acc_test=0
    for data in testing_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss_test = criterion(prediction, labels).item()
        total_loss_test += batch_loss_test
        acc = ((prediction).round() == labels).sum().item()
        total_acc_test += acc
print("Accuracy: ", round(total_acc_test/testing_data.__len__()*100,4))
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,5))
axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title("Training and Validation Loss over epochs")
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0,2])
axs[0].legend()
axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title("Training and Validation Accuracy over epochs")
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0,100])
axs[1].legend()
plt.show()
area = 6284/original_df['Area'].abs().max()
MajorAxisLength = 81/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = 42/original_df['MinorAxisLength'].abs().max()
Eccentricity = 32/original_df['Eccentricity'].abs().max()
ConvexArea = 12/original_df['ConvexArea'].abs().max()
EquivDiameter = 12/original_df['EquivDiameter'].abs().max()
Extent = 98/original_df['Extent'].abs().max()
Perimeter = 98/original_df['Perimeter'].abs().max()
Roundness = 677/original_df['Roundness'].abs().max()
AspectRation = 24/original_df['AspectRation'].abs().max()
my_prediction = model(torch.tensor([area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation], dtype=torch.float32).to(device))
print(round(my_prediction.item()))