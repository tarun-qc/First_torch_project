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
from torchsummary import summary #part of torchsummary
from sklearn.model_selection import train_test_split #part of scikit-learn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
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
#########################

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

## Data Loading and Preparation
def load_data():
    """Download and load the rice classification dataset"""
    # Download dataset
    dataset_path = kagglehub.dataset_download("mssmartypants/rice-type-classification")
    csv_path = os.path.join(dataset_path, "riceClassification.csv")
    
    # Load data
    data_df = pd.read_csv(csv_path)
    print(f"Initial dataset shape: {data_df.shape}")
    
    # Data cleaning
    data_df.dropna(inplace=True)  # Remove missing values
    data_df.drop(['id'], axis=1, inplace=True)  # Remove ID column
    
    print(f"Cleaned dataset shape: {data_df.shape}")
    print("\nClass distribution:")
    print(data_df["Class"].value_counts())
    
    return data_df

## Data Preprocessing
class RicePreprocessor:
    def __init__(self, data_df):
        self.data_df = data_df
        self.original_df = data_df.copy()
        self.feature_columns = [col for col in data_df.columns if col != 'Class']
        
    def normalize_features(self):
        """Normalize features to [0, 1] range"""
        for column in self.feature_columns:
            self.data_df[column] = self.data_df[column] / self.data_df[column].abs().max()
        return self.data_df
    
    def split_data(self, test_size=0.3, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split into train+val and test
        X = self.data_df[self.feature_columns].values
        y = self.data_df['Class'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Then split test into validation and final test
        val_ratio = val_size / test_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=1-val_ratio, random_state=random_state
        )
        
        print(f"\nData splits:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

## Dataset Class
class RiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

## Neural Network Model
class RiceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2):
        super(RiceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size//2, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

## Training Utilities
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            predicted = outputs.round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = outputs.round()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                preds = outputs.round()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return accuracy_score(all_labels, all_preds)

## Visualization
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

## Inference Function
def predict_sample(model, sample_data, original_df):
    """Predict a single sample"""
    # Normalize the sample data using the same scaling as training data
    normalized_sample = []
    for i, col in enumerate(original_df.columns[:-1]):
        normalized_val = sample_data[i] / original_df[col].abs().max()
        normalized_sample.append(normalized_val)
    
    # Convert to tensor and predict
    sample_tensor = torch.tensor(normalized_sample, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(sample_tensor)
    
    return prediction.item()

## Main Execution
if __name__ == "__main__":
    # Load and prepare data
    data_df = load_data()
    preprocessor = RicePreprocessor(data_df)
    normalized_df = preprocessor.normalize_features()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    # Create datasets and dataloaders
    train_dataset = RiceDataset(X_train, y_train)
    val_dataset = RiceDataset(X_val, y_val)
    test_dataset = RiceDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = RiceClassifier(input_size, hidden_size=128, dropout_rate=0.3)
    summary(model, (input_size,))
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    trainer = Trainer(model, criterion, optimizer, device)
    
    # Train the model
    history = trainer.train(train_loader, val_loader, epochs=15)
    plot_training_history(history)
    
    # Evaluate on test set
    test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Example prediction
    sample_data = [
        6284,  # Area
        81,    # MajorAxisLength
        42,    # MinorAxisLength
        32,    # Eccentricity
        12,    # ConvexArea
        12,    # EquivDiameter
        98,    # Extent
        98,    # Perimeter
        677,   # Roundness
        24     # AspectRation
    ]
    
    pred_prob = predict_sample(model, sample_data, preprocessor.original_df)
    print(f"\nSample prediction probability: {pred_prob:.4f}")
    print(f"Predicted class: {'Jasmine' if pred_prob > 0.5 else 'Gonen'}")