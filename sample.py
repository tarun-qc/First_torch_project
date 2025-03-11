# first import method
# # import opendatasets as od
# # od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")

#second import method
import kagglehub
# Download latest version
path = kagglehub.dataset_download("mssmartypants/rice-type-classification")
print("Path to dataset files:", path)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)