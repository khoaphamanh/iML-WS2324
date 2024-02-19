import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas as pd
import numpy as np
from torchinfo import summary
from sklearn.metrics import f1_score, precision_score, recall_score

#fix parameter and constante
SEED = utils.seed
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TEST_SIZE = utils.test_size
INPUT_SIZE = utils.input_size 
HIDDEN_SIZE = utils.hidden_size
HIDDEN_LAYERS = utils.hidden_layers 
OUTPUT_SIZE = utils.output_size 
LR = utils.lr_blackbox
EPOCH = utils.epoch_blackbox
WD = utils.wd_blackbox

#load data
name_X = utils.name_preprocessed_data_X
name_y = utils.name_preprocessed_data_y
path_X = os.path.join(project_path,"data",name_X)
path_y = os.path.join(project_path,"data",name_y)
X = pd.read_csv(path_X,index_col=0).to_numpy()
y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

#turn data to tensor and split it
def split_and_normalize(X,y,seed,test_size):
    """ 
    This function will split the data into train and test, then normalize the features in X_train using Min-Max-Scaler and fit_transform on X_test
    
    Parameters:
	    X (np.array): data
        y (np.array): label
        seed (int): random seed
        test_size (float): size of the test data
    Returns:
	    X_train (torch.tensor): train data
        X_test (torch.tensor): test data
        y_train (torch.tensor): train label
        y_test (torch.tensor): test label
    """
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
    
    #normalize the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    #turn data to tensor
    X_train, X_test, y_train, y_test = torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(y_train), torch.tensor(y_test)
    
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = split_and_normalize(X=X,y=y,seed=SEED,test_size=TEST_SIZE)
# count = 
# print("count:", count)

#create model
class BlackBoxModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, hidden_layers:int, output_size ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(hidden_layers)])
        #self.hidden_layers = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for hid_layer in self.hidden_layers:
            x = hid_layer(x)
            x = self.relu(x)
        #x = self.hidden_layers(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

#init model
model = BlackBoxModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS,output_size=OUTPUT_SIZE)
summary(model)

#optimizer and loss
count = torch.bincount(y_train)
pos_weight = count[0] / count[1]
print("pos_weight:", pos_weight)
loss = nn.BCEWithLogitsLoss() #pos_weight=pos_weight
optimizer = torch.optim.Adam(model.parameters(),lr=LR, weight_decay=WD)

# accuracy function
def accuracy_function (y_pred_label, y_true):
    check = torch.eq(y_pred_label,y_true)
    accuracy = sum(check) / len(y_true) * 100
    return accuracy

for epoch in range(EPOCH):

    #training mode:
    model.train()

    #forward pass:
    y_pred_train_logit = model(X_train).ravel()
    y_pred_train_probability = torch.sigmoid(y_pred_train_logit)
    y_pred_train_label = torch.round(y_pred_train_probability)

    #calculate the loss and accuracy:
    loss_train = loss(y_pred_train_logit,y_train.float())
    accuracy_train = accuracy_function(y_pred_train_label,y_train)
    f1_train = f1_score(y_pred=y_pred_train_label.detach().numpy(),y_true=y_train.detach().numpy())
    
    #zero grad the gradient
    optimizer.zero_grad()

    #backpropagation
    loss_train.backward()

    #update parameters
    optimizer.step()

    # testing
    model.eval()
    with torch.inference_mode():

        # forward pass test
        y_pred_test_logit = model(X_test).ravel()
        y_pred_test_probability = torch.sigmoid(y_pred_test_logit)
        y_pred_test_label = torch.round(y_pred_test_probability)  

        # calculate the loss and accuracy test
        loss_test = loss(y_pred_test_logit, y_test.float())
        accuracy_test = accuracy_function(y_pred_test_label,y_test)
        f1_test = f1_score(y_pred=y_pred_test_label,y_true=y_test)
        
    # tracking loss train test accuracy:
    if epoch % 10 == 0:
        print('epoch = {}'.format(epoch))
        print('loss train = {}, accuracy train = {}, f1 train {}'.format(loss_train, accuracy_train, f1_train))
        print('loss test = {}, accuracy test = {} , f1 test {}'.format(loss_test, accuracy_test, f1_test))
        print()

y_out = model(X_test)
y_out = torch.round(torch.sigmoid(y_out)).ravel().int()  
count_pred = torch.bincount(y_out)
print("count_pred:", count_pred)
count_true = torch.bincount(y_test)
print("count_true:", count_true)

y_out = model(X_test)
y_out = torch.round(torch.sigmoid(y_out)).ravel().int()  
count_pred = torch.bincount(y_out)
print("count_pred:", count_pred)
count_true = torch.bincount(y_test)
print("count_true:", count_true)
