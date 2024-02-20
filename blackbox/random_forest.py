import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

#fix parameter and constante
SEED = utils.seed
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
    
    # #turn data to tensor
    # X_train, X_test, y_train, y_test = torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(y_train), torch.tensor(y_test)
    
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = split_and_normalize(X=X,y=y,seed=SEED,test_size=TEST_SIZE)


count = np.unique(y_train, return_counts=True)
print("count:", count)
class_weights = {0: count[0], 1: count[1]} 

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=SEED,class_weight=class_weights)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

count_pred = np.unique(y_pred, return_counts=True)
print("count_pred:", count_pred)

count_true = np.unique(y_test, return_counts=True)
print("count_true:", count_true)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#calculate f1 score
f1_train = f1_score(y_true=y_test,y_pred=y_pred)
print("f1_train:", f1_train)

#calculate f1 score
f1_train = f1_score(y_true=y_train,y_pred=rf_classifier.predict())
print("f1_train:", f1_train)