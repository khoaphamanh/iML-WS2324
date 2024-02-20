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
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import joblib

"""
Output of this file is the pretrained blackbox model as random forest on COMPAS dataset. User can run this file with syntax from argparse to test our file:

    python random_forest.py --name "yourname" --n_estimators "your_n_estimators"
    
Then the output should be yourname.pkl. Default of the arguments:

--name: "random_forest" and our output files is random_forest.pkl
--n_estimators: 100
"""

#fix parameter and constante
SEED = utils.seed
TEST_SIZE = utils.test_size

# add argument
parse = argparse.ArgumentParser()
parse.add_argument("--name", type=str, default=utils.name_blackbox_rf,
                   help="name of the pretrained blackbox model name.pth, default is 'random_forest'")
parse.add_argument("--n_estimators", type=int, default=utils.n_estimators_blackbox,
                   help="number of trees in model, default is 100")

# read the argument
args = parse.parse_args()
NAME = args.name
N_ESTIMATORS = args.n_estimators

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
    
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = split_and_normalize(X=X,y=y,seed=SEED,test_size=TEST_SIZE)

#compute the class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train),y= y_train)
class_weights = {0: class_weights[0], 1: class_weights[1]} 

#train the model
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, class_weight=class_weights) 
model.fit(X_train, y_train)

# Make predictions 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("accuracy_train:", accuracy_train)
print("accuracy_test:", accuracy_test)

#calculate f1 score
f1_train = f1_score(y_true=y_train,y_pred=y_pred_train)
f1_test = f1_score(y_true=y_test,y_pred=y_pred_test)
print("f1_train:", f1_train)
print("f1_test:", f1_test)

#save the model
current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_path,NAME+".pkl")
if not os.path.exists(model_path):
    joblib.dump(model,model_path)
print("Done!!! Your model is saved under name {}".format(NAME+".pkl"))
