import numpy as np
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler


#constante
SEED = utils.seed
NUM_PERTUBE = 10# utils.num_pertube 
np.random.seed(SEED)
NUMERICAL_FEATURE_INDEX = utils.numerical_feature_index
PERTUBE_STD = utils.pertube_std

#load data
name_X = utils.name_preprocessed_data_X
name_y = utils.name_preprocessed_data_y
path_X = os.path.join(project_path,"data",name_X)
path_y = os.path.join(project_path,"data",name_y)
X = pd.read_csv(path_X,index_col=0)
y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
X = X.to_numpy()

# pertube data
def pertube_data (X:np.array,x:np.array,numerical_feature_index: list,num_pertube:int,pertube_std:float, scaler: StandardScaler):
    """ 
    This function will create neighbor points by adding Gaussian noise around a single instance. 
    
    Parameters:
	    X (np.array): datasets
        x (np.array): a single instance
        num_pertube (int): number of neighbor points
        pertube_std (float): standivation of gausian noise
        scaler (StandardScaler): standardize scaler
    Returns:
	    new_datasets (np.array): real data and pertube samples shape (n_samples, num_pertube + 1, n_feature). num_pertube + 1 contains 1 real sample and num_pertube sample from Gaussian nois
    """
    
    #Normalize the data
    scaler.fit(X)
    x = scaler.transform(x.reshape(1, -1)).flatten()
    
    #a list contains and sampled x
    new_X = []
    
    for n in range(num_pertube):
        x_ = np.array([x[i] + np.random.normal(0,pertube_std,1)[0] if i in numerical_feature_index else x[i]  for i in range(len(x))])
        new_X.append(x_)
    
    new_X = np.array(new_X)
    return scaler.inverse_transform(new_X)

out = pertube_data(X,X[0],numerical_feature_index=NUMERICAL_FEATURE_INDEX,num_pertube=NUM_PERTUBE,pertube_std=PERTUBE_STD, scaler=StandardScaler())
x = X[0]
print("x:", x)
print("out:", out)
print("out shape:", out.shape)
