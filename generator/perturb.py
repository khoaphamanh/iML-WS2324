import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

#fix parameter and constante
SEED = utils.seed
np.random.seed(SEED)
NUMERICAL_FEATURE_INDEX = utils.numerical_feature_index
PERTUBE_STD = utils.pertube_std
NUM_GEN = utils.num_gen 
NAME_GEN = utils.name_pertube_gen

#load data
name_X = utils.name_preprocessed_data_X
name_y = utils.name_preprocessed_data_y
path_X = os.path.join(project_path,"data",name_X)
path_y = os.path.join(project_path,"data",name_y)
X = pd.read_csv(path_X,index_col=0).to_numpy()
y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

# pertube data
def pertube_data_generator (X:np.array,x:np.array,numerical_feature_index: list,num_pertube:int,pertube_std:float, return_label = True):
    """ 
    This function will create neighbor points by adding Gaussian noise around a single instance. 
    
    Parameters:
	    X (np.array): datasets
        x (np.array): a single instance
        num_pertube (int): number of neighbor points
        pertube_std (float): standivation of gausian noise
        scaler (StandardScaler): standardize scaler
    Returns:
	    new_samples (np.array): real data and pertube samples shape (num_pertube + 1, n_feature). num_pertube + 1 contains 1 real sample and num_pertube sample from Gaussian nois
    """
    
    #Normalize the data using standardize
    scaler = StandardScaler()
    scaler.fit(X)
    x = scaler.transform(x.reshape(1, -1)).flatten()
    
    #a list contains and sampled x
    new_X = [x]
    new_y = [1]
    
    for n in range(num_pertube):
        x_ = np.array([x[i] + np.random.normal(0,pertube_std,1)[0] if i in numerical_feature_index else x[i] for i in range(len(x))])
        new_X.append(x_)
        new_y.append(0)
        
    new_X = np.array(new_X)
    new_X = scaler.inverse_transform(new_X)
    
    # return label real and fake
    if return_label == True:
        new_samples = np.column_stack((np.array(new_X), np.array(new_y))) 
        return new_samples
    
    # return only sampled x
    else:
        return new_X
    
#run localy in this file
if __name__ == "__main__":
    
    #generate pertubed data
    new_dataset = []
    for x in X:
        new_samples = pertube_data_generator(X = X,x=x,numerical_feature_index=NUMERICAL_FEATURE_INDEX,num_pertube=NUM_GEN,pertube_std=PERTUBE_STD)
        new_dataset.append(new_samples)
    new_dataset = np.array(new_dataset)
    
    #saved the generated data
    current_path = os.path.dirname(os.path.abspath(__file__))
    new_dataset_path = os.path.join(current_path,NAME_GEN+".npy")
    if not os.path.exists(new_dataset_path):
        np.save(new_dataset_path,new_dataset)
    print("Done!!! Your generated datasets is saved under name {} with shape {}".format(NAME_GEN+".npy", new_dataset.shape))    
    