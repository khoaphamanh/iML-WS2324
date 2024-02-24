from sklearn.linear_model import LogisticRegression,Ridge
import numpy as np
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import joblib

#constante
SEED = utils.seed
np.random.seed(SEED)
NUM_PERTUBE = utils.num_pertube
NUMERICAL_FEATURE_INDEX = utils.numerical_feature_index

NAME_MODEL = utils.name_blackbox_nn 
INPUT_SIZE = utils.input_size 
HIDDEN_SIZE = utils.hidden_size
OUTPUT_SIZE = utils.output_size
HIDDEN_LAYERS = utils.hidden_layers

# add argument
parse = argparse.ArgumentParser()
parse.add_argument("--pertube_std", type=float, default=utils.pertube_std,
                   help="name of the pretrained generator model name.pth, default is '{}'".format(utils.pertube_std))
parse.add_argument("--num_pertube", type=int, default=utils.num_pertube,
                   help="number of intermediate dimensions in the first hidden layer in model, default is {}".format(utils.num_pertube))

# read the argument
args = parse.parse_args()
PERTUBE_STD = args.pertube_std
NUM_PERTUBE = args.num_pertube

#load data
name_X = utils.name_preprocessed_data_X
name_y = utils.name_preprocessed_data_y
path_X = os.path.join(project_path,"data",name_X)
path_y = os.path.join(project_path,"data",name_y)
X = pd.read_csv(path_X,index_col=0)
y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

# add unrelated columns, setup
# X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
# X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
X = X.to_numpy()

# pertube data
def pertube_data (X:np.array,x:np.array,numerical_feature_index: list,num_pertube:int,pertube_std:float):
    """ 
    This function will create neighbor points by adding Gaussian noise around a single instance. 
    
    Parameters:
	    X (np.array): datasets
        x (np.array): a single instance
        num_pertube (int): number of neighbor points
        pertube_std (float): standivation of gausian noise
    Returns:
	    new_datasets (np.array): real data and pertube samples shape (n_samples, num_pertube + 1, n_feature). num_pertube + 1 contains 1 real sample and num_pertube sample from Gaussian nois
    """
    
    #Normalize the data using standardize
    scaler = StandardScaler()
    scaler.fit(X)
    x = scaler.transform(x.reshape(1, -1)).flatten()
    
    #a list contains and sampled x
    new_X = []
    
    for n in range(num_pertube):
        x_ = np.array([x[i] + np.random.normal(0,pertube_std,1)[0] if i in numerical_feature_index else x[i]  for i in range(len(x))])
        new_X.append(x_)
    
    #inverse transform the sampled x
    new_X = np.array(new_X)
    new_X = scaler.inverse_transform(new_X)
    
    return new_X

# load interpretabel model
model = LogisticRegression()

# adversarial (black box) model
import torch
from torch import nn

#create model
class BlackBoxModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, hidden_layers:int, output_size:int ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(hidden_layers)])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for hid_layer in self.hidden_layers:
            x = hid_layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x
    
    def predict(self,x:np.array,prob = True):
        x = torch.tensor(x).float()
        with torch.inference_mode():
            out =  self.forward(x)
            if prob:
                return torch.sigmoid(out).numpy().ravel().astype(int)
            else:
                return out.numpy().ravel()

#pertube data
scaler = StandardScaler()

#load model
state_dict_path = os.path.join(project_path,"blackbox", NAME_MODEL+ ".pth")
state_dict = torch.load(state_dict_path)
model = BlackBoxModel(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE,hidden_layers=HIDDEN_LAYERS)
model.load_state_dict(state_dict)

#lime
class LimeCustom:
    def __init__(self, blackbox: nn.Module):
        """
        This class will have the misstion to calculate LIME from scratch. 
        
        Parameters:
            black_box : blackbox model, which can used to predict the label, needs to have a .predict function to get a label of input x
        """
        self.black_box = blackbox
        
    def get_lime (self, instance: np.array, sampled_instance: np.array, seed:int,sigma = None):
        """
        In this method we will apply LIME to explane an instance using interpretable model
        
        Parameters:
            inter_model (LogisticRegression): Interpretable model. We only use Linear Regression model in sklearn
            instance (np.array): instance needs to be explaned
            sampled_instance (np.array): sampled instance around x
            seed (int): random seed
        """
        
        #calculate phi
        X, phi = self.weights(instance=instance,sampled_instance=sampled_instance,sigma=sigma)
        #print("X shape:", X.shape)
        
        #train the interpretable mode
        inter_model = LogisticRegression(random_state=seed)
        scaler_path = os.path.join(project_path,"blackbox","mms_scaler.bin")
        scaler = joblib.load(scaler_path)
        
        X = scaler.transform(X)
        #print("X:", X[10:20])
        y = self.black_box.predict(X)
        print("y:", y)
        inter_model.fit(X,y, sample_weight=phi)

        #contribution each feature to the result
        weights = inter_model.coef_
        print("weights:", weights)
        bias = inter_model.intercept_
        print("bias:", bias)
        contribution = instance.reshape(1,-1) * weights
        
        return inter_model, contribution

    def weights(self,instance:np.array, sampled_instance: np.array, sigma = None ):
        """
        In this method we will calculate the weights of instance x and sampled_instance.
        
        Parameters:
            inter_model (LogisticRegression): Interpretable model. We only use Linear Regression model in sklearn
            instance (np.array): instance needs to be explaned
            sigma (float), default = None = sqrt(n_feature)*0.75: Kernel width
        
        Returns:
            sampled_data (np.array): instance x and 
	        phi (np.array): weights of sampled instance shape (len(sampled_instance),)
        """
        # sigma
        if sigma == None:
            sigma = np.sqrt(x.shape[-1]) * 0.75
            
        #calculate phi
        sampled_instance = np.concatenate((x.reshape(1,-1),sampled_instance),axis=0)
        d = np.sqrt(np.sum((instance - sampled_instance)**2,axis=1))
        phi = np.exp((-d**2) / (sigma**2))

        return sampled_instance, phi
    
test = LimeCustom(blackbox=model)
inter_model = LogisticRegression()
x = X[50]
sampled_instance = pertube_data(X,x,NUMERICAL_FEATURE_INDEX,NUM_PERTUBE,PERTUBE_STD)
#print("sampled_instance:", sampled_instance)
#sampled_instance = np.concatenate((x.reshape(1,-1),pertube_data(X,x,NUMERICAL_FEATURE_INDEX,NUM_PERTUBE,PERTUBE_STD)),axis=0) 
#print("sampled_instance:", sampled_instance)
#print("sampled_instance:", sampled_instance)
# lemon = test.get_lime(inter_model=inter_model,instance=x,sampled_instance=sampled_instance)
_, w = test.weights(instance=x,sampled_instance=sampled_instance)
#print("w shape:", w.shape)
#print("w:", w)

model ,contribution = test.get_lime(instance=x,sampled_instance=sampled_instance,seed=SEED)
print("contribution:", contribution)

        