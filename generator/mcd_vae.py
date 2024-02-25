import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt 
import joblib

"""
This file train the generator model as Monte Carlo Dropout Variational Auto Encoder (MCD VAE) on COMPAS dataset. The task is to generate data in same distribution of COMPAS dataset. In this file we use the hyperparameter and structure of the MCD VAE from paper and repostory https://github.com/domenVres/Robust-LIME-SHAP-and-IME

Output of this file are:
    - gen_data.npy: generated data
    - mcd_vae.pth: a pretrained MCD VAE
    - metrics_vae.png: performance visualization of MCD VAE
    - scaler.bin: scaler for normalization before train the model
"""

#fix parameter and constante
SEED = utils.seed
torch.manual_seed(SEED)
ORIGIN_DIM = utils.origin_dim_vae
TEST_SIZE = utils.test_size
CATEGORICAL_FEATURE_INDEX = utils.categorical_feature_index

NAME = utils.name_vae
INTERMEDIATE_DIM = utils.intermediate_dim_vae
LATENT_DIM = utils.latent_dim_vae
LR = utils.lr_vae
EPOCH = utils.epoch_vae
WD = utils.wd_vae
DROPOUT = utils.dropout_vae
BATCH_SIZE = utils.batch_size_vae
LANDA = utils.landa_vae
NUM_GEN = utils.num_gen
NAME_GEN = utils.name_gen
NAME_METRICS = utils.name_metrics_vae
NAME_SCALER = utils.name_scaler_vae

#create model
class MonteCarloDropoutVariationalAutoEncoder(nn.Module):
    def __init__(self, origin_dim: int, intermediate_dim: int, latent_dim:int, dropout:float):
        super().__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=origin_dim,out_features=intermediate_dim),
            nn.ReLU(),
            nn.Linear(in_features=intermediate_dim,out_features=intermediate_dim//2),
            nn.ReLU(),
        )
        
        #latent mean and variance
        self.latent_mean = nn.Linear(in_features=intermediate_dim//2,out_features=latent_dim)
        self.latent_variance = nn.Linear(in_features=intermediate_dim // 2, out_features=latent_dim)
        
        #decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=intermediate_dim//2),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=intermediate_dim//2,out_features=intermediate_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=intermediate_dim,out_features=origin_dim),
            nn.Sigmoid(),
        )
    
    def encode_mean_variance (self,x: torch.tensor):
        x = self.encoder(x)
        mean, variance = self.latent_mean(x), self.latent_variance(x)
        return mean, variance
    
    def reparameterization (self,mean: torch.tensor, variance: torch.tensor):
        epsilon = torch.randn_like(variance)
        z = mean + variance*epsilon
        return z

    def forward(self,x: torch.tensor):
        mean, variance = self.encode_mean_variance(x)
        z = self.reparameterization(mean,torch.exp(variance*0.5))
        x_ = self.decoder(z)
        return x_, mean, variance
    
    def latent_z (self, x: torch.tensor):
        with torch.inference_mode():
            mean, variance = self.encode_mean_variance(x)
            z = self.reparameterization(mean,torch.exp(variance*0.5))
            return z
        
    def generate_data (self,x: np.array,num_gen:int, scaler:MinMaxScaler, categorical_feature_index: list):
        
        """
        This function creates fake samples with label 0. True sample will have label 1. The categorical features should be rounded between 0 and 1.
        
        Parameters:
            x (torch.tensor): true sample
            num_gen (int): number of fake samples
            scaler (sklearn.MinMaxScaler): scaler of the train data
            categorical_feature_index (list): index of the categorcal feature 
        Returns:
            new_samples (np.array): array shape (num_gen + 1, n_feature + 1) contains true sample at row 0, fake samples at index row 1 to (num_gen-1). Last column is the label.
        """
        x_tensor = torch.tensor(x).float()
        new_X = [x]
        new_y = [1]
        for i in range(num_gen):
            with torch.inference_mode():
                # reconstruct 
                self.train()
                x_,_,_ = self.forward(x_tensor)
                x_ = x_.numpy()
                
                #round the categorical feature
                for i in categorical_feature_index:
                    #rounded the catorical feature between value 0 and 1
                    cat_feat = x_[i]
                    cat_feat = np.round(np.clip(cat_feat,0,1))
                    x_[i] = cat_feat
                
                #save it to list
                new_X.append(x_)
                new_y.append(0)
                
        #invers scaler        
        new_X = scaler.inverse_transform(new_X)
        new_samples = np.column_stack((np.array(new_X), np.array(new_y))) 
        return new_samples
    
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
    
    return X_train, X_test, y_train, y_test, scaler

#run localy in this file
if __name__ == "__main__":
    #load data
    name_X = utils.name_preprocessed_data_X
    name_y = utils.name_preprocessed_data_y
    path_X = os.path.join(project_path,"data",name_X)
    path_y = os.path.join(project_path,"data",name_y)
    X = pd.read_csv(path_X,index_col=0).to_numpy()
    y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

    #normalize and split data
    X_train, X_test, y_train, y_test, scaler = split_and_normalize(X=X,y=y,seed=SEED,test_size=TEST_SIZE)

    #create tensor dataset and dataloader
    train_data = TensorDataset(X_train,y_train)
    test_data = TensorDataset(X_test,y_test)

    train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    #init model
    model = MonteCarloDropoutVariationalAutoEncoder(origin_dim=ORIGIN_DIM,intermediate_dim=INTERMEDIATE_DIM,latent_dim=LATENT_DIM,dropout=DROPOUT)
    summary(model)

    #create custom loss
    class Loss (nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,x_: torch.tensor, x: torch.tensor, mean: torch.tensor,variance: torch.tensor, lamda = 1):
            BCE = nn.BCELoss(reduction="sum")
            reconstruction_loss = BCE(x_,x)
            KLD = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
            return reconstruction_loss + lamda * KLD, reconstruction_loss, KLD
        
    # optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=WD,lr=LR)
    loss = Loss()

    #train loop
    loss_train_list = []
    loss_reconstruct_train_list = []
    loss_kld_train_list = []

    loss_test_list = []
    loss_reconstruct_test_list = []
    loss_kld_test_list = []

    for ep in range(EPOCH):
        #train process
        loss_total_train = 0
        loss_reconstruct_train = 0
        loss_kld_train = 0
        model.train()
        
        for batch_train,(X_train,y_train) in enumerate(train_loader):
            
            #forward pass
            X_pred_train,mean_train,variance_train = model(X_train)
            
            #calculate the loss
            loss_train_total_this_batch, loss_train_reconsruct_this_batch, loss_train_kld_this_batch = loss(X_pred_train,X_train,mean_train,variance_train,LANDA)
            loss_total_train = loss_total_train + loss_train_total_this_batch
            loss_reconstruct_train = loss_reconstruct_train + loss_train_reconsruct_this_batch
            loss_kld_train = loss_kld_train + loss_train_kld_this_batch

            #zero grad
            optimizer.zero_grad()

            #backpropagation
            loss_train_total_this_batch.backward()

            #update the parameters
            optimizer.step()

        #calculate the mean of the loss
        loss_total_train = loss_total_train / len(train_loader)
        loss_reconstruct_train = loss_reconstruct_train / len(train_loader)
        loss_kld_train = loss_kld_train / len(train_loader)
        
        #evaluate step
        loss_total_test = 0
        loss_reconstruct_test = 0
        loss_kld_test = 0
        model.eval()

        with torch.inference_mode():
            for batch_test,(X_test, y_test) in enumerate(test_loader):
                
                #forward pass
                X_pred_test,mean_test,variance_test = model(X_test)
                
                #calculate the loss
                loss_test_total_this_batch, loss_test_reconsruct_this_batch, loss_test_kld_this_batch = loss(X_pred_test,X_test,mean_test,variance_test,LANDA)
                loss_total_test = loss_total_test + loss_test_total_this_batch
                loss_reconstruct_test = loss_reconstruct_test + loss_test_reconsruct_this_batch
                loss_kld_test = loss_kld_test + loss_test_kld_this_batch

            #calculate the mean of the loss
            loss_total_test = loss_total_test / len(test_loader)
            loss_reconstruct_test = loss_reconstruct_test / len(test_loader)
            loss_kld_test = loss_kld_test / len(test_loader)
        
        #print the metrics
        if ep % 10 == 0 or ep == EPOCH-1:
            print("epoch {}, loss total train {}, loss reconstruct train {}, loss kld train {}".format(ep,loss_total_train, loss_reconstruct_train, loss_kld_train))
            print("epoch {}, loss total test {}, loss reconstruct test {}, loss kld test {}".format(ep,loss_total_test, loss_reconstruct_test, loss_kld_test))
            print()

        #store the metrics in list
        loss_train_list.append(loss_total_train.detach().numpy())
        loss_reconstruct_train_list.append(loss_reconstruct_train.detach().numpy())
        loss_kld_train_list.append(loss_kld_train.detach().numpy())
        
        loss_test_list.append(loss_total_test.detach().numpy())
        loss_reconstruct_test_list.append(loss_reconstruct_test.detach().numpy())
        loss_kld_test_list.append(loss_kld_test.detach().numpy())

    #save the model
    state_dict = model.state_dict()
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path,NAME+".pth")
    if not os.path.exists(model_path):
        torch.save(state_dict,model_path)
    print("Done!!! Your model is saved under name {}".format(NAME+".pth"))

    #plot the metrics:
    plt.figure(figsize=(13,7))
    plt.suptitle("Metrics of MCD VAE")
    metrics_path = os.path.join(current_path,NAME_METRICS+".png")
    epoch_list = [i for i in range(EPOCH)]

    plt.subplot(1,3,1)
    plt.plot(epoch_list,loss_train_list,label = 'train')
    plt.plot(epoch_list, loss_test_list, label='test')
    plt.legend(loc = 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Total loss of MCD VAE')

    plt.subplot(1,3,2)
    plt.plot(epoch_list,loss_reconstruct_train_list,label = 'train')
    plt.plot(epoch_list, loss_reconstruct_test_list, label='test')
    plt.legend(loc = 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Reconstruct loss (BCE Loss)')

    plt.subplot(1,3,3)
    plt.plot(epoch_list,loss_kld_train_list,label = 'train')
    plt.plot(epoch_list, loss_kld_test_list, label='test')
    plt.legend(loc = 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('KLD Loss')

    if not os.path.exists(metrics_path):
        plt.savefig(metrics_path)
    print("Done!!! Metrics of model is saved under name {}".format(NAME_METRICS+".png"))

    #generated data
    new_dataset = []
    for x in X:
        x = scaler.transform(x.reshape(1, -1)).flatten()
        new_samples = model.generate_data(x = x,num_gen=NUM_GEN,scaler=scaler,categorical_feature_index=CATEGORICAL_FEATURE_INDEX)
        new_dataset.append(new_samples)
    new_dataset = np.array(new_dataset)

    #save generated data
    new_dataset_path = os.path.join(current_path,NAME_GEN+".npy")
    if not os.path.exists(new_dataset_path):
        np.save(new_dataset_path,new_dataset)
    print("Done!!! Your generated datasets is saved under name {} with shape {}".format(NAME_GEN+".npy", new_dataset.shape))

    #save scaler from preprocessing step:
    scaler_path = os.path.join(current_path,NAME_SCALER+".bin")
    if not os.path.exists(scaler_path):
        joblib.dump(scaler,scaler_path,compress=True)
    