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
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader

"""
Output of this file is the generator model as neural network on COMPAS dataset. The task is binary classification, we use Adam optimizer and BCE Loss to train model. User can run this file with syntax from argparse to test our file:

    python mcd_vae.py --name "yourname" --intermediate_dim "your_intermediate_dim" --latent_dim "your_latent_dim" --dropout "your_dropout" --batch_size "your_batch_size" --lr "your_lr" --epoch "your_epoch" --wd "your_wd" --landa "your_landa"
    
Then the output should be yourname.pth. Default of the arguments:

--name: "mcd_vae" and our output files is mcd_vae.pth
--intermediate_dim: 8
--latent_dim: 4
--dropout: 0.3
--batch_size: 100
--lr: 0.001
--epoch: 100
--wd: 1e-4
--landa: 1
"""

#fix parameter and constante
SEED = utils.seed
torch.manual_seed(SEED)
ORIGIN_DIM = utils.origin_dim_vae
TEST_SIZE = utils.test_size

# add argument
parse = argparse.ArgumentParser()
parse.add_argument("--name", type=str, default=utils.name_vae,
                   help="name of the pretrained generator model name.pth, default is 'mcd_vae'")
parse.add_argument("--intermediate_dim", type=int, default=utils.intermediate_dim_vae,
                   help="number of intermediate dimensions in the first hidden layer in model, default is 8")
parse.add_argument("--latent_dim", type=int, default=utils.latent_dim_vae,
                   help="number of dimension in latent space in model, default is 4")
parse.add_argument("--dropout", type=float, default=utils.dropout_vae,
                   help="dropout rate, default is 0.3")
parse.add_argument("--batch_size", type=int, default=utils.batch_size_vae,
                   help="batch_size, default is 100")
parse.add_argument("--lr", type=float, default=utils.lr_blackbox,
                   help="learning rate, default is 0.001")
parse.add_argument("--epoch", type=int, default=utils.epoch_vae,
                   help="epoch, default is 100")
parse.add_argument("--wd", type=float, default=utils.wd_vae,
                   help="weight decay, default is 1e-4")
parse.add_argument("--landa", type=float, default=utils.landa_vae,
                   help="landa for Kullback Leibler Divergence in loss, default is 1")

# read the argument
args = parse.parse_args()
NAME = args.name
INTERMEDIATE_DIM = args.intermediate_dim
LATENT_DIM = args.latent_dim
LR = args.lr
EPOCH = args.epoch
WD = args.wd
DROPOUT = args.dropout
BATCH_SIZE = args.batch_size
LANDA = args.landa

#load data
name_X = utils.name_preprocessed_data_X
name_y = utils.name_preprocessed_data_y
path_X = os.path.join(project_path,"data",name_X)
path_y = os.path.join(project_path,"data",name_y)
X_train = pd.read_csv(path_X,index_col=0).to_numpy()
y_train = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

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

X_train, X_test, y_train, y_test = split_and_normalize(X=X_train,y=y_train,seed=SEED,test_size=TEST_SIZE)

#create tensor dataset and dataloader
train_data = TensorDataset(X_train,y_train)
test_data = TensorDataset(X_test,y_test)

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

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
    
    def encode_mean_variance (self,x):
        x = self.encoder(x)
        mean, variance = self.latent_mean(x), self.latent_variance(x)
        return mean, variance
    
    def reparameterization (self,mean, variance):
        epsilon = torch.randn_like(variance)
        z = mean + variance*epsilon
        return z

    def forward(self,x):
        mean, variance = self.encode_mean_variance(x)
        z = self.reparameterization(mean,torch.exp(variance*0.5))
        x_ = self.decoder(z)
        return x_, mean, variance
    
#init model
model = MonteCarloDropoutVariationalAutoEncoder(origin_dim=ORIGIN_DIM,intermediate_dim=INTERMEDIATE_DIM,latent_dim=LATENT_DIM,dropout=DROPOUT)
summary(model)

#create custom loss
class Loss (nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x_: torch.tensor, x: torch.tensor, mean: torch.tensor,variance: torch.tensor, lamda = 1):
        BCE = nn.BCELoss() #reduction="sum"#nn.MSELoss(reduction="sum") #reduction="sum" # #nn.L1Loss(reduction="sum") nn.L1Loss(reduction="sum")nn.BCEWithLogitsLoss(reduction="sum") 
        reconstruction_loss = BCE(x_,x)
        KLD = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
        return reconstruction_loss + lamda * KLD, reconstruction_loss, KLD
    
# optimizer and loss
optimizer = torch.optim.Adam(params=model.parameters(),weight_decay=WD,lr=LR)
loss = Loss()

#train loop
for ep in range(EPOCH):
    #train process
    loss_total_train = 0
    loss_reconstruct_train = 0
    loss_kld_train = 0
    model.train()
    
    for batch_train,(X_train,y_train) in enumerate(train_loader):
        
        #forward pass
        X_pred_train,mean_train,variance_train = model(X_train)
        
        # print("X_ shape:", X_.shape)
        # print("mean:", mean.shape)
        # print("var:", variance.shape)
    
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
    print("epoch {}, loss total {}, loss reconstruct {}, loss kld {}".format(ep,loss_total_train, loss_reconstruct_train, loss_kld_train))
    
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
        print("epoch {}, loss total {}, loss reconstruct {}, loss kld {}".format(ep,loss_total_test, loss_reconstruct_test, loss_kld_test))
        print()

test = X_train[0]
print("test:", test)
y_train_out = model(test)
print("y_train_out:", y_train_out)