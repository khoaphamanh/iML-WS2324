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
from sklearn.metrics import f1_score

"""
Output of this file is the pretrained blackbox model as neural network on COMPAS dataset. The task is binary classification, we use Adam optimizer and BCE Loss to train model. User can run this file with syntax from argparse to test our file:

    python neural_network.py --name "yourname" --hidden_size "your_hidden_size" --hidden_layers "your_hidden_layers" --lr "your_lr" --epoch "your_epoch" --wd "your_wd"
    
Then the output should be yourname.pth. Default of the arguments:

--name: "neural_network" and our output files is neural_network.pth
--hidden_size: 16
--hidden_layers: 10
--lr: 0.01
--epoch: 2000
--wd: 1e-5
"""

#fix parameter and constante
SEED = utils.seed
torch.manual_seed(SEED)
INPUT_SIZE = utils.input_size 
TEST_SIZE = utils.test_size
OUTPUT_SIZE = utils.output_size 

# add argument
parse = argparse.ArgumentParser()
parse.add_argument("--name", type=str, default=utils.name_blackbox_nn,
                   help="name of the pretrained blackbox model name.pth, default is 'neural_network'")
parse.add_argument("--hidden_size", type=int, default=utils.hidden_size,
                   help="number of hidden units in one hidden layer in model, default is 16")
parse.add_argument("--hidden_layers", type=int, default=utils.hidden_layers,
                   help="number of hidden layers in model, default is 10")
parse.add_argument("--lr", type=float, default=utils.lr_blackbox,
                   help="learning rate, default is 0.01")
parse.add_argument("--epoch", type=int, default=utils.epoch_blackbox,
                   help="epoch, default is 2000")
parse.add_argument("--wd", type=float, default=utils.wd_blackbox,
                   help="weight decay, default is 1e-5")

# read the argument
args = parse.parse_args()
NAME = args.name
HIDDEN_SIZE = args.hidden_size
HIDDEN_LAYERS = args.hidden_layers 
LR = args.lr
EPOCH = args.epoch
WD = args.wd

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
        x = self.output_layer(x)
        return x

#init model
model = BlackBoxModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS,output_size=OUTPUT_SIZE)
summary(model)

#optimizer and loss
count = torch.bincount(y_train)
pos_weight = count[0] / count[1]
print("pos_weight:", pos_weight)
loss = nn.BCEWithLogitsLoss(weight=pos_weight) #pos_weight=pos_weight
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

#save the model
state_dict = model.state_dict()
current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_path,NAME+".pth")
if not os.path.exists(model_path):
    torch.save(state_dict,model_path)
print("Done!!! Your model is saved under name {}".format(NAME+".pth"))
