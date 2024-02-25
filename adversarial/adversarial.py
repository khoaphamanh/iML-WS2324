import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
import utils

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib

#fix parameter and constante
SEED = utils.seed
np.random.seed(SEED)
NAME_GEN_VAE = utils.name_gen
NAME_GEN_CUSTOM = utils.name_gen_custom
NAME_GEN_PERTURB = utils.name_pertube_gen
N_ESTIMATORS = utils.n_estimators_classifier
TEST_SIZE = utils.test_size
NAME_CLASSIFIER = utils.name_classifier

class AdversarialModel:
    def __init__(self,n_estimators:int,name_classifier:str, test_size:float, seed:int):
        """
        Init (hyper)parameter of the adversarial model
        
    	Parameters:
            n_estimators (int): number of the trees in random forest.
            test_size (float): test size in train test split
            name_classifier (str): name of the classifier
            seed (int): randomm seed
        """
        self.n_estimators = n_estimators
        self.name_classifier = name_classifier
        self.test_size = test_size
        self.seed = seed
    
    def train_classifier_on_sampled_data (self, input_data:str):
        """
        Train the RandomForestClassifier on the sampled data and save it.
        
    	Parameters:
            input_data (str): name of the generated data in directory /generator. Sampled data is of type numpy.array, of shape (number of instances, number of original and sampled values for each instance, number of features))
        """

        #load data
        input_data_path = os.path.join(project_path,"generator",input_data+".npy")
        all_x_y = np.load(input_data_path)
        all_x = all_x_y[:,:,:-1]
        all_y = all_x_y[:,:,-1:]
        
        # Generate unrelated columns
        unrelated_column_one = np.random.choice([0, 1], size=(all_x.shape[0], 3, 1))
        unrelated_column_two = np.random.choice([0, 1], size=(all_x.shape[0], 3, 1))
        all_x = np.concatenate((all_x, unrelated_column_one, unrelated_column_two), axis=2)
        
        # Split the data into train and test dataset
        xtrain, xtest, ytrain, ytest = train_test_split(all_x, all_y, test_size=self.test_size)
        xtrain_flatten = xtrain.reshape(-1,11)
        ytrain_flatten = ytrain.reshape(-1,1).ravel()
        xtest_flatten = xtest.reshape(-1,11)
        ytest_flatten = ytest.reshape(-1,1).ravel()
        
        # Train the Random Forest Classifier
        sampled_data_identifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state = self.seed)
        sampled_data_identifier.fit(xtrain_flatten, ytrain_flatten)
        
        # print performance
        print("Classifier trained on {}.npy".format(input_data))
        ytrain_pred = sampled_data_identifier.predict(xtrain_flatten)
        accuracy_train = accuracy_score (y_pred=ytrain_pred, y_true=ytrain_flatten)
        f1_train = f1_score(y_pred=ytrain_pred, y_true=ytrain_flatten)

        print("accuracy_train:", accuracy_train)
        print("f1_train:", f1_train)
        
        ytest_pred = sampled_data_identifier.predict(xtest_flatten)
        accuracy_test = accuracy_score (y_pred=ytest_pred, y_true=ytest_flatten)
        f1_test = f1_score(y_pred=ytest_pred, y_true=ytest_flatten)
        print("accuracy_test:", accuracy_test)
        print("f1_test:", f1_test)
        print()
        
        #save the model
        name_model = self.name_classifier + "_" + input_data + ".pkl"
        model_path = os.path.join(current_path,name_model)
        if not os.path.exists(model_path):
            joblib.dump(sampled_data_identifier,model_path)

        #return sampled_data_identifier
    
    def evaluated_on_classifier (self, input_data:str, X:np.array):
        """
        Load pretrained model on input_name dataset and evaluate unseen data 
        
    	Parameters:
            input_data (str): name of the generated data in directory /generator.
            name_classifier (str): name of the classfier
            X (np.array): unseen dataset
        Returns:
            X_train (torch.tensor): train data
            X_test (torch.tensor): test data
            y_train (torch.tensor): train label
            y_test (torch.tensor): test label
        """
        #load model
        name_model = self.name_classifier + "_" + input_data + ".pkl"
        model_path = os.path.join(current_path,name_model)
        if not os.path.exists(model_path):
            self.train_classifier_on_sampled_data(input_data=input_data)
        
        model = joblib.load(model_path)
        
        # Generate unrelated columns
        unrelated_column_one = np.random.choice([0, 1], size=(X.shape[0], 1))
        unrelated_column_two = np.random.choice([0, 1], size=(X.shape[0], 1))
        X = np.column_stack((X,unrelated_column_one,unrelated_column_two))
        
        # predict real fake sample
        y_pred = model.predict(X)
        
        return y_pred
        
    def bias_model (self, X:np.array):
        """
        Bias model return the label (0) if instance has the feature race equal to African-Black (0), else 1
        
    	Parameters:
            data (np.array): data
        """

def one_hot_encode(y):
    y_hat_one_hot = np.zeros((len(y), 2))
    y_hat_one_hot[np.arange(len(y)), y] = 1
    return y_hat_one_hot

# Bias (racist) model
class biased_model_f:
    # Decision rule: classify negatively (0) if race is black, 
    # and positively (1) if race is others
    def predict(self,X):
        return np.array([0 if x[8] == 0 else 1 for x in X])
    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))
    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

#run localy in this file
if __name__ == "__main__":
    
    #init adversarial model
    adv = AdversarialModel(n_estimators=N_ESTIMATORS,name_classifier=NAME_CLASSIFIER,test_size=TEST_SIZE,seed=SEED)

    #load data
    import pandas as pd
    name_X = utils.name_preprocessed_data_X
    name_y = utils.name_preprocessed_data_y
    path_X = os.path.join(project_path,"data",name_X)
    path_y = os.path.join(project_path,"data",name_y)
    X = pd.read_csv(path_X,index_col=0).to_numpy()
    y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()
    
    # evaluate 
    test = adv.evaluated_on_classifier(input_data=NAME_GEN_CUSTOM,X=X)
    print("test shape:", test.shape)
    print("test:", test)
    print("test unique:", np.unique(test,return_counts=True))
    