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
RACE_FEATURE_INDEX = utils.race_feature_index
UNRELATED_FEATURE_INDEX = utils.unrelated_feature_index

class AdversarialModel:
    def __init__(self,n_estimators:int,name_classifier:str, test_size:float, seed:int, race_feature_index:int,unrelated_feature_index:int):
        """
        Init (hyper)parameter of the adversarial model
        
    	Parameters:
            n_estimators (int): number of the trees in random forest.
            test_size (float): test size in train test split
            name_classifier (str): name of the classifier
            seed (int): randomm seed
            race_feature_index (int): index of race feature (bias feature)
            unrelated_feature_index (list): index of unrelated feature
        """
        self.n_estimators = n_estimators
        self.name_classifier = name_classifier
        self.test_size = test_size
        self.seed = seed
        self.race_feature_index = race_feature_index
        self.unrelated_feature_index = unrelated_feature_index
        
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
        xtrain, xtest, ytrain, ytest = train_test_split(all_x, all_y, test_size=self.test_size,random_state=self.seed)
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

    def bias_model (self, X_bias: np.array):
        """
        Bias model return the label (0) if instance has the feature race equal to African-Black (0), else 1
        
    	Parameters:
            X_bias (np.array): unseen data that classified as real samples (label 1)
        Returns:
            X_bias (np.array): unseen data
            y_bias (np.array): 0 if race equal to African-Black else 1
        """
        #get label
        y_bias = np.where(X_bias[:,self.race_feature_index]==0,0,1)
        
        return X_bias, y_bias
        
    def unbias_model(self, X_unbias: np.array):
        """
        Unbias model return the label (0) if instance has the feature race equal to African-Black (0), else 1
            
        Parameters:
            X_unbias (np.array): unseen data that classified as fake samples (label 0)
        Returns:
            X_bias (np.array): unseen data
            y_unbias (np.array): xor logic of y unrelated 1 and y unrelated 2, y unrelated equal to the value of the unrelated feature
        """
        # Unbiased model with two unrelated features
        y_unrelated_1 = np.where(X_unbias[:, self.unrelated_feature_index[0]] == 1, 1, 0)
        y_unrelated_2 = np.where(X_unbias[:, self.unrelated_feature_index[1]] == 1, 1, 0)
        y_unbias = np.logical_xor(y_unrelated_1, y_unrelated_2).astype(int)
    
        return X_unbias, y_unbias
        
    def evaluate (self, input_data:str, X:np.array):
        """
        Load pretrained model on input_name dataset and evaluate unseen data to check if the sample real hay fake. Real sample goes to bias model, which will get the label 0 if the feature race equal to African-Black. Fake samples will go to unbias model, which will get logic xor of value of feature unrelated 1 and 2
        
    	Parameters:
            input_data (str): name of the generated data in directory /generator.
            name_classifier (str): name of the classfier
            X (np.array): unseen dataset
        Returns:
            X (torch.tensor): concatenate of X_bias and X_unbias 
            y (torch.tensor): concatenate of y_bias and y_unbias 
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
        
        #sort to bias and unbias model
        X_bias = X[y_pred==1]
        X_unbias = X[y_pred==0]
        
        # put the predicted real samples to bias mode and fake samples to unbias model
        X_bias, y_bias = self.bias_model(X_bias=X_bias)
        print("X_bias:", y_bias)
        print("X_bias shape:", y_bias.shape)
        X_unbias, y_unbias = self.unbias_model(X_unbias=X_unbias)
        print("X_unbias:", y_unbias)
        print("X_unbias shape:", y_unbias.shape)
        
        #concat back to 1 datasets
        X = np.concatenate((X_bias,X_unbias),axis=0)
        print("X:", X)
        y = np.concatenate((y_bias,y_unbias))
        
        return X, y

    
#run localy in this file
if __name__ == "__main__":
    
    #init adversarial model
    adv = AdversarialModel(n_estimators=N_ESTIMATORS,name_classifier=NAME_CLASSIFIER,test_size=TEST_SIZE,seed=SEED,race_feature_index=RACE_FEATURE_INDEX,unrelated_feature_index=UNRELATED_FEATURE_INDEX)

    #load data
    import pandas as pd
    name_X = utils.name_preprocessed_data_X
    name_y = utils.name_preprocessed_data_y
    path_X = os.path.join(project_path,"data",name_X)
    path_y = os.path.join(project_path,"data",name_y)
    X = pd.read_csv(path_X,index_col=0).to_numpy()
    y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()
    
    # evaluate 
    test1, test2 = adv.evaluate(input_data=NAME_GEN_CUSTOM,X=X)
    print("test1 shape:", test1.shape)
    print("test1 shape:", test2.shape)
    
    
    #print("test shape:", test.shape)
    #print("test:", test)
    #print("test unique:", np.unique(test,return_counts=True))
    