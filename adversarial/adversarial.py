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
    def __init__(self):
        pass 
    
    def train_classifier_on_sampled_data (self, input_data:str,n_estimators:int, seed:int, test_size:float, name_classifier):
        """
        Train the RandomForestClassifier on the sampled data. 
        
    	Parameters:
    	----------
    	input_data (str): path of the generated data in directory /generator. Sampled data is of type numpy.array, of shape (number of instances, number of original and sampled values for each instance, number of features))
        n_estimators (int): number of the trees in random forest.
        seed (int): randomm seed
        test_size (float): test size in train test split
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
        xtrain, xtest, ytrain, ytest = train_test_split(all_x, all_y, test_size=test_size)
        xtrain_flatten = xtrain.reshape(-1,11)
        ytrain_flatten = ytrain.reshape(-1,1).ravel()
        xtest_flatten = xtest.reshape(-1,11)
        ytest_flatten = ytest.reshape(-1,1).ravel()
        
        # Train the Random Forest Classifier
        sampled_data_identifier = RandomForestClassifier(n_estimators=n_estimators, random_state = seed)
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
        name_model = name_classifier + "_" + input_data + ".pkl"
        model_path = os.path.join(current_path,name_model)
        if not os.path.exists(model_path):
            joblib.dump(sampled_data_identifier,model_path)

#run localy in this file
if __name__ == "__main__":
    
    #init adversarial model
    adv = AdversarialModel()
    
    # train classfider on generated data from MCD VAE
    adv.train_classifier_on_sampled_data(NAME_GEN_VAE,N_ESTIMATORS,SEED,TEST_SIZE,NAME_CLASSIFIER)
    
    # train classfider on generated data from MCD AE custom
    adv.train_classifier_on_sampled_data(NAME_GEN_CUSTOM,N_ESTIMATORS,SEED,TEST_SIZE,NAME_CLASSIFIER)
    
    # train classfider on generated data from perturbation
    adv.train_classifier_on_sampled_data(NAME_GEN_PERTURB,N_ESTIMATORS,SEED,TEST_SIZE,NAME_CLASSIFIER)
    
    