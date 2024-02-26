from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import utils
import pandas as pd
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import torch
from tqdm import tqdm
from generator.perturb import pertube_data_generator
from generator.mcd_ae_custom import MonteCarloDropoutAutoEncoderCustom
from generator.mcd_vae import MonteCarloDropoutVariationalAutoEncoder
from adversarial.adversarial import AdversarialModel

#parameter lime
SEED = utils.seed
np.random.seed(SEED)
NUM_GEN_POINTS = utils.num_gen_points

#constant datasets
FEATURE_NAME = utils.feature_name
UNRELATED_FEATURE_INDEX = utils.unrelated_feature_index
CATEGORICAL_FEATURE_INDEX = utils.categorical_feature_index
RACE_FEATURE_INDEX = utils.race_feature_index
TEST_SIZE = utils.test_size
NUMERICAL_FEATURE_INDEX = utils.numerical_feature_index
    
#parameter adversarial
N_ESTIMATORS = utils.n_estimators_classifier
TEST_SIZE = utils.test_size
NAME_CLASSIFIER = utils.name_classifier

#parameter generator VAE
ORIGIN_DIM = utils.origin_dim_vae
NAME_VAE = utils.name_vae
NAME_GEN_VAE = utils.name_gen
INTERMEDIATE_DIM_VAE = utils.intermediate_dim_vae
LATENT_DIM_VAE = utils.latent_dim_vae
DROPOUT_VAE = utils.dropout_vae
NAME_SCALER = utils.name_scaler_vae

#parameter generator VAE
NAME_VAE_CUSTOM = utils.name_vae_custom
NAME_GEN_CUSTOM = utils.name_gen_custom
INTERMEDIATE_DIM_CUSTOM = utils.intermediate_dim_vae_custom
LATENT_DIM_CUSTOM = utils.latent_dim_vae_custom
DROPOUT_CUSTOM = utils.dropout_vae_custom 

#parameter for perturb generator
PERTURB_STD = utils.pertube_std
NAME_GEN_PERTURB = utils.name_pertube_gen

#parameter lime
TEXT_RESULT_NAME = utils.text_result_name

class LimeCustom:
    def __init__(self, blackbox: AdversarialModel, seed:int):
        """
        This class will have the misstion to calculate LIME from scratch. 
        
        Parameters:
            blackbox (AdversarialModel): blackbox model, which can used to predict the label of an instance x
            seed (int): random seed
        """
        self.black_box = blackbox
        self.seed = seed
        
    def get_lime (self, sampled_instance: np.array, name_gen:str,sigma = None):
        """
        In this method we will apply LIME to explane an instance using interpretable model
        
        Parameters:
            inter_model (LogisticRegression): Interpretable model. We only use Linear Regression model in sklearn
            sampled_instance (np.array): x and sampled instance from x, real x at index 0, fake x at the rest
            name_gen (str): Name of the generator that used for sampling fake data
        """
       
        #add uncorrelated features
        X = self.black_box.add_uncorrelated_feaatures(sampled_instance)
        instance = X[0]
        X, y = self.black_box.evaluate(name_gen,X)
        if len(np.unique(y)) == 1:
            return 1,np.zeros(shape=(1,X.shape[-1]))
        
        #calculate phi
        X, phi = self.weights(sampled_instance=X,sigma=sigma)

        #train the interpretable mode
        inter_model = LogisticRegression(random_state=self.seed)
        inter_model.fit(X,y, sample_weight=phi)

        #contribution each feature to the result
        weights = inter_model.coef_
        bias = inter_model.intercept_
        contribution = np.abs(instance.reshape(1,-1) * weights)
        
        return inter_model, contribution

    def weights(self, sampled_instance: np.array, sigma = None):
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
        real_instance = sampled_instance[0]
        d = np.sqrt(np.sum((real_instance - sampled_instance)**2,axis=1))
        phi = np.exp((-d**2) / (sigma**2))

        return sampled_instance, phi

#run localy in this file
if __name__ == "__main__":

    #load data
    name_X = utils.name_preprocessed_data_X
    name_y = utils.name_preprocessed_data_y
    path_X = os.path.join(project_path,"data",name_X)
    path_y = os.path.join(project_path,"data",name_y)
    X = pd.read_csv(path_X,index_col=0).to_numpy()
    y = pd.read_csv(path_y,index_col=0).to_numpy().flatten()

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=SEED)

    #load adversarial model (same roll as blackbox)
    adv = AdversarialModel(n_estimators=N_ESTIMATORS,name_classifier=NAME_CLASSIFIER,test_size=TEST_SIZE,seed=SEED,race_feature_index=RACE_FEATURE_INDEX,unrelated_feature_index=UNRELATED_FEATURE_INDEX)

    #init lime
    glime = LimeCustom(blackbox=adv,seed=SEED)
    
    #load generator
    #generator VAE
    generator_VAE = MonteCarloDropoutVariationalAutoEncoder(origin_dim=ORIGIN_DIM,intermediate_dim=INTERMEDIATE_DIM_VAE,latent_dim=LATENT_DIM_VAE,dropout=DROPOUT_VAE)
    model_path_VAE = os.path.join(project_path, "generator",NAME_VAE+".pth") 
    state_dict_VAE = torch.load(model_path_VAE)
    generator_VAE.load_state_dict(state_dict_VAE)

    #load scaler
    scaler_path = os.path.join(project_path,"generator",NAME_SCALER+".bin")
    scaler = joblib.load(scaler_path)
    
    #generator AE cusom
    generator_AE = MonteCarloDropoutAutoEncoderCustom(origin_dim=ORIGIN_DIM,intermediate_dim=INTERMEDIATE_DIM_CUSTOM,latent_dim=LATENT_DIM_CUSTOM,dropout=DROPOUT_CUSTOM)
    model_path_AE = os.path.join(project_path, "generator", NAME_VAE_CUSTOM+".pth") 
    state_dict_AE = torch.load(model_path_AE)
    generator_AE.load_state_dict(state_dict_AE)

    #name pretraied classifier
    name_gen = [NAME_GEN_VAE,NAME_GEN_CUSTOM, NAME_GEN_PERTURB]# ,,,]NAME_GEN_PERTURB NAME_GEN_CUSTOM NAME_GEN_VAE
    
    generator = [generator_VAE, generator_AE] #,]  generator_AE
  
    #use glime for MCD_VAE and AE_CUSTOM
    result = ""
    for name in tqdm(name_gen):
        for idx ,gen in enumerate(tqdm(generator)):
            contri = []
            for x in tqdm(X_test):
                
                #normalized x
                x_scaled = scaler.transform(x.reshape(1,-1)).ravel()
                
                #sampled x
                x_sampled = gen.generate_data(x = x_scaled,num_gen=NUM_GEN_POINTS,scaler=scaler,categorical_feature_index=CATEGORICAL_FEATURE_INDEX, return_label = False)

                #use glime to calculate the contribution of significan feature value
                _, contribution = glime.get_lime(sampled_instance=x_sampled,name_gen=name)
                argmax_contribution = np.argmax(contribution,axis=1)
                contri.append(argmax_contribution[0])
            
            contri = np.array(contri)
            accuracy = np.sum(contri == RACE_FEATURE_INDEX) / len(contri)
            text = "Result of adversarial trained on {} data evaluated on {} data: Accuracy {} \n".format(name,name_generator[idx],accuracy)
            result = result + text

    #use glime for perturb generator
    for name in tqdm(name_gen):
        contri = []
        for x in tqdm(X_test):
            #sampled x
            x_sampled = pertube_data_generator(X = X,x=x,numerical_feature_index=NUMERICAL_FEATURE_INDEX,num_pertube=NUM_GEN_POINTS,pertube_std=PERTURB_STD, return_label = False)
            
            #use glime to calculate the contribution of significan feature value
            _, contribution = glime.get_lime(sampled_instance=x_sampled,name_gen=name)
            argmax_contribution = np.argmax(contribution,axis=1)
            contri.append(argmax_contribution[0])

        contri = np.array(contri)
        accuracy = np.sum(contri == RACE_FEATURE_INDEX) / len(contri)
        text = "Result of adversarial trained on {} data evaluated on {} data: Accuracy {} \n".format(name,"perturb generator",accuracy)
        result = result + text
        
#save the result as text
current_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(current_path,TEXT_RESULT_NAME+".txt")
if not os.path.exists(result_path):
    with open(result_path,"w") as file:
        file.write(result)
        