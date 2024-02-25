import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import utils
import pandas as pd

def preprocessing_compas(name_raw_dataset:pd.DataFrame, name = utils.name_preprocessed):
    """
    This function will preprocess the raw dataset COMPAS "compas-scores-two-years.csv" from https://github.com/domenVres/Robust-LIME-SHAP-and-IME contains 7214 samples and 52 features.
    We wil use only 7 fearures, they are: "age, "two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race". Two categorical features  "c_charge_degree" and "sex" will be one-hot-encoding encoded. 
    Fearure "race" will have value 1 if they are African-American else 0.
    Label are the value of "score_text" column. They have value 1 if the score are "High" else 0.
    Preprocessing does not contain the normalization.
    
    Parameters:
	    name_raw_dataset str): name of the COMPAS dataset "compas-scores-two-years.csv"
        name (str), defalt = "compas": preprocessed data will be save under the name_X.csv and name_y.csv
    Returns:
	    X (pd.DataFrame): preprocessed COMPAS dataset shape (6172,9)
        y (pd.DataFrame): label of COMPAS dataset shape(6172,1)
    """

    #load raw dataset to DataFrame
    current_path = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(current_path,name_raw_dataset)
    dataset = pd.read_csv(path_dataset,index_col=0)
    
    #only takes the value between -30 and 30 in the column 'days_b_screening_arrest'
    dataset = dataset[(dataset['days_b_screening_arrest'] <= 30) & (dataset['days_b_screening_arrest'] >= -30)]

    #create feature length_of_stay is the subtraction of feature c_jail_out and c_jail_in
    dataset["length_of_stay"] = (pd.to_datetime(dataset["c_jail_out"]) - pd.to_datetime(dataset["c_jail_in"])).dt.days

    #create data X for training contain feature age, two_year_recid, c_charge_degree, race, sex, priors_count, length_of_stay
    X = dataset[["age","two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race"]]
    
    #one hot encode the catorical feature, in this case is sex, c_charge_degree, True equal to 1, False equal to 0
    catorical_feature = ["c_charge_degree", "sex"]
    for fea in catorical_feature:
        one_hot = pd.get_dummies(X[fea],prefix=fea)
        for col in one_hot.columns:
            one_hot[col] = one_hot[col].astype(int)
        X = pd.concat((X.drop(fea,axis=1),one_hot),axis=1)
        
    # feature race will equal to 1 if African-American else 0
    race = [1 if r == "African-American" else 0 for r in X["race"]]
    X = X.drop("race",axis=1)
    X["race"] = race

    # create label from column 
    y = [0 if score == "High" else 1 for score in dataset["score_text"]]
    y = pd.DataFrame(y, columns = ["label"])
    
    #save the dataset
    path_x = os.path.join(current_path,name+"_X.csv")
    path_y = os.path.join(current_path,name+"_y.csv")
    X.to_csv(path_x)
    y.to_csv(path_y)
    print("Done!!!! preprocessed datasets are saved under the name {} and {}".format(name+"_X.csv",name+"_y.csv"))
    
    return X,y

if __name__ == "__main__":
    """
    Output of this file are the preprocessed COMPAS dataset and label. 
    Then the output should be yourname_X.csv and yourname_y.csv. Default of the --name is "compas" and our output files are compas_X.csv and compas_y.csv
    """
    # name preprocessed
    name = utils.name_preprocessed
    
    # preprocessing
    name_raw_dataset = utils.name_raw_data
    X, y = preprocessing_compas(name_raw_dataset=name_raw_dataset,name=name)
