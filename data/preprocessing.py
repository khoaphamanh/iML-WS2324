import os
import pandas as pd
import argparse

def preprocessing_compas(name_raw_dataset:pd.DataFrame, name = "compas",save = True):
    """
    This function will preprocess the raw dataset COMPAS compas-scores-two-years.csv from https://github.com/domenVres/Robust-LIME-SHAP-and-IME contains 7214 samples and 52 features
    
    Parameters:
	    name_raw_dataset str): name of the COMPAS dataset "compas-scores-two-years.csv"
        save (boolean), defaule = True: preprocessed data is saved as .csv
        name (str), defalt = "compas_preprocessed"
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
    if save:
        X.to_csv(name+"_X.csv")
        y.to_csv(name+"_y.csv")
        
    return X,y

if __name__ == "__main__":
    """
    Output of this file are the preprocessed COMPAS dataset and label. User can run this file with syntax from argparse to test our function:
    
        python preprocessing.py --name "yourname" --save True
        
    Then the output should be yourname_X.csv and yourname_y.csv. Default of the --name and --save are "compas" and True and our output files are compas_X.csv and compas_y.csv
    """
    # create a argpase
    parse = argparse.ArgumentParser()

    # add argument
    parse.add_argument("--name", type=str, default="compas")
    parse.add_argument("--save", type=bool, default=True)
    
    # read the argument
    args = parse.parse_args()
    name = args.name
    save = args.save
    
    # preprocessing
    name_raw_dataset = "compas-scores-two-years.csv"
    preprocessed_dataset = preprocessing_compas(name_raw_dataset=name_raw_dataset,name=name,save=save)


