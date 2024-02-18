import os
import numpy as np
import pandas as pd


def preprocessing_compas(raw_dataset:pd.DataFrame):
    """
    This function will preprocess the raw dataset COMPAS compas-scores-two-years.csv from https://github.com/domenVres/Robust-LIME-SHAP-and-IME
    
    Parameters
	----------
	raw_dataset: pd.DataFrame
 
    Returns
	----------
	pd.DataFrame
    """

    return 


current_path = os.path.dirname(os.path.abspath(__file__))
name_dataset = "compas-scores-two-years.csv"
path_dataset = os.path.join(current_path,name_dataset)
dataset = pd.read_csv(path_dataset)
print("dataset type:", type(dataset))


