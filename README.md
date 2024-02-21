[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)
# IML WS 23 Project
## Installation
To run the code in this project, Anaconda should installed. In this project we code entirely on Linux machine. Libraries and frameworks can be installed in a virtual environment through the following commands in the terminal:
```bash
conda env create -f environment.yml
```
```bash
source activate iML
```
## Preprocessing 
The raw data is downloaded in the [github](https://github.com/domenVres/Robust-LIME-SHAP-and-IME/tree/master) of the [article](https://arxiv.org/pdf/2101.11702.pdf) "Better sampling in explanation methods can prevent dieselgate-like deception". The raw data is saved in the data folder under the name compas-scores-two-years.csv. Initially we will preprocess this data. Only 7 features will be selected including "age, "two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race". Label will be selected from the "score_text" column. File preprocessing.py perform this task. The output will be 2 files compas_X.csv, containing the processed features and compas_y.csv is the label. Users can run this file again with the syntax:
```bash
python preprocessing.py --name "yourname" 
```
The --name argument will determine the name of the processed data file. Our default value is "compas". For more details, please go to the preprocessing.py file and read the description.

## Blackbox models
git add 