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
The raw data is downloaded in the [github](https://github.com/domenVres/Robust-LIME-SHAP-and-IME/tree/master/Fooling-LIME-SHAP/data) [[1]](#1) of the [article](https://arxiv.org/pdf/2101.11702.pdf) "Better sampling in explanation methods can prevent dieselgate-like deception" from D Vres, et al [[2]](#2). The raw data is saved in the /data folder under the name compas-scores-two-years.csv. Initially we will preprocess this data. Only 7 features will be selected including "age, "two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race". Label will be selected from the "score_text" column. File preprocessing.py perform this task. The output will be 2 files compas_X.csv, containing the processed features and compas_y.csv is the label. Users can run this file again with the syntax:
```bash
python preprocessing.py --name "yourname" 
```
The --name argument will determine the name of the processed data file. Our default value is "compas". For more details, please go to the preprocessing.py file and read the description.

## Blackbox models
To apply LIME, we first need to have a blackbox model trained on the training data. Blackbox models will be implemented in the /blackbox folder. The first blackbox model is an artificial neural network created from the neural_network.py file under the name neural_network.pth, its metrics can be visualized in the metrics_neural_network.png file. Users can run this file again with the syntax:
```bash
python neural_network.py --name "yourname" --hidden_size "your_hidden_size" --hidden_layers "your_hidden_layers" --lr "your_lr" --epoch "your_epoch" --wd "your_wd" --name_metrics "your_name_metrics"
```
Default of the arguments:

    + --name: "neural_network" and our output files is neural_network.pth
    + --hidden_size: 16
    + --hidden_layers: 10
    + --lr: 0.01
    + --epoch: 2000
    + --wd: 1e-5
    + --name_metrics: "metrics_neural_network" and the metrics of the model is saved under the name metrics_neural_network.png

## References

<a id="1">[1]</a>
Vreš, D.
Robust-LIME-SHAP-and-IME
https://github.com/domenVres/Robust-LIME-SHAP-and-IME/tree/master

<a id="2">[2]</a>
Vreš, D. and Robnik Šikonja, M. (2020)
Better sampling in explanation methods can prevent dieselgate-like deception
Submitted to International Conference on Learning Representations
https://arxiv.org/pdf/2101.11702.pdf