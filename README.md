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
The raw data is downloaded in the [github](https://github.com/domenVres/Robust-LIME-SHAP-and-IME/tree/master/Fooling-LIME-SHAP/data) [[1]](#1) of the article "Better sampling in explanation methods can prevent dieselgate-like deception" from D Vres, et al [[2]](#2). The raw data is saved in the /data folder under the name compas-scores-two-years.csv. Initially we will preprocess this data. Only 7 features will be selected including "age, "two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race". Label will be selected from the "score_text" column. File preprocessing.py perform this task. The output will be 2 files compas_X.csv, containing the processed features and compas_y.csv is the label. Users can run this file again with the syntax from argparse:
```bash
python preprocessing.py --name "yourname" 
```
The --name argument will determine the name of the processed data file. Our default value is "compas". For more details, please go to the preprocessing.py file and read the description.

## Black-box models
To apply LIME, we first need to have a black-box model trained on the training data. Blackbox models will be implemented in the /blackbox folder. The first black-box model is an artificial neural network created from the neural_network.py file under the name neural_network.pth, its metrics can be visualized in the metrics_neural_network.png file. Users can run this file again with the syntax from argparse:
```bash
python neural_network.py --name "yourname" --hidden_size "your_hidden_size" --hidden_layers "your_hidden_layers" --lr "your_lr" --epoch "your_epoch" --wd "your_wd" --name_metrics "your_name_metrics"
```
Default of the arguments are saved in ultis.py:

    --name: "neural_network" and our output files is neural_network.pth
    --hidden_size: 16
    --hidden_layers: 10
    --lr: 0.01
    --epoch: 500
    --wd: 1e-5
    --name_metrics: "metrics_neural_network" and the metrics of the model is saved under the name metrics_neural_network.png

The second black-box model is the random forest created from the random_forest.py file under the name random_forest.pkl. Users can run this file again with the syntax from argparse:
```bash
python random_forest.py --name "yourname" --n_estimators "your_n_estimators"
```
Default of the arguments are saved in ultis.py:

    --name: "random_forest" and our output files is random_forest.pkl
    --n_estimators: 100

## Generator models
To get samples with the same distribution of the data, we will use Monte Carlo Dropout Variational Auto Encoder (MCD VAE) [[2]](#2) as shown in the paper [[1]](#1). In the mcd_vae.py file, we take the model structure and hyperparameters directly from the paper. The output of this file is a generator (MCD VAE) saved as mcd_vae.pth, generated data as gen_data.npy. The performance of the model can be visualized with metrics_vae.png. Users can run this file again with the syntax from argparse:
```bash
python mcd_vae.py --name "yourname" --intermediate_dim "your_intermediate_dim" --latent_dim "your_latent_dim" --dropout "your_dropout" --batch_size "your_batch_size" --lr "your_lr" --epoch "your_epoch" --wd "your_wd" --landa "your_landa" --num_gen "your_num_gen" --name_gen "your_name_gen" --name_metrics "your_name_metrics"
```
Default of the arguments are saved in ultis.py:

    --name: "mcd_vae" and our output files is mcd_vae.pth
    --intermediate_dim: 8
    --latent_dim: 4
    --dropout: 0.3
    --batch_size: 100
    --lr: 0.001
    --epoch: 100
    --wd: 1e-4
    --landa: 1
    --num_gen: 100
    --name_gen: "gen_data" and our generated data files is gen_data.npy shape (n_samples, num_gen+1, n_feature + 1)
    --name_metrics: "metrics_ae_custom" and the metrics of the model is saved under the name metrics_vae.png

We noticed that, as the article says, MCD VAE doesn't really have good performance, or in other words, there is a visible difference to the naked eye between a sample in datasets and a generated sample. Therefore, we customized a model that includes more linear layers, more ReLU and Dropout activation functions, and uses an Auto Encoder structure instead of VAE to increase the generator's performance. This can be expressed through the mcd_ae_custom.py file. The output of this file includes pretrained generator mcd_ae_custom.pth, generated data gen_data_custom.npy and metrics curves metrics_vae.png. Users can run this file again with the syntax from argparse:
```bash
python mcd_vae.py --name "yourname" --intermediate_dim "your_intermediate_dim" --latent_dim "your_latent_dim" --dropout "your_dropout" --batch_size "your_batch_size" --lr "your_lr" --epoch "your_epoch" --wd "your_wd" --num_gen "your_num_gen" --name_gen "your_name_gen" --name_metrics "your_name_metrics"
```

Default of the arguments are saved in ultis.py:

    --name: "mcd_ae_custom" and our output files is mcd_ae_custom.pth
    --intermediate_dim: 64
    --latent_dim: 2
    --dropout: 0.05
    --batch_size: 100
    --lr: 0.01
    --epoch: 100
    --wd: 1e-5
    --num_gen: 100
    --name_gen: "gen_data_custom" and our generated data files is gen_data_custom.npy shape (n_samples, num_gen+1, n_feature + 1)
    --name_metrics: "metrics_ae_custom" and the metrics of the model is saved under the name metrics_ae_custom.png

It should be noted that the two files gen_data.npy and gen_data_custom.npy both have shape (n_samples, num_gen+1, n_feature + 1), more specifically (6172, 101, 10). For each real sample (label 1) in n_samples (6172 samples) will be generated 100 fake samples (label 0). The last column of this array will be the label (1 if the sample is real, 0 if it is fake).

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

<a id="2">[3]</a>
Kristian Miok, Deng Nguyen-Doan (2019)
Generating Data using Monte Carlo Dropout
https://arxiv.org/pdf/1909.05755v2.pdf