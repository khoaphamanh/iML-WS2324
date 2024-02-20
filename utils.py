"""
This file contains the parameters that are used in this project
"""

# Random seed 
seed = 123454321

#datasets
name_preprocessed = "compas"
name_raw_data = "compas-scores-two-years.csv"
name_preprocessed_data_X = "compas_X.csv"
name_preprocessed_data_y = "compas_y.csv"

#Parameter for black-box model as neuralnetwork
test_size = 0.1
input_size = 9
hidden_size = 16
hidden_layers = 10
output_size = 1
lr_blackbox = 0.01
epoch_blackbox = 2000
wd_blackbox = 1e-5
name_blackbox_nn = "neural_network"

#Parameter for black-box model as neuralnetwork
name_blackbox_rf = "random_forest"
n_estimators_blackbox = 100

#Parameter for generator as MCD_VAE
name_vae = "mcd_vae"
origin_dim_vae = 9
intermediate_dim_vae = 8
latent_dim_vae = origin_dim_vae // 2
epoch_vae = 100
dropout_vae = 0.3
batch_size_vae = 100
lr_vae = 0.001
wd_vae = 1e-4
landa_vae = 1