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

#Parameter for generator as MCD_VAE in paper
name_vae = "mcd_vae"
origin_dim_vae = 9
intermediate_dim_vae = 8
num_gen = 100
batch_size_vae = 100

latent_dim_vae = origin_dim_vae // 2
epoch_vae = 100
dropout_vae = 0.3
batch_size_vae = 100
lr_vae = 0.001
wd_vae = 1e-4
landa_vae = 1
name_gen = "gen_data"
name_metrics_vae= "metrics_vae"

#Parameter for generator as MCD_VAE custom
name_vae_custom = "mcd_ae_custom"
intermediate_dim_vae_custom = 64
latent_dim_vae_custom = 2
epoch_vae_custom = 100
dropout_vae_custom = 0.05
lr_vae_custom = 0.01
wd_vae_custom = 1e-5
landa_vae_custom = 0.1
name_gen_custom = "gen_data_custom"
name_metrics_custom = "metrics_ae_custom"