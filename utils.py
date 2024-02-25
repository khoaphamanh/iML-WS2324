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
feature_name = ['age', 'two_year_recid', 'priors_count', 'length_of_stay', 'c_charge_degree_F', 'c_charge_degree_M', 'sex_Female', 'sex_Male', 'race']
categorical_feature = ["two_year_recid", "c_charge_degree_F", "c_charge_degree_M" ,"sex_Female","sex_Male","race"]
categorical_feature_index = [1, 4, 5, 6, 7, 8 ]
numerical_feature = ["age", "priors_count", "length_of_stay", "unrelated_column_one", "unrelated_column_two"]
numerical_feature_index = [0, 2, 3]
unrelated_feature_index = [9, 10]
race_feature_index = 8

#Parameter for generator over all
test_size = 0.1
origin_dim_vae = 9
intermediate_dim_vae = 8
num_gen = 2
batch_size_vae = 100

#Parameter for generator as MCD_VAE in paper
latent_dim_vae = origin_dim_vae // 2
epoch_vae = 100
dropout_vae = 0.3
batch_size_vae = 100
lr_vae = 0.001
wd_vae = 1e-4
landa_vae = 1
name_gen = "gen_data"
name_vae = "mcd_vae"
name_metrics_vae= "metrics_vae"
name_scaler_vae = "scaler"

#Parameter for generator as MCD_AE custom
name_vae_custom = "mcd_ae_custom"
intermediate_dim_vae_custom = 64
latent_dim_vae_custom = 2
epoch_vae_custom = 100
dropout_vae_custom = 0.05
lr_vae_custom = 0.01
wd_vae_custom = 1e-5
name_gen_custom = "gen_data_custom"
name_metrics_custom = "metrics_ae_custom"

#Parameter for generator as pertube
pertube_std = 0.3
name_pertube_gen = "gen_data_perturb"

#Parameter for adversarial/classifier
n_estimators_classifier = 100
name_classifier = "classfier"

#lime
num_gen_points = 5000
text_result_name = "result"

