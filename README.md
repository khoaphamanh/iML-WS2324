[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/k0DpfI3g)
# IML WS 23 Project

![gLIME](/glime.svg)

The COMPAS dataset undergoes processing through generators (perturbation and MCD-VAE) to produce sampled datasets. Three sampled datasets are created: one via perturbation technique, one through MCD-VAE with parameters from the original dataset, and one with custom parameters. These datasets are then divided into train and evaluate sets. The train datasets are utilized to train the adversarial models, resulting in trained adversarial models. The evaluate datasets serve as input for the trained adversarial models. Subsequently, the output is passed through LIME to determine the features with the most significant contribution.

## Installation
To run the code in this project, Anaconda should installed. In this project we code entirely on Linux machine. Libraries and frameworks can be installed in a virtual environment through the following commands in the terminal:
```bash
conda env create -f environment.yml
```
```bash
source activate iML
```
## Preprocessing 
The raw data is downloaded in the [github](https://github.com/domenVres/Robust-LIME-SHAP-and-IME/tree/master/Fooling-LIME-SHAP/data) [[1]](#1) of the article "Better sampling in explanation methods can prevent dieselgate-like deception" from D Vres, et al [[2]](#2). The raw data is saved in the /data folder under the name compas-scores-two-years.csv. Initially we will preprocess this data. Only 7 features will be selected including "age, "two_year_recid", "c_charge_degree", "sex", "priors_count", "length_of_stay", "race". Label will be selected from the "score_text" column. File preprocessing.py perform this task and the output will be: 

    - compas_X.csv contains the processed features.
    - compas_y.csv contains the processed labels.

## Generator models
To get samples with the same distribution of the data, we will use Monte Carlo Dropout Variational Autoencoder (MCD VAE) [[2]](#2), which is stored in directory /generator as shown in the paper [[1]](#1). In the mcd_vae.py file, we take the model structure and hyperparameters directly from the paper. The output of this file is a generator (MCD VAE) saved as mcd_vae.pth, generated data as gen_data.npy. The performance of the model can be visualized with metrics_vae.png. The output of the genrators are: 

    - mcd_vae.pth is the pretrained MCD-VAE generators.
    - gen_data.npy contains the generated dataset.
    - metrics_vae.png shows the performance of the model.

Default hyperparameters are saved in ultis.py:

    - intermediate_dim: 8
    - latent_dim: 4
    - dropout: 0.3
    - batch_size: 100
    - lr: 0.001
    - epoch: 100
    - wd: 1e-4
    - landa: 1

We noticed that, as the article says, MCD VAE doesn't performs well, or in other words, there is a visible difference to the naked eye between a sample in datasets and a generated sample. Therefore, we customized a model that includes more linear layers, more activation functions ReLU and Dropout layers, and uses an Autoencoder [[4]](#4) structure instead of VAE to increase the generator's performance. This can be expressed through the mcd_ae_custom.py file. The output of this file includes pretrained generator mcd_ae_custom.pth, generated data gen_data_custom.npy and metrics curves metrics_vae.png. The output of the genrators are: 

    - mcd_ae_custom.pth is the pretrained MCD-AE generators with custom parameters and model structure.
    - gen_data_custom.npy contains the generated dataset from custom generator.
    - metrics_vae_custom.png shows the performance of the model.
    - scaler.bin is scaler in normalization step before training the generator.
    
Default of the arguments are saved in ultis.py:

    - intermediate_dim: 64
    - latent_dim: 2
    - dropout: 0.05
    - batch_size: 100
    - lr: 0.01
    - epoch: 100
    - wd: 1e-5

Moreover, file perturb.py will perturb data based on adding Gaussian noise to the neighborhood of each instance in the dataset. The output of this file is:

    - gen_data_perturb.npy contains perturbed data
    
We should notice that with each instance from original dataset (real sample), we have two sampled instances (fake samples). Real sample has label of 1 while fake samples have labels of 0.

## Adversarial Model
![Adversarial Model](/adversarial.png)
Generated datasets, which are stored in directory /generator, are split to train and evaluate datasets. Train dataset is passed to the adversarial model to train the decision model (Random Forest Classifier). If the decision model decides that the instance is from original distribution, the biased model will be used. Otherwise, the unbiased model will be used when the instance is decided to be a sampled instance. The output of the biased model is based on sensitive feature "race", while the output of the unbiased model is biased on unrelated features. In directory /adversarial:

    - adversarial.py trains the classifier and store them as pre-trained models. Moreover, with each new instance, their label will be predicted and returned by the adversarial model.
    - classifier_gen_data.pkl is trained classifier on generated data from generator MCD-VAE (mcd_vae.pth)
    - classifier_gen_data_custom.pkl is trained classifier on generated data from generator MCD-VAE with custom parameters (mcd_vae_custom.pth)
    - classifier_gen_data_perturb.pkl is trained classifier on generated data from perturbation (perturb.py)

## LIME
In this project, we modifided gLIME in directory /glime to find the most significant feature of the evaluate dataset. We need notice, that the adversarial acts as a black box, which could return predicted labels from an explanable instance. We use generators in directory /generator to sample new instances in the neighborhood of each explanable instance. After that, weights are calculated (phi) to show the similarity between the explanable instance and its neighborhood. If generated instance is close to the explanable instance, its weight is also close to 1. In this project, we choose Logistic Regression as the interpretable model for gLIME. The most significant feature is calculated in file glime.py as followings:

alpha = argmax(w<sub>i</sub> * x<sub>i</sub>)

with:

    - alpha: value of the most significant feature (feature with the largest contribution to the prediction).
    - wi is the weight of the interpretable model of feature index i. 
    - xi is the value of feature index i.

The equation above applies only for one instance, we performs gLIME for all instances in evaluate dataset and find out how many instances in the evaluate dataset return the sensitive feature, which is feature "race" in this case.
Since we have three pre-trained adversarial models stored in directory /adversarial, and three evaluate datasets generated by three pre-trained generators stored in directory /generator, we have in total 9 unique results in this project.

## Conclusion
Since the project is a lot of work for a small team of two students, who still lack experiences with implementing big projects, we still not received the desired results. However, it would be possible to optimize our implementation if more time is provided. In our re-implementation, the MCD-AE method for better sampling is programmed differently from the original paper. It is still promising that our method could perform well if it could be adjusted correctly, since autoencoders generally only need to optimize its rescontruction loss (distance between original and generated instances), while variational autoencoders contain reconstruction loss and Kullback Leiblber devergence. Optimizing hyperparameters of MCD-AE and MCD-VAE is still a challenge in general. Since the unbiased model of the adversarial model depends on randomness of the unrelated features, its performance is not ensure to be stable. The results are stored in result.txt

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

<a id="2">[4]</a>
Dor Bank, Noam Koenigstein, Raja Giryes (2021)
Autoencoders
https://arxiv.org/pdf/2003.05991.pdf
