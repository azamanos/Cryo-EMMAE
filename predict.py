import os
import time
import wandb
import torch
import joblib
import torchvision
import numpy as np
import torch.nn as nn
from functools import partial
from model.mae_model import MaskedAutoencoderViT
import utils.misc as misc
import utils.lr_decay as lrd
from params.params_predict import get_args
from torch.utils.data import DataLoader
import timm.optim.optim_factory as optim_factory
from model.model_utils import interpolate_pos_embed
from datasets.mae_dataset import Train_Dataset, Validation_Dataset
from train_module.mae_train_modules import train_loop, validation_loop
from utils.torch_utils import load_checkpoint, save_checkpoint
from utils.prediction_utils import compute_mean_and_std, compute_kmeans_on_training_set, compute_image_latent_embeddings, find_particles_cluster, predict_particles_maps, particle_diameter_512, pick_particles


def main():
    #torch.set_num_threads(1)
    #First load the arguments
    config = get_args()
    #Load training set
    config.train_data_list = np.load(config.train_data_list)
    #Set experiment descriptive name
    if not config.prediction_description:
        config.prediction_description = str(config.experiment)
    #Initialize model
    model = MaskedAutoencoderViT(img_size=config.img_size, patch_size=config.patch_size, in_chans = 1,
                                 embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads,
                                 decoder_embed_dim=config.decoder_embed_dim, decoder_depth=config.decoder_depth,\
                                 decoder_num_heads=config.decoder_num_heads,mlp_ratio=config.mlp_ratio,\
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), pos_encode_weight = config.pos_encode_weight)
    #Set epoch path
    config.load_model = f'./checkpoints/{config.experiment}/MAE_epoch_{config.epoch}.pth.tar'
    #Load checkpoint
    checkpoint = torch.load(config.load_model, map_location='cpu')
    checkpoint_model = checkpoint['model']
    try:
        del checkpoint_model['cls_token_classifier.weight']
        del checkpoint_model['cls_token_classifier.bias']
    except:
        pass
    load_checkpoint(checkpoint_model, model, config.epoch)
    msg = model.to(config.device)
    msg = model.eval()
    #Compute standarization mean and sigma
    if config.standarization:
        standarization_path = f'./results/standarization/standarization_run_{config.experiment}.npy'
        if os.path.exists(standarization_path):
            config.mean, config.std = np.load(standarization_path)
        else:
            print('Computing Mean and Standard Devivation.')
            config.mean, config.std = compute_mean_and_std(config)
            np.save(standarization_path, np.array([config.mean, config.std]))
            print('Done')
    #Define kmeans path
    kmeans_path = f'./results/kmeans/kmeans_validation_4_run_{config.experiment}_epoch_{config.epoch}.pkl'
    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
    else:
        print('Computing KMeans on Training Set.')
        #Compute kmeans on training set
        compute_kmeans_on_training_set(config, model, cl=4)
        print('Done')
    ## Proceed to prediction ##
    #Set output shape segmentation map
    config.output_image_len = config.initial_img_length//config.patch_size
    #Find particles cluster on the computed kmeans
    your_cluster = find_particles_cluster(kmeans, config.initial_img_length, config, model, 4)
    #Load dataset to predict
    prediction_set = np.load(config.prediction_set_path)
    #Predict the embeddings for your set
    temp_embeddings_path = './results/temp_embeddings/'
    if not os.path.exists(temp_embeddings_path):
        os.mkdir(temp_embeddings_path)
    for image in prediction_set:
        np.save(f'{temp_embeddings_path}{image.split(".")[0]}.npy',compute_image_latent_embeddings([image,], config, model, predict=True)[0])
    print('Computing Segmentation Map of Prediction Set.')
    #Predict particles segmentation maps for you set
    prediction_maps = predict_particles_maps(prediction_set, temp_embeddings_path, kmeans, your_cluster, config, particle_diameter_512)
    print('Done')
    #Predict particles
    cap_values = list(np.round(np.linspace(1,0.0,30),2))
    print('Pick Particles for Prediction Set.')
    pick_particles(prediction_maps, prediction_set, config.experiment, config.prediction_description, config.initial_img_length, cap_values, particle_diameter_512,\
                   f'./results/prediction_{config.experiment}_{config.prediction_description}_{config.initial_img_length}_npy/')
    print('Done')
    if config.remove_embeddings_directory:
        os.system(f'rm {temp_embeddings_path} -rf')
    return

if __name__ == '__main__':
    main()