import os
import time
import wandb
import torch
import torchvision
import numpy as np
import torch.nn as nn
from functools import partial
from model.mae_model import MaskedAutoencoderViT
import utils.misc as misc
import utils.lr_decay as lrd
from params.params_train import get_args
from torch.utils.data import DataLoader
import timm.optim.optim_factory as optim_factory
from model.model_utils import interpolate_pos_embed
from datasets.mae_dataset import Train_Dataset, Validation_Dataset
from train_module.mae_train_modules import train_loop, validation_loop
from utils.torch_utils import load_checkpoint, save_checkpoint
from utils.prediction_utils import compute_image_latent_embeddings
from sklearn.cluster import KMeans

def main():
    #torch.set_num_threads(1)
    #First load the arguments
    config = get_args()
    config.train_data_list = os.listdir(config.input_dataset_path)
    config.validation_data_list = os.listdir(config.input_dataset_path)
    config.keep_training_image = int(len(config.train_data_list)*config.initial_img_patches_num/config.training_batch_size)-1
    if config.random_resized_crop:
        config.rrc = torchvision.transforms.RandomResizedCrop((config.img_size,config.img_size),scale=config.rrc_crop_range,antialias=True)
    if not os.path.exists(config.checkpoints_path):
        os.mkdir(config.checkpoints_path)
    #Initialize logging
    config.wandb_log = wandb.init(project=config.wandb_project, config=config, resume='allow', id=config.wandb_id)
    #Load training coordinates data
    train_dataset = Train_Dataset(config)
    #Load validation coordinates data
    validation_dataset = Validation_Dataset(config)
    #Define training loader
    train_loader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)
    #Define validation loader
    validation_loader = DataLoader(validation_dataset, batch_size=config.training_batch_size*2, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    #Initialize model
    model = MaskedAutoencoderViT(img_size=config.img_size, patch_size=config.patch_size, in_chans = 1,
                                 embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads,
                                 decoder_embed_dim=config.decoder_embed_dim, decoder_depth=config.decoder_depth, decoder_num_heads=config.decoder_num_heads,
                                 mlp_ratio=config.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), pos_encode_weight = config.pos_encode_weight)
    #Load model
    if config.load_model:
        epoch = config.starting_epoch
        checkpoint = torch.load(config.load_model, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % config.load_model)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        #Delete projection weights and biases of 3 input channels, we have only one
        del checkpoint_model['patch_embed.proj.weight']
        del checkpoint_model['patch_embed.proj.bias']

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # manually initialize fc layer
        #trunc_normal_(model.head.weight, std=2e-5)
    #Use all available GPUs if you want
    if config.parallel:
        if torch.cuda.device_count() > 1:
            print(f"Let's use, {torch.cuda.device_count()}, GPUs!")
            model = nn.DataParallel(model)
    #Send model to device
    model.to(config.device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    config.eff_batch_size = config.training_batch_size * config.accum_iter

    if config.learning_rate is False:  # only base_lr is specified
        config.learning_rate = config.base_learning_rate * config.eff_batch_size / 256

    print("base lr: %.2e" % (config.learning_rate * 256 / config.eff_batch_size))
    print("actual lr: %.2e" % config.learning_rate)

    print("accumulate grad iterations: %d" % config.accum_iter)
    print("effective batch size: %d" % config.eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, config.weight_decay)

    config.optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.999))

    config.loss_scaler = misc.NativeScalerWithGradNormCount()

    #Load if there is finetune checkpoint
    misc.load_model(config, model_without_ddp, config.optimizer, config.loss_scaler)

    #Start Training
    start_training_time = time.time()
    for epoch in range(config.starting_epoch, config.starting_epoch+config.num_epochs):
        config.epoch = epoch
        print(f'Epoch: {epoch}')
        #Train you model
        epoch_loss, training_grid_array = train_loop(train_loader, model, config)
        #Evaluate models with validation set.
        validation_epoch_loss, validation_image_info = validation_loop(validation_loader, model, epoch, config)
        if epoch-1 and epoch%config.save_image_per_epoch:
            #Write wandb report
            config.wandb_log.log({'train loss': epoch_loss,
                                  'epoch time (minutes)': np.round((time.time()-start_training_time)/60,2),
                                  'validation loss': validation_epoch_loss,
                                  'Epoch learning rate': config.epoch_lr})
        else:
            ex_image_shape = np.array(training_grid_array.shape)
            example_image = np.zeros((ex_image_shape[0]*2,ex_image_shape[1]))
            example_image[:ex_image_shape[0],:] = training_grid_array
            example_image[ex_image_shape[0]:,:] = validation_image_info[0]
            #Write wandb report
            config.wandb_log.log({'train loss': epoch_loss,
                                  'epoch time (minutes)': np.round((time.time()-start_training_time)/60,2),
                                  'validation loss': validation_epoch_loss,
                                  'Image Examples': wandb.Image(example_image, caption=f"Top: Train Image, Bottom: Validation Image, Left: Input Image, Right: Predicted Image, Validation Example: {validation_image_info[1]}"),
                                  'Epoch learning rate': config.epoch_lr})
        #Save model every five epochs.
        if config.keep_checkpoints and not epoch%config.keep_checkpoints or not (epoch+1)-(config.starting_epoch+config.num_epochs):
            #Save model checkpoints according to training devices.
            if config.parallel:
                model_checkpoint = {"state_dict": model.module.state_dict()}
            else:
                misc.save_model(f'{config.checkpoints_path}/MAE_epoch_{epoch}.pth.tar', epoch, model, model_without_ddp, config.optimizer, config.loss_scaler)
    #In case you want to compute kmeans
    if config.compute_kmeans:
        config.ip = config.input_dataset_path
        x = compute_image_latent_embeddings(config.train_data_list, config, model, resize=False)
        flat = x.shape[0]*x.shape[1]
        x = x.reshape(flat,-1)
        kmeans = KMeans(config.compute_kmeans,random_state=0, n_init="auto").fit(x)
        if not config.kmeans_id:
            config.kmeans_id = 'default'
        joblib.dump(kmeans, f'./results/kmeans/kmeans_validation_{cconfig.compute_kmeans}_run_{config.c.split(".")[0]}_epoch_{epoch}_finetuning.pkl')
    return

if __name__ == '__main__':
    main()
