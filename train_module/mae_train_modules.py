import torch
import time
import numpy as np
from tqdm import tqdm
from utils.evaluation_metrics import iou_and_dice_recall_precision_f1
from utils.utils import normalize_array
import utils.lr_sched as lr_sched

def rcc_batch_preparation_single(config,loader_list):
    #loader_list = list(loader)
    shuffled_data_resized = config.rrc(loader_list[0])
    for t in loader_list[1:]:
        shuffled_data_resized = torch.cat((shuffled_data_resized,config.rrc(t)),0)
    return shuffled_data_resized

def rcc_batch_preparation(config,loader_list):
    for i, p in enumerate(range(config.initial_img_patches_num)):
        if not i:
            c = rcc_batch_preparation_single(config,loader_list)
        else:
            c = torch.cat((c,rcc_batch_preparation_single(config,loader_list)),0)
    #Random shuffling
    sdri = np.arange(len(c))
    np.random.shuffle(sdri)
    c = c[sdri]
    #Prepare final loader
    final_loader = [c[i*config.training_batch_size:(i+1)*config.training_batch_size]\
                    for i in range(int(np.ceil(len(c)/config.training_batch_size)))]
    return final_loader

def train_loop(loader, model, config):
    '''
    Train loop, trains the model with the embedding from the predicted water coordinates of the 3D Unet.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads training data.

    model : torch.nn.Model
        Pytorch 3D Unet model that process embedding information.

    config : class
        Config class containing all the parameters needed for the training.

    Returns
    -------
    Total normalized epoch loss.
    '''
    grid_array = 0
    epoch_loss = 0
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.train()
    if config.random_resized_crop:
        loader_list = list(loader)
        shuffled_data_resized = rcc_batch_preparation(config,loader_list)
        batches = tqdm(shuffled_data_resized)
        loader_len = len(shuffled_data_resized)
    else:
        batches = tqdm(loader)
        loader_len = len(loader)
    for i, data in enumerate(batches):
        # we use a per iteration (instead of per epoch) lr scheduler
        if i % config.accum_iter == 0:
            config.epoch_lr = lr_sched.adjust_learning_rate(config, i / loader_len + config.epoch)
        x = data
        #x, y = data
        #Prepare them for the model
        x = x[:,None,:,:].to(config.device).float()
        #x, y = x[:,None,:,:].to(config.device).float(), y[:,None,:,:].to(config.device).float()
        #Predict
        loss, prediction, mask = model(x, config.mask_ratio)
        loss = loss/config.accum_iter
        #Backwards
        loss.backward()
        #Optimize
        config.optimizer.step()
        #Zero Gradients
        config.optimizer.zero_grad()
        batches.set_postfix(loss=loss.item())
        #Keep epoch loss normalized by size of batch
        epoch_loss += loss.item()#/x.shape[0]
        #Keep train set image
        if config.save_image_per_epoch and not i-config.keep_training_image:
            model.eval()
            with torch.no_grad():
                loss, pred_example, mask = model(x[0][None,:,:,:], config.mask_ratio)
                pred_example = model.unpatchify(pred_example)
                #input_norm, true_norm, pred_norm
                example_images = [normalize_array(x[0,0].to('cpu').detach().numpy())*255,
                                  normalize_array(pred_example[0,0].to('cpu').detach().numpy())*255]
                image_shape = example_images[0].shape
                grid_array = np.zeros((image_shape[0],image_shape[1]*len(example_images)))
                for im in range(len(example_images)):
                    grid_array[:image_shape[0],im*image_shape[1]:(im+1)*image_shape[1]] = example_images[im]
            model.train()
    #print(f"After epoch {config.epoch} learning rate is {epoch_lr}.")
    return epoch_loss/(i+1), grid_array

def validation_loop(loader, model, epoch, config):
    '''
    Validation loop, evaluates the validation set on the trained 3D Unet based on the recall, precision and F1 of the prediction.

    Parameters
    ----------
    loader : torch.nn.DataLoader
        Pytorch dataloader that loads validation data.

    model : torch.nn.Model
        Pytorch 2D Unet model that process embedding information.

    epoch : int
        Number of epoch that is going to be validated.

    config : class
        Config class containing all the parameters needed for the validation.

    Returns
    -------
    mean_recall : float
        mean recall computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    mean_precision : float
        mean precision computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    mean_F1 : float
        mean F1 computed for current epoch of the model and for the high F1 selected cap value, correspons to the whole ground truth and predicted waters of validation set.

    selected_cap_value : float
        selected cap value based on the highest F1 for the selected threshold.

    epoch_loss : float
        total normalized epoch loss for the validation set.
    '''
    count = 0
    pick_random_image = np.random.randint(0, high=config.initial_img_patches_num*len(config.validation_data_list))
    #iou, dice, recall, precision, f1
    #metrics = np.zeros((5,len(config.sigmoid_cap),len(config.validation_data_list)))
    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        st = time.time()
        loaderlen = len(loader)
        epoch_loss = 0
        for v, data in enumerate(loader):
            #Load data
            x = data
            #Prepare them for the model
            x = x[:,None,:,:].to(config.device).float()
            #Predict
            loss, prediction, mask = model(x, config.mask_ratio)
            #Divide my loss accumulator
            loss = loss/config.accum_iter
            #Keep epoch loss normalized by size of batch
            epoch_loss += loss.item()#/x.shape[0]
            #Compute metrics
            for pred_i, pred in enumerate(prediction):
                if config.save_image_per_epoch and not count-pick_random_image:
                    #Cap outputs
                    pred_example = model.unpatchify(pred[None,:,:]).detach().cpu()[0,0]
                    #input_norm, true_norm, pred_norm
                    example_images = [normalize_array(np.array(x[pred_i][0].to('cpu')))*255,
                                      normalize_array(np.array(pred_example))*255]
                    image_shape = example_images[0].shape
                    grid_array = np.zeros((image_shape[0],image_shape[1]*len(example_images)))
                    for im in range(len(example_images)):
                        grid_array[:image_shape[0],im*image_shape[1]:(im+1)*image_shape[1]] = example_images[im]
                    image_info = [grid_array, config.validation_data_list[int(pick_random_image/(config.initial_img_patches_num))]]
                #Keep counting images
                count += 1
    return epoch_loss/(v+1), image_info
