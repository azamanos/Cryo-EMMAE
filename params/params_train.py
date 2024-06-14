import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Cryo-EM Training Parser')

    parser.add_argument('-c','--config_file', type=str, default=None, help='Path to YAML config file')

    # Paths
    parser.add_argument('--input_dataset_path', type=str, help='Path to input dataset')
    parser.add_argument('--target_dataset_path', type=str, help='Path to target dataset')
    parser.add_argument('--train_data_list', type=str, help='Path to train data list')
    parser.add_argument('--validation_data_list', type=str, help='Path to validation data list')

    # Parameters
    parser.add_argument('--wandb_project', type=str, default='Cryo-MAE', help='Wandb project name')
    parser.add_argument('--wandb_id', type=str, help='Wandb ID name')
    parser.add_argument('--standarization', type=bool, default=True, help='Standarization of the data, (x-mean)/std')

    # Model Parameters
    parser.add_argument('--img_size', type=int, default=64, help='Input image size')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--patch_size', type=int, default=4, help='Number of pixels per side for the ViT patch')
    parser.add_argument('--embed_dim', type=int, default=192, help='Embedding dimension of the Encoder')
    parser.add_argument('--depth', type=int, default=14, help='Depth of the Encoder')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of Heads for the Transfomers of the Encoder')
    #Decoder
    parser.add_argument('--decoder_embed_dim', type=int, default=128, help='Embedding dimension of the Decoder')
    parser.add_argument('--decoder_depth', type=int, default=7, help='Depth of the Decoder')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Number of Heads for the Transfomers of the Decoder')
    #
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help='MLP ratio')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Masking ratio')
    parser.add_argument('--pos_encode_weight', type=float, default=0.08, help='Positional encoding weight for the encoder.')
    parser.add_argument('--initial_img_length', type=int, default=512, help='Image Length of the initial micrograph')
    parser.add_argument('--patches_num', type=int, default=16, help='Number of patches per side for the ViT')
    parser.add_argument('--initial_img_patches_num', type=int, default=64, help='Number of patches of the initial image')

    # Training Parameters
    parser.add_argument('--training_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--base_learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise learning rate decay')
    parser.add_argument('--minimum_learning_rate', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Epochs to warmup LR')
    parser.add_argument('--accum_iter', type=int, default=1, help='Accumulate gradient iterations')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--keep_checkpoints', type=int, default=50, help='Keep checkpoints')
    parser.add_argument('--load_model', type=str, default=False, help='Load model, give model path')
    parser.add_argument('--starting_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Pin memory')
    parser.add_argument('--random_resized_crop', type=bool, default=False, help='Randomly Resize and Crop the initial micrograph')
    parser.add_argument('--rrc_crop_range', nargs='+', type=int, help='Range of random resize')

    # Validation Parameters
    parser.add_argument('--sigmoid_cap', nargs='+', type=float, default=[0.5], help='Sigmoid cap')
    parser.add_argument('--save_image_per_epoch', type=int, default=10, help='Save image per epoch')

    # Log Parameters
    parser.add_argument('--keep_log_loss_per_steps', type=bool, default=False, help='Keep log loss per steps')
    #parser.add_argument('--log_loss_per_steps', type=int, default=1600, help='Log loss per steps')
    #parser.add_argument('--keep_training_image', type=int, default=80, help='Keep training image')

    # Other Parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    #parser.add_argument('--parallel', type=bool, default=False, help='Parallel')

    args = parser.parse_args()

    if args.config_file is not None:
        # Load parameters from YAML file
        with open(args.config_file, 'r') as file:
            yaml_args = yaml.safe_load(file)

        # Update argparse namespace with YAML args
        parser.set_defaults(**yaml_args)

    # Parse command-line arguments again to allow overriding YAML parameters
    args = parser.parse_args()

    return args
