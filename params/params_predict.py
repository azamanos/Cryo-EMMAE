import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Cryo-EM Prediction Parser')

    parser.add_argument('-c','--config_file', type=str, default=None, help='Path to YAML config file')

    # Paths
    parser.add_argument('--input_dataset_path', type=str, help='Path to input dataset')
    parser.add_argument('--target_dataset_path', type=str, help='Path to target dataset')
    parser.add_argument('--train_data_list', type=str, help='Path to train data list')
    parser.add_argument('--validation_data_list', type=str, help='Path to validation data list')

    # Parameters
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
    parser.add_argument('--initial_img_length', type=int, default=1024, help='Image Length of the initial micrograph')
    parser.add_argument('--patches_num', type=int, default=16, help='Number of patches per side for the ViT')
    parser.add_argument('--initial_img_patches_num', type=int, default=64, help='Number of patches of the initial image')

    # Prediction Parameters
    parser.add_argument('--experiment', type=int, default=None, required=True,help='Experiment number to load')
    parser.add_argument('--epoch', type=int, default=None, required=True, help='Number of epoch to load')
    parser.add_argument('--prediction-description', type=str, default=None, required=True, help='Predictions descriptive name')
    parser.add_argument('--prediction-set-path', type=str, default=None, required=True, help='Path to npy file array with images names of your prediction set')
    parser.add_argument('--remove-embeddings-directory', type=bool, default=True, help='Argument to remove predicted embeddings, default True')

    parser.add_argument('--particle-diameter', type=int, default=None, required=False,help='Particle diameter in pixels for the original shape')
    parser.add_argument('--dataset-ID', type=str, default=None, required=True,help='Dataset ID, saves particle diameter and original shape for use during prediction')

    # Other Parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

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
