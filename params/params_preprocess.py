import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Cryo-EM Preprocess Parser')

    # Preprocess Parameters
    parser.add_argument('--md', type=str, default=None, required=True,help='Directory with micrograph to be processed')
    parser.add_argument('--ml', type=str, default=None, help='List of micrographs to be processed in npy format')
    parser.add_argument('--t', type=bool, default=True, help='If data type is mrc or image type file, default is True == mrc type files.')
    parser.add_argument('--rs', type=int, default=1024, help='Resize shape')
    parser.add_argument('--pd', type=int, default=200, required=False,help='Particle diameter in pixels for the original shape')
    parser.add_argument('--od', type=str, default=None, required=True,help='Directory to output processed images')
    parser.add_argument('--id', type=str, default=None, required=True,help='Dataset identifier, saves particle diameter and original shape for use during prediction')

    args = parser.parse_args()

    return args
