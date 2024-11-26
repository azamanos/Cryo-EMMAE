import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Cryo-EM Preprocess Parser')

    # Preprocess Parameters
    parser.add_argument('--micrographs-directory', type=str, default=None, required=True,help='Directory with micrograph to be processed')
    parser.add_argument('--micrographs-list', type=list, default=None, help='List of micrographs to be processed')
    parser.add_argument('--mrc-type', type=bool, default=True, help='If data type is mrc or image type file.')
    parser.add_argument('--resize-shape', type=int, default=1024, help='Resize shape')
    parser.add_argument('--particle-diameter', type=int, default=None, required=True,help='Particle diameter in pixels for the original shape')
    parser.add_argument('--output-directory', type=str, default=None, required=True,help='Directory to output processed images')

    args = parser.parse_args()

    return args
