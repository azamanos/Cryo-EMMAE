import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Cryo-EM Evaluation Parser')

    # Parameters
    parser.add_argument('--initial_img_length', type=int, default=1024, help='Image Length of the initial micrograph')

    # Evaluation Parameters
    parser.add_argument('--experiment', type=int, default=None, required=True,help='Experiment number to load')
    parser.add_argument('--prediction-description', type=str, default=None, required=True, help='Predictions descriptive name')
    parser.add_argument('--prediction-set-path', type=str, default=None, required=True, help='Path to npy file array with images names of your prediction set')
    parser.add_argument('--ground-truth-path', type=str, default='./results/target_512_20_npy/', help='Path to npy file array with ground truth positions of particles')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='IoU threshold to count prediction as true positive.')

    args = parser.parse_args()

    return args
