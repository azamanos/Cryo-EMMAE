import numpy as np
from params.params_evaluation import get_args
from utils.evaluation_utils import compute_od_metrics_per_dataset

def main():
    #torch.set_num_threads(1)
    #First load the arguments
    config = get_args()
    #Load predicted dataset
    prediction_set = np.load(config.prediction_set_path)
    #Define different empiar sets
    empiar_sets = np.unique([i[:5] for i in prediction_set])
    if len(empiar_sets)==1:
        empiar_sets = [empiar_sets[0],]
    #Set experiment descriptive name
    if not config.experiment_description:
        config.experiment_description = str(config.experiment)
    #Compute metrics
    total_results = []
    total_results.append([])
    path_of_pred =  f'./results/prediction_{config.experiment}_{config.experiment_description}_{config.initial_img_length}_npy/'
    print(f'Metrics for model trained with {config.experiment_description}')
    print('{:<5s} {:<8s} {:<8s} {:<12s} {:<8s} {:<8s}'.format('Dset', 'IoU', 'Recall', 'Precision', 'F1', 'F3'))
    for es in empiar_sets:
        fr = compute_od_metrics_per_dataset(path_of_pred, config.ground_truth_path, [es,], cap_values=[0,], beta=3, gt_threshold=config.iou_threshold)[0,0]
        total_results[-1].append(list(fr))
        print('{:<5s} {:<8.3f} {:<10.3f} {:<9.3f} {:<8.3f} {:<8.3f}'.format(es, fr[0], fr[1], fr[2], fr[3], fr[4]))
    return

if __name__ == '__main__':
    main()
