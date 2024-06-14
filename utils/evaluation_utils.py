import os
import numpy as np
from utils.prediction_utils import coordinates_to_dmatrix

def object_detection_intersection_over_union(gt, pred, hsl=None, box_format="midpoint"):
    """
    Calculates the Intersection over Union (IoU) for object detection.

    Parameters
    ----------
    gt : numpy.ndarray
        Ground truth bounding boxes with shape (N, 2) if `box_format` is "midpoint" or (N, 4) if `box_format` is "corners".

    pred : numpy.ndarray
        Predicted bounding boxes with shape (N, 2) if `box_format` is "midpoint" or (N, 4) if `box_format` is "corners".

    hsl : float or numpy.ndarray, optional
        Half side length of the bounding boxes when `box_format` is "midpoint". Ignored if `box_format` is "corners".

    box_format : str, optional
        Format of the bounding boxes. Either "midpoint" (default) or "corners".
        - "midpoint": Bounding boxes are defined by their center coordinates and half side length (hsl).
        - "corners": Bounding boxes are defined by their corner coordinates [xmin, ymin, xmax, ymax].

    Returns
    -------
    numpy.ndarray
        IoU values for each pair of ground truth and predicted bounding boxes.
    """
    if box_format == "midpoint":
        # Calculate the corners of the bounding boxes
        xmin1, xmax1, ymin1, ymax1 = gt[:, 0] - hsl, gt[:, 0] + hsl, gt[:, 1] - hsl, gt[:, 1] + hsl
        xmin2, xmax2, ymin2, ymax2 = pred[:, 0] - hsl, pred[:, 0] + hsl, pred[:, 1] - hsl, pred[:, 1] + hsl
    elif box_format == "corners":
        xmin1, xmax1, ymin1, ymax1 = gt[:, 0], gt[:, 2], gt[:, 1], gt[:, 3]
        xmin2, xmax2, ymin2, ymax2 = pred[:, 0], pred[:, 2], pred[:, 1], pred[:, 3]

    # Calculate intersection area
    xmin = np.max(np.concatenate((xmin1[:, None], xmin2[:, None]), axis=1), axis=1)
    ymin = np.max(np.concatenate((ymin1[:, None], ymin2[:, None]), axis=1), axis=1)
    xmax = np.min(np.concatenate((xmax1[:, None], xmax2[:, None]), axis=1), axis=1)
    ymax = np.min(np.concatenate((ymax1[:, None], ymax2[:, None]), axis=1), axis=1)

    # Calculate intersection area
    intersection = np.clip(xmax - xmin, 0, None) * np.clip(ymax - ymin, 0, None)

    # Calculate area of the bounding boxes
    box1_area = abs((xmax1 - xmin1) * (ymax1 - ymin1))
    box2_area = abs((xmax2 - xmin2) * (ymax2 - ymin2))

    # Calculate union area
    union = np.clip(box1_area + box2_area - intersection, 1e-6, None)

    # Calculate and return IoU
    return intersection / union


def compute_od_metrics_per_dataset(path_of_pred, path_of_gt, datasets, cap_values=[0,], beta=3, gt_threshold=0.6):
    """
    Computes object detection metrics per dataset.

    Parameters
    ----------
    path_of_pred : str
        Path to the directory containing the predicted results.

    path_of_gt : str
        Path to the directory containing the ground truth results.

    datasets : list
        List of dataset identifiers to process.

    cap_values : list, optional
        List of threshold values for capping predictions. Default is [0,].

    beta : float, optional
        The beta value for the F1-score calculation. Default is 3.

    gt_threshold : float, optional
        The ground truth threshold for IoU calculation. Default is 0.6.

    Returns
    -------
    np.ndarray
        Array containing the computed metrics for each dataset and cap value.
    """
    # Initialize results array
    metrics = np.zeros((len(datasets), len(cap_values), 5))

    # Get list of prediction and ground truth file paths
    pred_files = os.listdir(path_of_pred)
    gt_files = [f'{path_of_gt}{file}' for file in pred_files]
    pred_files = [f'{path_of_pred}{file}' for file in pred_files]

    # Iterate over each dataset
    for dataset_idx, dataset in enumerate(datasets):
        for cap_idx, cap_value in enumerate(np.round(cap_values, 2)):
            total_pred_len = 0
            ious_list = []
            weights_list = []

            # Iterate over prediction and ground truth files
            for gt_file, pred_file in zip(gt_files, pred_files):
                if pred_file.split('/')[-1].split('_')[0] != dataset:
                    continue

                # Load ground truth and prediction arrays
                gt_array = np.load(gt_file)
                pred_array = np.load(pred_file, allow_pickle=True)

                if not len(pred_array):
                    ious_list.extend([0] * len(gt_array))
                    continue

                k2 = int(pred_array[0, 2])
                weights = pred_array[:, 3].astype(float)

                # Apply cap
                valid_indices = np.where(weights > cap_value)
                pred_array = pred_array[:, :2].astype(float)
                pred_array = pred_array[valid_indices]

                if not len(pred_array):
                    ious_list.extend([0] * len(gt_array))
                    continue

                # Compute IoUs
                distances = coordinates_to_dmatrix(gt_array.astype(float), pred_array)
                min_distances, min_indices = np.min(distances, axis=1), np.argmin(distances, axis=1)
                ious = object_detection_intersection_over_union(gt_array, pred_array[min_indices], np.repeat(k2, len(gt_array)))

                ious_list.extend(ious)
                weights_list.extend(weights[valid_indices])

                total_pred_len += len(pred_array)

            ious_array, weights_array = np.array(ious_list), np.array(weights_list)

            # Calculate metrics
            if len(ious_array):
                mean_iou = np.round(np.mean(ious_array), 3)
            else:
                mean_iou = 0

            recall = np.round(np.sum(ious_array > gt_threshold) / max(len(ious_array), 1e-6), 3)
            precision = np.round(np.sum(ious_array > gt_threshold) / max(total_pred_len, 1e-6), 3)
            f1_score = np.round(2 * recall * precision / max((recall + precision), 1e-6), 3)
            f1_weighted = np.round((1 + beta ** 2) * recall * precision / max((recall + (beta ** 2) * precision), 1e-6), 3)

            metrics[dataset_idx, cap_idx, :] = mean_iou, recall, precision, f1_score, f1_weighted

    return metrics

