import torch

def object_detection_intersection_over_union(gt, pred, hsl=None, box_format="midpoint"):
    if box_format == "midpoint":
        #hsl -> Half Side Length
        xmin1, xmax1, ymin1, ymax1 = gt[:, 0]-hsl, gt[:, 0]+hsl, gt[:, 1]-hsl, gt[:, 1]+hsl
        xmin2, xmax2, ymin2, ymax2 = pred[:, 0]-hsl, pred[:, 0]+hsl, pred[:, 1]-hsl, pred[:, 1]+hsl
    elif box_format == "corners":
        xmin1, xmax1, ymin1, ymax1 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        xmin2, xmax2, ymin2, ymax2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    # Calculate intersection area
    xmin = np.max(np.concatenate((xmin1[:,None],xmin2[:,None]),axis=1),axis=1)
    ymin = np.max(np.concatenate((ymin1[:,None],ymin2[:,None]),axis=1),axis=1)
    xmax = np.min(np.concatenate((xmax1[:,None],xmax2[:,None]),axis=1),axis=1)
    ymax = np.min(np.concatenate((ymax1[:,None],ymax2[:,None]),axis=1),axis=1)
    #Intersection calculation
    intersection = np.clip(xmax - xmin,0,None) * np.clip(ymax - ymin,0,None)
    #Calculate union area
    box1_area = abs((xmax1 - xmin1) * (ymax1 - ymin1))
    box2_area = abs((xmax2 - xmin2) * (ymax2 - ymin2))
    #Calculate union
    union = np.clip(box1_area + box2_area - intersection,1e-6,None)
    #Return IoU
    return intersection / union

## Metrics ##
def iou(prediction, target, epsilon=1e-6):
    prediction = torch.flatten(prediction)
    target = torch.flatten(target)
    #Intersection
    intersection = (prediction*target).sum(-1)
    union = prediction.sum() + target.sum() - intersection
    return intersection/union.clamp(epsilon)

def iou_batches(prediction, target, epsilon=1e-6):
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    #Intersection
    intersection = (prediction*target).sum(-1)
    #Union
    union = prediction.sum(-1) + target.sum(-1) - intersection
    return torch.mean(intersection/union.clamp(epsilon))

def dice(prediction, target, epsilon=1e-6):
    prediction = torch.flatten(prediction)
    target = torch.flatten(target)
    #Intersection
    intersection = (prediction*target).sum(-1)
    #Total area
    total_sum = (prediction).sum() + (target).sum()
    return 2*intersection/(total_sum.clamp(min=epsilon))

def dice_batches(prediction, target, epsilon=1e-6):
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    #Intersection
    intersection = (prediction*target).sum(-1)
    #Total area
    total_sum = (prediction).sum(-1) + (target).sum(-1)
    return torch.mean(2*intersection/(total_sum.clamp(min=epsilon)))

def iou_and_dice(prediction, target, epsilon=1e-6):
    prediction = torch.flatten(prediction)
    target = torch.flatten(target)
    #Intersection
    intersection = (prediction*target).sum(-1)
    #Total area
    total_sum = (prediction).sum() + (target).sum()
    #Union
    union = total_sum - intersection
    #Return IoU and Dice
    return intersection/union.clamp(epsilon), 2*intersection/(total_sum.clamp(min=epsilon))

def iou_and_dice_recall_precision_f1(prediction, target, epsilon=1e-4):
    prediction = torch.flatten(prediction)
    target = torch.flatten(target)
    #Intersection, also true_positive
    intersection = (prediction*target).sum(-1)
    #Total area
    total_sum = (prediction).sum() + (target).sum()
    #Union
    union = total_sum - intersection
    #Recall and Precision
    false_positive, false_negative = (prediction*(1-target)).sum(-1), ((1-prediction)*target).sum(-1)
    recall, precision = intersection/max(epsilon,(intersection+false_negative)), intersection/max(epsilon,(intersection+false_positive))
    f1 = 2*recall*precision/max(epsilon,(recall+precision))
    #Return IoU and Dice,
    return intersection/union.clamp(epsilon), 2*intersection/(total_sum.clamp(min=epsilon)), recall, precision, f1
