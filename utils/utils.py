import torch
import numpy as np

class Config(object):
    '''
    Empty class object to save various parameters for the training and inference of the models.
    '''
    def __init__(self):
        return

def dict_to_config(args_dict):
    config = Config()
    for keys,values in args_dict.items():
        if type(values) == str:
            exec(f"config.{keys} = '{values}'")
        else:
            exec(f"config.{keys} = {values}")
    config.train_data_list = np.load(config.train_data_list)
    config.validation_data_list = np.load(config.validation_data_list)
    config.keep_training_image = int(len(config.train_data_list)/config.training_batch_size)-1
    return config

def normalize_array(arr, min_value=None, max_value=None):
    divider = (arr.max() - arr.min())
    if not divider:
        return np.zeros(arr.shape)
    normalized_array = (arr - arr.min()) / (arr.max() - arr.min())  # Normalize to 0-1
    if max_value or min_value:
        normalized_array = normalized_array * (max_value - min_value) + min_value  # Scale to min_value-max_value
    return normalized_array

def create_2d_circular_mask(box_size, diameter, inverse=False):
    #Center of the circle
    cx = box_size[0]//2
    cy = box_size[1]//2
    #Create axis
    if not box_size[0]%2 and not diameter%2:
        x = np.concatenate((np.arange(1, cx+1),np.arange(cx, box_size[0])))
        y = np.concatenate((np.arange(1, cy+1),np.arange(cy, box_size[1])))
    else:
        x = np.arange(0, box_size[0])
        y = np.arange(0, box_size[1])
    #Compute radius
    r = diameter/2
    #Return mask
    if inverse:
        return (x[:,np.newaxis]-cx)**2 + (y[np.newaxis,:]-cy)**2 >= r**2
    return (x[:,np.newaxis]-cx)**2 + (y[np.newaxis,:]-cy)**2 < r**2

def create_submaps(original_image, square_size = 64, core_size = 50, data_type = np.float32):
    '''
    Function that creates subimage from an original map given box and core size.

    Parameters
    ----------
    original_image : numpy.array
        numpy array of shape (M,K) be careful, all M,K have to %50 = 0.

    square_size : int
        shape of square size, default 64

    core_size : int
        shape of core size, default 50

    data_type : type
        data type of array that will be returned, default np.float32.

    Returns
    -------
    subimages : numpy.array
        array that contains the subimages of the original map.
    '''
    #Keep image shape
    map_shape = np.shape(original_image)
    #Compute pad of the padded map
    pad = (square_size-core_size)//2
    #Initiliaze padded image of original map
    padded_map = np.pad(original_image, pad)
    #Initilize subimages list
    subimages = list()
    #Starting point in padded_map is 0
    x, y = 0, 0
    #While y index is not violating the size of original image shape
    while (x < map_shape[1]):
        #Get the next chunk of the padded image
        next_subimage = padded_map[x : x+square_size, y : y+square_size]
        #Appended to subimages list
        subimages.append(next_subimage)
        #Continue to the next chunk on the x axis
        y += core_size
        #If x index extends the x max
        if y >= map_shape[0]:
            #Increment y axis to the next chunk
            x += core_size
            #And reset x axis
            y = 0
    #When every chunk has been remove return
    return np.array(subimages, dtype=data_type)

def reconstruct_map(submaps, box_size = 64, core_size = 50, o_dim = None, data_type = np.float32):
    '''
    Function that reconstructs original shape map from given submaps, box and core size.

    Parameters
    ----------
    submaps : numpy.array
        numpy array of shape (X,M,K,L) be careful, all M,K,L have to %50 = 0, X is the number of submaps.

    box size : int
        shape of box size, default 64

    core_size : int
        shape of core size, default 50

    o_dim : tuple
        tuple with the original shape of map, default None.

    data_type : type
        data type of array that will be returned, default np.float32.

    Returns
    -------
    reconstructed_map : numpy.array
        array of the recostructed original map.
    '''
    #If dimensions are not given probably dimensions are the same at x, y, z.
    if not o_dim:
        #Compute dimensions
        o_dim = int(np.shape(submaps)[0]**(0.333334))
        o_dim = [int(i * o_dim) for i in (core_size,core_size)]
    #Extraction start and end of the submap, remember you only need the core
    s = (box_size - core_size)//2
    e = s + core_size
    #Initialize reconstruction map
    reconstructed_map = np.zeros(tuple(o_dim))
    #Initialize counter
    i = 0
    #For each submap corresponding to y axis
    for y in range(o_dim[1]//core_size):
        #For each submap corresponding to x axis
        for x in range(o_dim[0]//core_size):
            #Fill reconstruction map
            reconstructed_map[x*core_size:(x+1)*core_size, y*core_size:(y+1)*core_size] = submaps[i][s:e, s:e]
            i += 1
    #Return reconstruction map with given data_type
    return np.array(reconstructed_map, dtype=np.float32)

def image_patching(x, patch_number, patch_size):
    """
    x: (N, H, W)
    x: (N, patch_number, patch_size, patch_size)
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if len(x.shape)==2:
        x = x[None,:,:]
        
    patch_axis_number = int(patch_number**0.5)
    
    im_num =  x.shape[0]
    im_shape = x.shape[-1]
    x = x.reshape(im_num, patch_axis_number, patch_size, patch_axis_number, patch_size).swapaxes(2,3).reshape(im_num, 1, patch_number, patch_size, patch_size)
    return np.array(x)

def image_unpatching(x):
    """
    x: (N, patch_number, patch_size, patch_size)
    x: (N, H, W)
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if len(x.shape)==3:
        x = x[None,:,:,:]
        
    n, patch_number, patch_size, patch_size = x.shape
    patch_axis_number = int(patch_number**0.5)
    im_shape = patch_axis_number*patch_size
    x = x.reshape(n, patch_axis_number, patch_axis_number, patch_size, patch_size).swapaxes(2,3).reshape(n, im_shape,im_shape)
    return np.array(x)

def embedding_unpatching(x):
    """
    x: (N, patch_number, patch_size, patch_size, embedding_size)
    x: (N, H, W)
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if len(x.shape)==4:
        x = x[None,:,:,:,:]
        
    n, patch_number, patch_size, patch_size, embedding_size = x.shape
    patch_axis_number = int(patch_number**0.5)
    im_shape = patch_axis_number*patch_size
    x = x.reshape(n, patch_axis_number, patch_axis_number, patch_size, patch_size, embedding_size).swapaxes(2,3).reshape(n, im_shape,im_shape,embedding_size)
    return np.array(x)

def components_to_coordinates(label_where, arr):
    label_coords, label_coords_weights = np.array(label_where).T.astype(float), arr[label_where]
    weight_sum = np.sum(label_coords_weights)
    #Compute the weighted coordinates
    weighted_coords = label_coords*np.expand_dims(label_coords_weights/weight_sum, axis=1)
    return weighted_coords, weight_sum

def position_of_points_towards_line(a, b, q):
    """
    Check the direction of a point in respect with a line consists of two points in R^2.
    The direction of the line-points should be from bottom to top for vertical lines and from right to left for horizontal lines.
    Positive number means that your point exist on the top or right of your line.
    Negative number means that your point exist on the bottom or left of your line.
    Zero indicates that your point is on your line.
    For more: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    
    Args:
        a (array or list): (2) or (1,2) numpy array or list of a point's coordinates in R^2 space.
        b (array or list): (2) or (1,2) numpy array or list of another point's coordinates in R^2 space.
        q (array or list): (2) or (1,2) numpy array or list of a query point's coordinates in R^2 space.
    Returns:
        Float, that its sign and value indicates the direction of a point in respect with a line.
    """
    return (q[:,0]-a[0])*(b[1]-a[1])-(q[:,1]-a[1])*(b[0]-a[0])

def split_connected_component(label_where):
    #Transform indices as coordinates
    label_where_coords = np.array(label_where).T.astype(float)
    #Compute Distance Matrix
    d = coordinates_to_dmatrix(label_where_coords,label_where_coords)
    #Find maximum distance (longest axis)
    mdps = np.where(d==np.max(d))[0]
    #Keep the two outermost coordinates
    x1,y1 = label_where_coords[mdps][0]
    x2,y2 = label_where_coords[mdps][1]
    #Compute the midpoint
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    #Define points of the perpendicular
    xv1,xv2 = -512, 512
    #Calculate the slope of the perpendicular line
    if not x2 - x1:
        #Define y's on the perpendicular line
        yv1 = yv2 = midpoint[1]
    else:
        m = -1 / ((y2 - y1) / (x2 - x1))
        # Use the point-slope form to get the equation of the perpendicular line passing through the midpoint
        b = midpoint[1] - m * midpoint[0]
        #Compute two points on the perpendicular line
        yv1,yv2 = m*xv1+b, m*xv2+b
    #Compute the side of the initial label coordinates in respect with the vertical line
    point_relative_to_vertical = position_of_points_towards_line([xv1,yv1],[xv2,yv2],label_where_coords)
    #Split the connected component into two
    l1, l2 = label_where_coords[np.where(point_relative_to_vertical<0)[0]].astype(int),\
             label_where_coords[np.where(point_relative_to_vertical>0)[0]].astype(int)
    #Return the indices of the two new connected components
    return tuple(l1.T), tuple(l2.T)

def coordinates_to_dmatrix(a_coords, b_coords):
    '''
    Creates distance matrix for numpy.array input.

    Parameters
    ----------
    a_coords : numpy.array
        numpy.array of shape (N,3) that contains coordinates information.

    b_coords : numpy.array
        numpy.array of shape (M,3) that contains coordinates information.

    Returns
    -------
    numpy.array of shape (N,M), the distance matrix of a_coords and b_coords.
    '''
    a, b = torch.from_numpy(a_coords), torch.from_numpy(b_coords)
    return np.array(torch.cdist(a,b))

def remove_duplicates(predicted_coords, predicted_weights, dist, batch_size = 50):
    '''
    Function that removes duplicates simply, created for post processing of the 3D Unet, removes nearby water coordinates based on their scores.

    Parameters
    ----------
    predicted_coords : numpy.array
        numpy array of shape (N,3) of the predicted water coordinates.

    predicted_weights : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    dist : float
        distance value to look after for duplicates.

    batch_size : int
        batch size value to process for duplicates, default 50.

    Returns
    -------
    todelete : numpy.array
        indexes of predicted water coordinates to delete as duplicates.
    '''
    #Make a copy of your predicted_coords and predicted_weights
    predicted_coords_refined, predicted_weights_refined = np.copy(predicted_coords), np.copy(predicted_weights)
    coords_len = len(predicted_coords)
    #Find the batches according to batch size
    coords_batches = int(np.ceil(coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    batch_loop = np.linspace(0, coords_len, coords_batches, dtype=int)
    #todelete list will keep indexes you want to remove
    todelete = []
    #Start computing for each batch
    for b_i, batch in enumerate(batch_loop[:-1]):
        indexes = slice(batch, batch_loop[b_i+1])
        predicted_coords_batch = predicted_coords[indexes]
        #Calculate distances of predicted waters
        d_matrix = coordinates_to_dmatrix(predicted_coords_batch, predicted_coords)
        #Find duplicates within certain distance
        duplicates = np.unique(np.where(d_matrix<dist)[0])
        #Sort duplicates by the larger predicted weight index to the smallest predicted weight index.
        #duplicates = duplicates[np.argsort(-predicted_weights[duplicates])]
        #Keep the original indexing of duplicates
        duplicates_original_indexing = duplicates+batch
        #For each duplicate
        for i,j in zip(duplicates,duplicates_original_indexing):
            #Keep indexes of close predicted waters
            closeby = np.where(d_matrix[i]<dist)[0]
            #Remove j from closeby
            closeby_r = np.delete(closeby, np.argwhere(closeby==j)[0])
            #If your coordinate has the higher prediction in the region
            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                todelete += closeby_r.tolist()
            else:
                todelete.append(j)
    return np.unique(todelete)

def remove_and_refine_duplicates(predicted_coords, predicted_weights, dist, batch_size = 50):
    '''
    Function that removes duplicates and refines predicted coordinates, created for post processing of the mlp model, removes nearby water coordinates based on their scores, and refines coordinates by applying weighted average.

    Parameters
    ----------
    predicted_coords : numpy.array
        numpy array of shape (N,3) of the predicted water coordinates.

    predicted_weights : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    dist : float
        distance value to look after for duplicates.

    batch_size : int
        batch size value to process for duplicates, default 50.

    Returns
    -------
    predicted_coords_refined : numpy.array
        refined predicted coordinates of the water molecules, shape (N,3).

    predicted_weights_refined : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    todelete : numpy.array
        indexes of predicted water coordinates to delete as duplicates.
    '''
    #Make a copy of your predicted_coords and predicted_weights
    predicted_coords_refined, predicted_weights_refined = np.copy(predicted_coords), np.copy(predicted_weights)
    coords_len = len(predicted_coords)
    #Find the batches according to batch size
    coords_batches = int(np.ceil(coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    batch_loop = np.linspace(0, coords_len, coords_batches, dtype=int)
    #todelete list will keep indexes you want to remove
    todelete = []
    #Start computing for each batch
    for b_i, batch in enumerate(batch_loop[:-1]):
        indexes = slice(batch, batch_loop[b_i+1])
        predicted_coords_batch = predicted_coords[indexes]
        #Calculate distances of predicted waters
        d_matrix = coordinates_to_dmatrix(predicted_coords_batch, predicted_coords)
        #Find duplicates within certain distance
        duplicates = np.unique(np.where(d_matrix<dist)[0])
        #Sort duplicates by the larger predicted weight index to the smallest predicted weight index.
        #duplicates = duplicates[np.argsort(-predicted_weights[duplicates])]
        #Keep the original indexing of duplicates
        duplicates_original_indexing = duplicates+batch
        #For each duplicate
        for i,j in zip(duplicates,duplicates_original_indexing):
            #Keep indexes of close predicted waters
            closeby = np.where(d_matrix[i]<dist)[0]
            #Remove j from closeby
            closeby_r = np.delete(closeby, np.argwhere(closeby==j)[0])
            #If your coordinate has the higher prediction in the region
            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                #Keep to delete the coordinates around it
                todelete += closeby_r.tolist()
                #Compute weighted average for coordinates within 1.4 Angstrom
                #closeby = closeby[np.where(d_matrix[i][closeby]<1.4)[0]]
                #Keep indexes of close predicted waters
                region_indexes = closeby.tolist()
                #Compute the weights of the region
                region_weights = predicted_weights[region_indexes]
                #Sum the weights of the regions
                weight_sum = np.sum(region_weights)
                #Compute the weighted coordinates
                weighted_coords = predicted_coords[region_indexes]*np.expand_dims(region_weights/weight_sum, axis=1)
                predicted_coords_refined[j] = np.sum(weighted_coords, axis=0)
                #predicted_weights_refined[j] = weight_sum/len(region_weights)
            else:
                todelete.append(j)
    return predicted_coords_refined, predicted_weights_refined, np.unique(todelete)
