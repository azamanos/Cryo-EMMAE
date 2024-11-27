import os
import time
import numpy as np
from PIL import Image
from utils.utils import normalize_array
from params.params_preprocess import get_args
from utils.preprocess_utils import MRC, relion_and_contrast_preprocess

def main():
    #First load the arguments
    config = get_args()
    #If there is a given list
    if config.micrographs_list:
        micrographs_list = config.micrographs_list
    else:
        #All images in the given directory are going to be processed
        micrographs_list = os.listdir(config.micrographs_directory)
    #Lenght of image list
    m_list_len = len(micrographs_list)
    #Secure that particle diameter is even number
    if config.particle_diameter%2:
        particle_diameter_even = config.particle_diameter + 1
    else:
        particle_diameter_even = config.particle_diameter
    #Starting time
    stime = time.time()
    #For each micrograph
    for mi, micrograph in enumerate(micrographs_list):
        #Micrograph time
        mtime = time.time()
        #Load micrograph image
        if config.mrc_type:
            m = MRC(f'{config.micrographs_directory}{micrograph}')
            m = normalize_array(m.data[:,:,0])*255
            m = np.moveaxis(m, 0,1)
            m = np.flip(m,0)
        else:
            m = np.array(Image.open(f'{config.micrographs_directory}{micrograph}'))
        #Save micrograph parameters
        recover_resize_coeff = [m.shape[0]/config.resize_shape, m.shape[1]/config.resize_shape]
        pd_resized = max(config.particle_diameter/recover_resize_coeff[0], config.particle_diameter/recover_resize_coeff[1])
        np.save(f'./results/preprocess_info/{config.dataset_ID}.npy', [m.shape[0], m.shape[1], pd_resized, recover_resize_coeff[0], recover_resize_coeff[1]])
        micrograph_name = str(micrograph.split('.')[0])
        relion_and_contrast_preprocess(m, particle_diameter_even, config.output_directory, micrograph_name, config.resize_shape)
        ctime = time.time()
        print(f'Micrograph {mi+1}/{m_list_len} was processed in {round(ctime-mtime,2)} seconds, total minutes passed {round((ctime-stime)/60,2)}.',end='\r')
    return

if __name__ == '__main__':
    main()

