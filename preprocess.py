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
    if config.ml:
        micrographs_list = config.ml
    else:
        #All images in the given directory are going to be processed
        micrographs_list = os.listdir(config.md)
    if not os.path.exists(config.od):
        os.mkdir(config.od)
    #Lenght of image list
    m_list_len = len(micrographs_list)
    #Secure that particle diameter is even number
    if config.pd%2:
        particle_diameter_even = config.pd + 1
    else:
        particle_diameter_even = config.pd
    #Starting time
    stime = time.time()
    #For each micrograph
    for mi, micrograph in enumerate(micrographs_list):
        #Micrograph time
        mtime = time.time()
        #Load micrograph image
        if config.t:
            m = MRC(f'{config.md}{micrograph}')
            m = normalize_array(m.data[:,:,0])*255
            m = np.moveaxis(m, 0,1)
            m = np.flip(m,0)
        else:
            m = np.array(Image.open(f'{config.md}{micrograph}'))
        #Save micrograph parameters
        recover_resize_coeff = [m.shape[0]/config.rs, m.shape[1]/config.rs]
        pd_resized = max(config.pd/recover_resize_coeff[0], config.pd/recover_resize_coeff[1])
        np.save(f'./results/preprocess_info/{config.id}.npy', [m.shape[0], m.shape[1], pd_resized, recover_resize_coeff[0], recover_resize_coeff[1]])
        micrograph_name = str(micrograph.split('.')[0])
        relion_and_contrast_preprocess(m, particle_diameter_even, config.od, micrograph_name, config.rs)
        ctime = time.time()
        print(f'Micrograph {mi+1}/{m_list_len} was processed in {round(ctime-mtime,2)} seconds, total minutes passed {round((ctime-stime)/60,2)}.',end='\r')
    print(f'Micrograph {mi+1}/{m_list_len} was processed in {round(ctime-mtime,2)} seconds, total minutes passed {round((ctime-stime)/60,2)}.')
    return

if __name__ == '__main__':
    main()

