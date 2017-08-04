from util import LoadList
import numpy as np
import matplotlib.pyplot as plt

import glob
import pdb
if __name__ == "__main__":
    save_path = '../data/trained_model/traffic-regression-dl/dataset2/'
    meta_name = '24-07-17_meta'
    save_list = sorted(glob.glob(save_path + meta_name + '*.dat'))[-1]
    last_e, running_loss, all_loss = LoadList(save_list) 
    
    # Plotting
    sm_factor = 50 # Smoothing factor
    start = 100
    end = -1
    plt.plot(all_loss[start:end:sm_factor])
    #plt.yscale('log')
    plt.grid(linestyle='--') 
    plt.show()
