import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy.io
from util import *
import pdb

def vis_result(cnn_result_path, hog_result_path):

    file_list = sorted(glob.glob(cnn_result_path + '*.dat'))
    gt = np.asarray([]).reshape((0,1))
    pred = np.asarray([]).reshape((0,1))
    for f in file_list:
        result = LoadList(f)
        gt = np.concatenate((gt, result[1]), 0)
        pred = np.concatenate((pred, result[0]), 0)
	
    hog_result = scipy.io.loadmat(hog_result_path)['complete_fits']
    
    plt.scatter(gt[:,0], pred[:,0], color='g', marker='s', edgecolor='k', label='CNN')
    plt.scatter(gt[:,0], hog_result, color='b', marker='s', edgecolor='k', label='BoW + LSSVM')
    plt.scatter(gt[:,0], gt[:,0], color='r', marker='s', edgecolor='k', label='Ground truth')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('Ground truth count')
    plt.ylabel('Predicted count')
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    vis_result('../data/trained_model/traffic-regression-dl/dataset1/test_result/', 
	    '../data/trained_model/traffic-regression-dl/dataset1/benchmark/lin_kernel_no_tfidf_K300')
    vis_result('../data/trained_model/traffic-regression-dl/dataset2/test_result/', 
	    '../data/trained_model/traffic-regression-dl/dataset2/benchmark/cellsize_16_K_300_scale_07_lin_kernel__grid_[1 1].mat')