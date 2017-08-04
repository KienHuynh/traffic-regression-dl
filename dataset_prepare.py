import numpy as np
import matplotlib.image as mpimg
import scipy.misc
import scipy.io

import glob
from util import SaveH5

import pdb

def dataset_pack(datadir, max_num_img, imw=512, imh=512, train_ratio=0.7, normalize=False):
    """dataset_pack
    This function find a bunch of images in datadir and:
    * Compute the RGB mean
    * Pack all images into a numpy array of shape (N, 3, H, W)
    * Return the array and the mean
    Note that normalize is default to False, the data in train_x and test_x are unnormalized and kept as uint8 to save space. You must subtract them by the mean after loading them before you start training.
    :param datadir: string, image directory, image name should have an easily sortable format, i.e. 0001.jpg not 1.jpg 
    :param imw: int, width of the image to be resized
    :param imh: int, height of the image to be resized
    :param train_ratio: float, ratio of train samples over all samples

    :return h5_dict: the dictionary storing the array and the mean values
    """
    if (datadir[-1] != '/'):
        datadir += '/'

    # Getting files in the folder, jpg and png only
    file_list = glob.glob(datadir + '*.jpg')
    if (len(file_list) == 0):
        file_list = glob.glob(datadir + '*.png')
    
    num_file = len(file_list)
    assert num_file != 0, 'datadir incorrect, no image of extension .jpg or .png found'
    if (num_file > max_num_img):
        num_file = max_num_img

    file_list = sorted(file_list)
    #file_list = [datadir + file_list[i] for i in range(0, num_file)] 

    # Create the train/test data with 7/3 ratio
    num_train = int(num_file*train_ratio)

    h5_dict = {}
    h5_dict['train_x'] = np.zeros((num_train, 3, imh, imw), dtype=np.float32)
    h5_dict['test_x'] = np.zeros((num_file-num_train, 3, imh, imw), dtype=np.float32)
    
    for i in range(0, num_train):
        img = mpimg.imread(file_list[i])
        img = scipy.misc.imresize(img, (imh, imw), 'bicubic')
        img = np.transpose(img, (2,0,1))
        h5_dict['train_x'][i,:,:,:] = img
        
        if (i % 50 == 0):
            print(i)
    
    h5_dict['mean'] = np.mean(h5_dict['train_x'], (0,2,3), np.float64)
    h5_dict['mean'] = h5_dict['mean'].reshape(1, 3, 1, 1)
     
    for i in range(num_train, num_file): 
        img = mpimg.imread(file_list[i])
        img = scipy.misc.imresize(img, (imh, imw), 'bicubic')
        img = np.transpose(img, (2,0,1))
        h5_dict['test_x'][i-num_train,:,:,:] = img
        
        if (i % 50 == 0):
            print(i)
    
    if (normalize):
        h5_dict['train_x'] = h5_dict['train_x'] - h5_dict['mean']
        h5_dict['test_x'] = h5_dict['test_x'] - h5_dict['mean']
    else:
        h5_dict['train_x'] = h5_dict['train_x'].astype(np.uint8)
        h5_dict['test_x'] = h5_dict['test_x'].astype(np.uint8)

    return h5_dict

if __name__ == '__main__':
    # Create auxilary data for dataset 1
    #datadir = '/media/kien/D80079EB0079D14C/IPCV group/Datasets/jpg/'
    #labelpath = '../data/traffic-data/dataset1_countlabel.mat'
    #h5_dict =  dataset_pack(datadir, 1000)
    #label = scipy.io.loadmat(labelpath) 
    #label = label['Dataset']
    #label = np.delete(label, 3, 1)
    #label = label[0:1000,:]
    #h5_dict['train_y'] = label[0:700,:]
    #h5_dict['test_y'] = label[700:1000,:]

    #pdb.set_trace()
    #SaveH5(h5_dict, '../data/traffic-data/dataset1_count_aux.h5')

    datadir = '/media/kien/D80079EB0079D14C/IPCV group/Datasets/trungnguyen-renamed/'
    labelpath = '../data/traffic-data/dataset2_countlabel.mat'
    h5_dict =  dataset_pack(datadir, 1000)
    label = scipy.io.loadmat(labelpath)
    label = label['Dataset']
    pdb.set_trace()
    label = label[0:1000,:]
    h5_dict['train_y'] = label[0:700,:]
    h5_dict['test_y'] = label[700:1000,:]

    pdb.set_trace()
    SaveH5(h5_dict, '../data/traffic-data/dataset2_count_aux.h5')
