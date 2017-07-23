import numpy as np
import h5py
import os
import pdb

def create_one_hot(target_vector, num_class, dtype=np.float32):
    """create_one_hot
    Generate one-hot 4D tensor from a target vector of length N (num sample)
    The one-hot tensor will have the shape of (N x 1 x 1 x num_class)

    :param target_vector: Index vector, values are ranged from 0 to num_class-1

    :param num_class: number of classes/labels
    :return: target vector as a 4D tensor
    """
    one_hot = np.eye(num_class+1, num_class, dtype=dtype)
    one_hot = one_hot[target_vector]
    result = np.reshape(one_hot, (target_vector.shape[0], 1, 1, num_class))
    
    return result

def SaveH5(obj, file_name):
    """ SaveH5
    Save numpy data to HDF5 file
    Use this when pickle can't save large file 
    :param obj: dict of numpy arrays
    :param file_name: file name
    """
    with h5py.File(file_name, 'w') as hf:
        for k, v in obj.iteritems():
            hf.create_dataset(k, data=v)


def LoadH5(file_name):
    """ LoadH5
    Load numpy data from HDF5 file 
    :param obj: dict of numpy arrays
    :param file_name: file name
    """
    obj = {}
    with h5py.File(file_name, 'r') as hf:
        for k in hf.keys():
            obj[k] = np.asarray(hf.get(k))

    return obj

def SaveList(list_obj, file_name, type='pickle'):
    """ Save a list object to file_name
    :type list_obj: list
    :param list_obj: List of objects to be saved.*

    :type file_name: str
    :param file_name: file name

    :type type: str
    :param type: 'pickle' or 'hdf5'
    """

    if (type == 'pickle'):
        f = open(file_name, 'wb')
        for obj in list_obj:
            cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    # elif (type == 'hdf5' or type == 'h5'):
    #     with h5py.File(file_name, 'w') as hf:
    #         hf.create_dataset('data', data=list_obj)
    else:
        print('Encoding type not recognized, type should be \'pickle\' or \'hdf5\'')


def LoadList(file_name, type='pickle'):
    """ Load a list object to file_name
    :type file_name: str
    :param file_name: file name

    :type type: str
    :param type: 'pickle' or 'hdf5'
    """
    if (type == 'pickle'):
        end_of_file = False
        list_obj = []
        f = open(file_name, 'rb')
        while (not end_of_file):
            try:
                list_obj.append(cPickle.load(f))
            except EOFError:
                end_of_file = True
                print("EOF Reached")

        f.close()
        return list_obj
    # elif (type == 'hdf5' or type == 'h5'):
    #     with h5py.File(file_name, 'r') as hf:
    #         list_obj = hf.get('data')
    #         return list_obj
    else:
        print('Encoding type not recognized, type should be \'pickle\' or \'hdf5\'')

