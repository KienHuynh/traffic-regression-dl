import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util import LoadH5, SaveList, LoadList
import glob

import pdb

from net import RegressionCNN, grad_clip

prefix = '21-08-17'
use_cuda = True
init_lr = 0.0000125

#
num_fold = 5
num_e = 500
num_frac_test = 30

save_path = '../data/trained_model/traffic-regression-dl/dataset1/'
save_name = prefix + '-fold=%d_net'
meta_name = prefix + '-fold=%d_meta'
test_name = prefix + '-fold=%d_test'

def train_kfold_i(train_x, train_y, train_mean, fold_idx):
    """train_kfold_i
    This function is to train for a single fold of kfold cross validation
    :param train_x: train data
    :param train_y: train label
    :param train_mean: mean of the train data
    :param fold_idx: int, fold iteration
    """
    global use_cuda, save_path, save_name, meta_name

    # Create network & related training objects 
    net = RegressionCNN()
    if (use_cuda):
        net.cuda()
    
    init_lr = 0.0000125
    train_params = []
    for p in net.parameters(): 
        if (p.requires_grad):
            train_params.append(p)

    optimizer = optim.Adam(train_params, lr=init_lr)
    
    criterion = nn.MSELoss()
        
    # For debug purpose
    #train_x = train_x[0:2,:,:,:]
    #train_y = train_y[0:2,:] 

    # Put these numpy arrays into torch tensor
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    train_y = train_y[:,0]
 
    # Meta params
    batch_size = 2
    num_train = train_x.size()[0]
    num_ite_per_e = int(np.ceil(float(num_train)/float(batch_size)))
    full_ind = np.arange(num_train)
    rng = np.random.RandomState(1311) 
    all_loss = []
    num_e_to_reduce_lr = 20
    max_grad = 0.2 # For grad clip

    # Save params
    save_freq = 100
    
    save_name_local = save_name % fold_idx
    meta_name_local = meta_name % fold_idx
    save_list = glob.glob(save_path + save_name_local + '*.dat')
    last_e = -1

    # Printing and tracking params
    num_ite_to_log = 1 
    if (len(save_list) > 0): 
        save_list = sorted(save_list)[-1]
        print('Loading network save at %s' % save_list) 
        loadobj = torch.load(save_list)
        net.load_state_dict(loadobj['state_dict'])
        optimizer.load_state_dict(loadobj['opt'])
        del loadobj
        init_lr = optimizer.param_groups[0]['lr']
        save_list = sorted(glob.glob(save_path + meta_name_local + '*.dat'))[-1]
        last_e, running_loss, all_loss, train_mean = LoadList(save_list)  
    print('Current learning rate: %f' % init_lr)

    for e in range(last_e+1, num_e): 
        running_loss = 0.0

        # Divide lr by 1.5 after 200 epochv
        if (e % num_e_to_reduce_lr == (num_e_to_reduce_lr - 1)):
            init_lr = init_lr/1.5
            print('Current learning rate: %f' % init_lr)
            optimizer.param_groups[0]['lr'] = init_lr

        for i in range(num_ite_per_e): 
            rng.shuffle(full_ind)
            optimizer.zero_grad()
            
            if (i+1)*batch_size <= num_train:
                batch_range = range(i*batch_size, (i+1)*batch_size)
            else:
                batch_range = range(i*batch_size, num_train)
            batch_range = full_ind[batch_range]
            if (use_cuda):
                batch_x = Variable(train_x[torch.LongTensor(batch_range.tolist())].cuda())
                batch_y = Variable(train_y[torch.LongTensor(batch_range.tolist())].cuda())
            else: 
                batch_x = Variable(train_x[torch.LongTensor(batch_range.tolist())])
                batch_y = Variable(train_y[torch.LongTensor(batch_range.tolist())])
 
            outputs = net(batch_x)
	    #pdb.set_trace()
            loss = criterion(outputs, batch_y)
            loss.backward()
            grad_clip(net, max_grad)
            optimizer.step()
            
            running_loss += loss.data[0]
            all_loss.append(loss.data[0])
            
            if ((i % num_ite_to_log) == (num_ite_to_log-1)):
                print('%s - [%d, %d] loss: %.3f' % (save_name_local, e+1, e*num_ite_per_e+i+1, running_loss/num_ite_to_log)) 
                running_loss = 0.0

        if (e % save_freq == (save_freq-1)):
            abc = 1
            print('Saving at epoch %d' % e)
	    torch.save({'state_dict': net.state_dict(), 'opt': optimizer.state_dict()}, save_path + save_name_local + ('_%03d' % e) + '.dat')
            SaveList([e, running_loss, all_loss, train_mean], save_path + meta_name_local + ('_%03d' % e) + '.dat') 


def test_kfold_i(test_x, test_y, fold_idx, train, dropout_scale_factor, verbal, save):
    """test_kfold_i

    :param test_x: test data
    :param test_y: test label
    :param fold_idx: number of the current fold
    :param train: boolean, indicating if the network to run in train mode or not (only has effect for dropout and batchnorm)
    :param dropout_scale_factor: float, calculated from the fraction between train data prediction computed in train mode and the same computed in test mode
    :param save: boolean, tell the program to save the test result or not
    """
    global use_cuda, save_path, save_name, meta_name, test_name
    #dropout_scale_factor = 0.18
    # Create network & related training objects
    net = RegressionCNN()
    if (use_cuda):
        net.cuda() 

    # For debug purpose
    #train_x = train_x[0:4,:,:,:]
    #train_y = train_y[0:4,:] 
    #test_x = train_x
    #test_y = train_y
    

    # Put these numpy arrays into torch tensor
    num_test = test_y.shape[0]

    #train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    test_y = test_y[:,0] 
    test_pred = np.zeros((num_test, 1), dtype=np.float32)
    # Meta params
    batch_size = 1
    num_test = test_x.size()[0]
    num_ite_per_e = int(np.ceil(float(num_test)/float(batch_size)))
    full_ind = np.arange(num_test)
    rng = np.random.RandomState(1311) 
    all_loss = []

    # Save params 
    save_name_local = save_name % fold_idx
    test_name_local = test_name % fold_idx

    save_list = glob.glob(save_path + save_name_local + '*.dat')
    last_e = 0
    
    # Printing and tracking params
    num_ite_to_log = 10
    
    if (len(save_list) > 0): 
        save_list = sorted(save_list)[-1]
        print('Loading network save at %s' % save_list) 
        loadobj = torch.load(save_list)
        net.load_state_dict(loadobj['state_dict']) 
    
    if (not train):
        net.train(False)  
    for i in range(num_ite_per_e):   
        if (i+1)*batch_size <= num_test:
            batch_range = range(i*batch_size, (i+1)*batch_size)
        else:
            batch_range = range(i*batch_size, num_test)
        batch_range = full_ind[batch_range]
        if (use_cuda):
            batch_x = Variable(test_x[torch.LongTensor(batch_range.tolist())].cuda())
            
        else: 
            batch_x = Variable(test_x[torch.LongTensor(batch_range.tolist())])
            
        outputs = net.forward(batch_x)*dropout_scale_factor
        test_pred[batch_range] = outputs.data.cpu().numpy()
        
    test_y = test_y.numpy().reshape((num_test,1))
    ae = np.abs(test_pred - test_y)
    mae = np.mean(ae)
    mae_std = np.sqrt(np.mean((ae - mae)**2))
    re = np.abs(test_pred - test_y)/test_y
    mre = np.mean(re)
    mre_std = np.sqrt(np.mean((re-mre)**2))

    if (verbal):
        print('MAE: %.3f' % mae)
        print('MAE std: %.3f' % mae_std)
        print('MRE: %.3f' % mre)
        print('MRE std: %.3f' % mre_std)
    
    if (save):
        SaveList([test_pred, test_y, mae, mae_std, mre, mre_std, dropout_scale_factor], save_path + 'test_result/' + test_name_local + '.dat')
    
    return test_pred

def kfold():
    global use_cuda, save_path, save_name, meta_name, test_name      
    
    criterion = nn.MSELoss()

    # Load data
    h5_dict = LoadH5('../data/traffic-data/dataset1_count_aux.h5')
    train_x = h5_dict['train_x'].astype(np.float32)
    test_x = h5_dict['test_x'].astype(np.float32)
    train_y = h5_dict['train_y'].astype(np.float32)
    test_y = h5_dict['test_y'].astype(np.float32)
    train_mean = h5_dict['mean'].astype(np.float32)
    
    # Create all_x and all_y
    all_x = np.concatenate([train_x, test_x], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0) 
    n_sample = all_x.shape[0]

    n_sample_per_fold = n_sample/num_fold
    
    for i in range(0,num_fold):      
        test_idx = range(i*n_sample_per_fold,(i+1)*n_sample_per_fold)
        train_idx = range(0, n_sample)
        train_idx = [idx for idx in train_idx if (not(idx in test_idx))]
        train_idx = sorted(train_idx)
                
        train_x = all_x[train_idx,:,:,:]
        train_y = all_y[train_idx,:]
        test_x = all_x[test_idx,:,:,:]
        test_y = all_y[test_idx,:]
        
        # Compute train_mean and normalize the train dataset
        train_x = train_x.astype(np.float32)
        train_mean =np.mean(train_x, (0,2,3), np.float64)
        train_mean = train_mean.reshape(1,3,1,1)
        train_mean = train_mean.astype(np.float32)
        train_x = train_x - train_mean
        test_x = test_x - train_mean
        
        save_name_local = save_name % i
        save_list = glob.glob(save_path + '')
        train_kfold_i(train_x, train_y, train_mean, i)
        
        # Check if testing was done for that fold
        # If yes, skip it
        test_name_local = test_name % i
        test_list = glob.glob(save_path + 'test_result/' + test_name_local + '*.dat')
        if (len(test_list)==0):
            # Run prediction on train data with test-mode Dropout
            testmode_pred = test_kfold_i(train_x, train_y, i, False, 1, False, False)
            print('Computing prediction on train dataset with train mode off...')
            
            # Run prediction on train data for 30 times with train-mode dropout
            fracs = np.zeros((num_frac_test,1))
            for j in range(0,num_frac_test):
                trainmode_pred = test_kfold_i(train_x, train_y, i, True, 1, False, False)
                fracs[j] = np.sum(trainmode_pred)/np.sum(testmode_pred)
                print('Computing prediction on train dataset with train mode on... %d/%d' % (j+1, 30))

            fracs = np.mean(fracs)
            # Run prediction and evaluation on test data
            test_kfold_i(test_x, test_y, i, False, fracs, True, True)

if __name__ == '__main__':
    kfold()
