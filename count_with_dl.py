import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util import LoadH5, SaveList, LoadList
import glob

from net import RegressionCNN, grad_clip

import pdb

prefix = '21-08-17'
use_cuda = True
init_lr = 0.0000125

#
num_fold = 5

# Save params
save_path = '../data/trained_model/traffic-regression-dl/dataset2/'
save_name = prefix + '_net' 
test_name = prefix + '_test'
meta_name = prefix + '_meta'

def train():
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
    #optimizer = optim.SGD(train_params, lr=init_lr, momentum=0.9)
    optimizer = optim.Adam(train_params, lr=init_lr)
    
    criterion = nn.MSELoss()

    # Load data
    h5_dict = LoadH5('../data/traffic-data/dataset2_count_aux.h5')
    train_x = h5_dict['train_x'].astype(np.float32)
    test_x = h5_dict['test_x'].astype(np.float32)
    train_y = h5_dict['train_y'].astype(np.float32)
    test_y = h5_dict['test_y'].astype(np.float32)
    train_mean = h5_dict['mean'].astype(np.float32)
    
    # For debug purpose
    #train_x = train_x[0:2,:,:,:]
    #train_y = train_y[0:2,:]

    # Mean norm
    train_x = train_x - train_mean
    test_x = test_x - train_mean

    # Put these numpy arrays into torch tensor
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    #test_x = torch.from_numpy(test_x)
    #test_y = torch.from_numpy(test_y)

    train_y = train_y[:,0]
    test_y = test_y[:,0]
 
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
    save_freq = 20 
    save_list = glob.glob(save_path + save_name + '*.dat')
    last_e = 0
    
    # Printing and tracking params
    num_ite_to_log = 1
  
    if (len(save_list) > 0): 
        save_list = sorted(save_list)[-1]
        print('Loading network save at %s' % save_list) 
        loadobj = torch.load(save_list)
        net.load_state_dict(loadobj['state_dict'])
        optimizer.load_state_dict(loadobj['opt'])
        init_lr = optimizer.param_groups[0]['lr']
        save_list = sorted(glob.glob(save_path + meta_name + '*.dat'))[-1]
        last_e, running_loss, all_loss = LoadList(save_list) 
    
    print('Current learning rate: %f' % init_lr)

    for e in range(last_e, 500): 
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
                print('[%d, %d] loss: %.3f' % (e+1, e*num_ite_per_e+i+1, running_loss/num_ite_to_log)) 
                running_loss = 0.0

        if (e % save_freq == (save_freq-1)):
            abc = 1
            print('Saving at epoch %d' % e)
	    torch.save({'state_dict': net.state_dict(), 'opt': optimizer.state_dict()}, save_path + save_name + ('_%03d' % e) + '.dat')
            SaveList([e, running_loss, all_loss], save_path + meta_name + ('_%03d' % e) + '.dat') 

def test():
    global prefix, use_cuda, save_path, save_name, test_name
    dropout_scale_factor = 0.4427
    # Create network & related training objects
    net = RegressionCNN()
    if (use_cuda):
        net.cuda() 

    # Load data
    h5_dict = LoadH5('../data/traffic-data/dataset2_count_aux.h5')
    train_x = h5_dict['train_x'].astype(np.float32)
    test_x = h5_dict['test_x'].astype(np.float32)
    train_y = h5_dict['train_y'].astype(np.float32)
    test_y = h5_dict['test_y'].astype(np.float32)
    train_mean = h5_dict['mean'].astype(np.float32)
    
    # For debug purpose
    #train_x = train_x[0:4,:,:,:]
    #train_y = train_y[0:4,:]

    # Mean norm 
    test_x = test_x - train_mean
    train_x = train_x - train_mean
    
    # Use train data as test data to debug
    # If the error is still high, there must be an error somewhere
    # Turn it off after debug is done
    #test_x = train_x
    #test_y = train_y

    num_test = test_y.shape[0]
    
    # Put these numpy arrays into torch tensor
    #train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    test_y = test_y[:,0] 
    test_pred = np.zeros((num_test, 1), dtype=np.float32)
    # Meta params
    batch_size = 2
    num_test = test_x.size()[0]
    num_ite_per_e = int(np.ceil(float(num_test)/float(batch_size)))
    full_ind = np.arange(num_test)
    rng = np.random.RandomState(1311) 
    all_loss = []

    # Save params 
    save_list = glob.glob(save_path + save_name + '*.dat')
    last_e = 0 
    
    if (len(save_list) > 0): 
        save_list = sorted(save_list)[-5]
        print('Loading network save at %s' % save_list) 
        loadobj = torch.load(save_list)
        net.load_state_dict(loadobj['state_dict']) 
    
    train = False
    if (not train):
        net.train(False) 
    #net.eval()
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
            
        outputs = net.forward(batch_x)/(1-dropout_scale_factor)

        test_pred[batch_range] = outputs.data.cpu().numpy()
        
    pdb.set_trace()
    test_y = test_y.numpy().reshape((num_test,1))
    ae = np.abs(test_pred - test_y)
    mae = np.mean(ae)
    mae_std = np.sqrt(np.mean((ae - mae)**2))
    re = np.abs(test_pred - test_y)/test_y
    mre = np.mean(re)
    mre_std = np.sqrt(np.mean((re-mre)**2))
    print('MAE: %.3f' % mae)
    print('MAE std: %.3f' % mae_std)
    print('MRE: %.3f' % mre)
    print('MRE std: %.3f' % mre_std)

    SaveList([test_pred, test_y, mae, mae_std, mre, mre_std], save_path + 'test_result/' + test_name + '.dat')

if __name__ == '__main__':
    #kfold()
    #train() 
    test()
