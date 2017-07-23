import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util import LoadH5, SaveList, LoadList
import glob

import pdb

class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        # Load weight values from file
        self.vgg = {} 
        self.vgg_conv1_1 = nn.Conv2d(3, 64, 3, bias=True, padding=1)
        self.vgg_conv1_2 = nn.Conv2d(64, 64, 3, bias=True, padding=1)
        
        self.vgg_conv2_1 = nn.Conv2d(64, 128, 3, bias=True, padding=1)
        self.vgg_conv2_2 = nn.Conv2d(128, 128, 3, bias=True, padding=1)
        
        self.vgg_conv3_1 = nn.Conv2d(128, 256, 3, bias=True, padding=1)
        self.vgg_conv3_2 = nn.Conv2d(256, 256, 3, bias=True, padding=1)
        self.vgg_conv3_3 = nn.Conv2d(256, 256, 3, bias=True, padding=1)
        
        self.vgg_conv4_1 = nn.Conv2d(256, 512, 3, bias=True, padding=1)
        self.vgg_conv4_2 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg_conv4_3 = nn.Conv2d(512, 512, 3, bias=True, padding=1)

        self.vgg_conv5_1 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg_conv5_2 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg_conv5_3 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        
        self.vgg_conv6_1 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg_conv6_2 = nn.Conv2d(512, 512, 3, bias=True, padding=1)
        self.vgg_conv6_3 = nn.Conv2d(512, 512, 3, bias=True, padding=1)

        self.vgg_fc6 = nn.Linear(8*8*512, 4096)
        self.vgg_fc7 = nn.Linear(4096, 4096)
        self.vgg_fc8 = nn.Linear(4096, 1)
       
        self.vgg_pool = nn.MaxPool2d(2,2)

    def forward(self, x): 
        x = F.relu(self.vgg_conv1_1(x))
        x = F.relu(self.vgg_conv1_2(x))
        x = self.vgg_pool(x)
        
        x = F.relu(self.vgg_conv2_1(x))
        x = F.relu(self.vgg_conv2_2(x))
        x = self.vgg_pool(x)

        x = F.relu(self.vgg_conv3_1(x))
        x = F.relu(self.vgg_conv3_2(x))
        x = F.relu(self.vgg_conv3_3(x))
        x = self.vgg_pool(x)
        
        x = F.relu(self.vgg_conv4_1(x))
        x = F.relu(self.vgg_conv4_2(x))
        x = F.relu(self.vgg_conv4_3(x))
        x = self.vgg_pool(x)

        x = F.relu(self.vgg_conv5_1(x))
        x = F.relu(self.vgg_conv5_2(x))
        x = F.relu(self.vgg_conv5_3(x))
        x = self.vgg_pool(x)
        
        x = F.relu(self.vgg_conv6_1(x))
        x = F.relu(self.vgg_conv6_2(x))
        x = F.relu(self.vgg_conv6_3(x))
        x = self.vgg_pool(x)

        x = x.view(-1, 8*8*512)
        x = self.vgg_fc6(x)
        x = self.vgg_fc7(x)
        x = self.vgg_fc8(x)
        
        return x



if __name__ == '__main__':
    use_cuda = False

    # Create network & related training objects
    net = RegressionCNN()
    if (use_cuda):
        net.cuda()
    
    init_lr = 0.001
    train_params = []
    for p in net.parameters(): 
        if (p.requires_grad):
            train_params.append(p)
    optimizer = optim.SGD(train_params, lr=init_lr, momentum=0.9)
    criterion = nn.MSELoss()

    # Load data
    h5_dict = LoadH5('../data/traffic-data/dataset1_count_aux.h5')
    train_x = h5_dict['train_x'].astype(np.float32)
    test_x = h5_dict['test_x'].astype(np.float32)
    train_y = h5_dict['train_y'].astype(np.float32)
    test_y = h5_dict['test_y'].astype(np.float32)
    train_mean = h5_dict['mean'].astype(np.float32)
    
    # For debug purpose
    train_x = train_x[0:4,:,:,:]
    train_y = train_y[0:4,:]

    # Mean norm
    train_x = train_x - train_mean
    test_x = test_x - train_mean

    # Put these numpy arrays into torch tensor
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    train_y = train_y[:,0]
    test_y = test_y[:,0]
 
    # Meta params
    batch_size = 2
    num_train = train_x.size()[0]
    num_test = test_x.size()[0]
    num_ite_per_e = int(np.ceil(float(num_train)/float(batch_size)))
    full_ind = np.arange(num_train)
    rng = np.random.RandomState(1311) 
    all_loss = []
    num_e_to_reduce_lr = 200

    # Save params
    save_freq = 5
    save_path = '../data/trained_model/traffic-regression-dl/'
    save_name = '24-07-17'
    meta_name = save_name + '_meta'
    save_list = glob.glob(save_path + save_name + '*.dat')
    last_e = 0
    
    # Printing and tracking params
    num_ite_to_log = 10


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

    for e in range(1000): 
        running_loss = 0.0

        # Divide lr by 1.5 after 200 epochv
        if (e % num_e_to_reduce_lr == (num_e_to_reduce_lr - 1)):
            init_lr = init_lr/1.5
            print('Current learning rate: %f' % init_lr)
            optimizer.param_groups[0]['lr'] = init_lr

        for i in range(num_ite_per_e): 
            #rng.shuffle(full_ind)
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
                pdb.set_trace()
                batch_x = Variable(train_x[torch.LongTensor(batch_range.tolist())])
                batch_y = Variable(train_y[torch.LongTensor(batch_range.tolist())])

            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            all_loss.append(loss.data[0])
            
            if ((i % num_ite_to_log) == (num_ite_to_log-1)):
                print('[%d, %d] loss: %.3f' % (e+1, e*num_ite_per_e+i+1, running_loss/100)) 
                running_loss = 0.0

        if (e % save_freq == (save_freq-1)):
            torch.save({'state_dict': net.state_dict(), 'opt': optimizer.state_dict()}, save_path + save_name + ('%03d' % e) + '.dat')
            SaveList([e, running_loss, all_loss], save_path + meta_name + ('%03d' % e) + '.dat') 