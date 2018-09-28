# coding:utf8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import time
import os
from data.dataset import DefectsData
from torch.utils.data import DataLoader
from config import DefaultConfig
from torch.utils.data.sampler import  WeightedRandomSampler
from focal_loss import FocalLoss

# from utils.visualize import Visualizer
from config import opt
import models
import pandas as pd

def train():
    # opt.parse(kwargs)
    since = time.time()


    """
        class_dic = {'正常': 0, '涂层开裂': 1, '横条压凹': 2, '桔皮': 3, 
                   '擦花': 4, '漏底': 5, '凸粉': 6, '不导电': 7, '起坑': 8,
                   '碰伤':9, '脏点': 10, '其他': 11}
        number_dic = {
            1 : 20 : '涂层开裂': 36, '横条压凹': 48, '不导电': 40, '起坑': 55,
                        [1,2,7,8]
            2 : 10 : '桔皮': 91, '碰伤':78, '凸粉': 44+61-39
                        [3,6,9]
            3 : 7  : '擦花': 137, '其他': 140+20-13
                        [4,11]
            4 : 4  : '漏底': 249+97-76, '脏点': 210,
                        [5,10]
            5 : 1  : '正常': 1019,  }
                        [0]
    """
    # step1: load model
    # model = models.resnet34()
    model = getattr(models, opt.model)()
    if opt.use_gpu: model.cuda()
    print("model loaded")

    best_model_wts = model.state_dict()
    # step2: get dataset
    trainset = DefectsData(opt.train_data_root, transforms=None, train=True)
    valset = DefectsData(opt.train_data_root, transforms=None, train=False)

    print('train_set_len: ', len(trainset))
    print('valid_set_len: ', len(valset))
    
    weights1=[]
    for data, label in trainset:
        if label in [1,2,7,8]:
            weights1.append(20)
        elif label in [3,6,9]:
            weights1.append(10)
        elif label in [4,11]:
            weights1.append(7)
        elif label in [5,10]:
            weights1.append(4)
        else:
            weights1.append(1)

    weights2=[]
    for data, label in valset:
        if label in [1,2,7,8]:
            weights2.append(20)
        elif label in [3,6,9]:
            weights2.append(10)
        elif label in [4,11]:
            weights2.append(7)
        elif label in [5,10]:
            weights2.append(4)
        else:
            weights2.append(1)

    sampler1 = WeightedRandomSampler(weights1,\
                                num_samples=opt.sampler_num1,\
                                replacement=True)

    sampler2 = WeightedRandomSampler(weights2,\
                                num_samples=opt.sampler_num2,\
                                replacement=True)

    #load dataset
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=opt.num_workers,sampler=sampler1)
    valloader = DataLoader(valset, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=opt.num_workers,sampler=sampler2)




    # step3: define loss function and optimize
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(class_num=12)
    if opt.use_gpu: criterion.cuda()

    # define optimize method
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)  # , weight_decay=5e-4
    # define learning rate strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=4,verbose=True,min_lr=1e-8)

    # step4: 
    
    best_loss = 1e10

    # step5 :train 
    for epoch in range(opt.max_epoch):
        print('Epoch {}:'.format(epoch))
        epc_time = time.time()
        model.train()
        # ----------training-------------------
        train_loss = 0.
        train_acc = 0.
        for ii,(data,label) in enumerate(trainloader):
            inputs = Variable(data)
            labels = Variable(label)
            if opt.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            #forward propogation for network
            # print(inputs.size())

            #reset all grad in this network
            optimizer.zero_grad()
            #atain loss
            out = model(inputs)
            # print(out,labels,out.size(),labels.size())
            # predd = torch.max(out, 1)
            # print(predd)
            loss = criterion(out, labels)
            train_loss += loss.item()

            pred = torch.max(out, 1)[1]
            # print('real label:',labels)
            # print('pred label:',pred)
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
            

            #backward propogation
            loss.backward()
            optimizer.step()


            #
            #print('batch [{:.0f}|{:.0f}]: Train Loss: {:.6f}, Acc: {:.6f}'.format(ii, len(trainset)//(opt.batch_size),
             #   train_loss / opt.sampler_num1, train_acc / opt.sampler_num1))

            if (epoch+1) % opt.save_epoch_freq == 0:
                if not os.path.exists(opt.save_path):
                    os.makedirs(opt.save_path)
                torch.save(model, os.path.join(opt.save_path, "epoch_" + str(epoch) + ".pth.tar"))
        #print('---->Epoch {:d}:--> Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch,
        #        train_loss / (len(trainset)), train_acc / (len(trainset))))
        print('---->Epoch {:d}:--> Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch,
                train_loss / opt.sampler_num1, train_acc / opt.sampler_num1))

        # val:

        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for ii, data in enumerate(valloader):
            inputs_val, labels_val = data
            val_input = Variable(inputs_val)
            val_label = Variable(labels_val)
            
            if opt.use_gpu:
                val_input = val_input.cuda()
                val_label = val_label.cuda()

            out_val = model(val_input)
            loss = criterion(out_val, val_label)

            eval_loss += loss.item()
            pred_val = torch.max(out_val, 1)[1]
            # print('val label:',val_label)
            # print('pre label:',pred_val)
            num_correct = (pred_val == val_label).sum()
            eval_acc += num_correct.item()

        scheduler.step(eval_loss/opt.sampler_num2)

        print('---->Val Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (opt.sampler_num2), eval_acc / (opt.sampler_num2)))
        if best_loss >= (eval_loss / (opt.sampler_num2)):
             best_loss = (eval_loss / (opt.sampler_num2))
             best_model_wts = model.state_dict()
             torch.save(model,opt.save_path+"best_model.pkl")
             print('best model update.')
        time_ed = time.time() - epc_time
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_ed // 60, time_ed % 60))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def test(model):

    clas_dic ={
    0:'norm', 7:'defect1',4:'defect2',
    2:'defect3',3:'defect4',5:'defect5',
    9:'defect6',8:'defect7',6:'defect8',
    1:'defect9',10:'defect10',11:'defect11' }
    # opt.parse(kwargs)

    # step1: load model
    model.eval()
    if opt.use_gpu: model.cuda()
    print('model loaded')

    # step2: get dataset
    test_data = DefectsData(opt.test_data_root, transforms=None, test=True)

    #load dataset
    testloader = DataLoader(test_data, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=opt.num_workers)

    names,labels_ = [],[]
    for ii ,(data,name_) in enumerate(testloader):

        inputs = torch.autograd.Variable(data)
        if opt.use_gpu: inputs = inputs.cuda()

        # print('inputs:',inputs.size())
        score = model(inputs)
        pred_ = torch.max(score, 1)[1]
        # print('soc:',pred_.size())
        # print(pred_.tolist())
        # print(name_)
        for a,b in zip(list(name_),pred_.tolist()):
            names.append(a)
            labels_.append(clas_dic[b])
       
    # write_csv:
    """
    class_dic = {'正常': 0, '涂层开裂': 1, '横条压凹': 2, 
    '桔皮': 3,'擦花': 4, '漏底': 5, '凸粉': 6, '不导电': 7,
     '起坑': 8,'碰伤':9, '脏点': 10, '其他': 11}
    """

    df = pd.DataFrame({'name':names,'label':labels_},index=None)
    df.to_csv('tmp.csv',header=False,index=False)
    df = pd.read_csv('tmp.csv',header=None)
    df[0] = df[0].str.split('.jpg').str[0]
    df.columns = ['name', 'cls']
    df['name']  = df['name'].astype(int)
    df.sort_values(by="name",axis=0,inplace=True)
    df['name'] = df['name'].map(lambda x : str(x)+'.jpg')
    df.to_csv(opt.result_file,header=False,index=False)

    return


if __name__=='__main__':
    model = train()
    test(model)

