# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:39:37 2022
@author: armelle, enzo
"""

# overfitting

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

dn = 50.
h=20
nepochs=100
lr = 0.0001

'''
#Data: convert to tensor
with open("meteo/2019.csv","r") as f: ls=f.readlines()
trainx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-7]]).view(1,-1,1)
trainy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[7:]]).view(1,-1,1)
#print(trainx.shape)
#print(trainx[0,0,0])
with open("meteo/2020.csv","r") as f: ls=f.readlines()
testx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-7]]).view(1,-1,1)
testy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[7:]]).view(1,-1,1)

#Tensor to Dataset
'''
num = 2
decal = 2
dn = 10000.

train, val, test = np.load(open('jr_val_train'+'%s'%num+'.pkl', 'rb'),allow_pickle=True), np.load(open('jr_val_val'+'%s'%num+'.pkl', 'rb'),allow_pickle=True), np.load(open('jr_val_test'+'%s'%num+'.pkl', 'rb'),allow_pickle=True)
#trainx, trainy, valx, valy, testx, testy = torch.Tensor(train[:-decal]).view(1,-1,1), torch.Tensor(train[decal:]).view(1,-1,1), torch.Tensor(val[:-decal]).view(1,-1,1), torch.Tensor(train[decal:]).view(1,-1,1), torch.Tensor(test[:-decal]).view(1,-1,1), torch.Tensor(test[decal:]).view(1,-1,1)
#train_year=2020, val_year=2019, test_year=2021
print(train[3, 0])
train[3, 0] = train[3, 0]/dn
print(train[3, 0])
for i in range(train.shape[0]):
    train[i, 0] = train[i, 0]/dn
for i in range(val.shape[0]):
    val[i, 0] = val[i, 0]/dn
for i in range(test.shape[0]):
    test[i, 0] = test[i, 0]/dn
trainx, trainy, testx, testy = torch.Tensor(val).view(1,-1,1), torch.Tensor(train).view(1,-1,1), torch.Tensor(train).view(1,-1,1), torch.Tensor(test).view(1,-1,1)

    
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)

testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)

#on calcule la mean scare error à chaque fois 
crit = nn.MSELoss()

#Model: simple RNN with a linear regression layer:

class Rnn(nn.Module):
    def __init__(self,nhid):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(1,nhid)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.rnn(xx)
        T,B,H = y.shape
        
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y
    
    
def r2_score(output, target):
    # print(output.detach().numpy()[0])
    return 1- torch.sum((target-output.detach().numpy())**2) / torch.sum((target-target.float().mean())**2)

#compute MSE on test:
def test(mod):
    testloss, testr2score, nbatch = 0., 0., 0
    for data2 in testloader:
        inputs2, goldy2 = data2
        inputs2 = inputs2[:,:-1,:]
        print("haty",inputs2.size())
        print("goldy",goldy2.size())
        #goldyy2 = goldy2[:,:-2,:] # QUE POUR LE CNN
        haty2 = mod(inputs2)
        loss2 = crit(haty2,goldy2)
        testr2score += r2_score(haty2,goldy2)
        testloss += loss2.item()
        nbatch += 1
    testr2score /= float(nbatch)
    testloss /= float(nbatch)
    return testloss, testr2score

#Training loop:
def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr)
    plot_values = {"epoch": [], "loss": [], "test_loss":[], "r2score":[], "test_r2score":[]}
    for epoch in range(nepochs):
        mod.train(True)
        totloss, totr2score, nbatch = 0., 0., 0
        for data in trainloader:
            inputs, goldy = data
            goldy = goldy[:,:-1,:]
            #goldyy = goldy[:,:-2,:] # QUE POUR LE CNN
            optim.zero_grad()
            haty = mod(inputs)
            print("haty",haty.size())
            print("goldy",goldy.size())
            loss = crit(haty,goldy)
            totr2score += r2_score(haty, goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
            
        totloss /= float(nbatch)
        totr2score /= float(nbatch)
        mod.train(False)
        testloss, testr2score = test(mod)
        plot_values["epoch"].append(epoch)
        plot_values["loss"].append(totloss)
        plot_values["test_loss"].append(testloss)
        plot_values["r2score"].append(totr2score)
        plot_values["test_r2score"].append(testr2score)
            
        print(epoch,"err",totloss,testloss)
        
    # Plot loss evolution
    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(plot_values["epoch"], plot_values["loss"], 'b', label='training loss')
    ax1.plot(plot_values["epoch"], plot_values["test_loss"], 'g', label='validation loss')
    ax1.set(xlabel='epoch', ylabel='MSE loss', title='Training supervision for lr = %s'%lr)
    ax1.axis(ymin=0)
    ax1.grid()
    ax1.legend()
    
    ax2.plot(plot_values["epoch"], plot_values["r2score"], 'b', label='training r2 score')
    ax2.plot(plot_values["epoch"], plot_values["test_r2score"], 'g', label='validation r2 score')
    ax2.set(xlabel='epoch', ylabel='r2 score')
    ax2.axis(ymin=0)
    ax2.grid()
    ax2.legend()
    print(f"\nfin de l'entrainement\nMSE loss : train = {totloss:.3g}   test = {testloss:.3g}\nR2 score : train = {totr2score:.3g}     test = {testr2score:.3g}")


class Cnn(nn.Module):
    def __init__(self,nhid):
        super(Cnn, self).__init__()
        self.cnn = nn.Conv1d(1,nhid,3,1)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # B,T,D need B,D,T
        # print("x : ", x.shape)
        y = x.transpose(1,2)
        y = self.cnn(y)
        y = torch.relu(y)
        # B,D,T need B*T,D
        B,D,T = y.shape
        y = y.transpose(1,2)
        y = y.view(-1,D)
        # print("y0 :     ", y.shape)
        y = self.mlp(y)
        # print("y1 :     ", y.shape)
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        # print("y2 :     ", y.shape)
        return y


#The Main
mod=Rnn(h)
start_time = time.time()
train(mod)
print(f"Training time : {(time.time() - start_time)} seconds for {nepochs}")
plt.show()