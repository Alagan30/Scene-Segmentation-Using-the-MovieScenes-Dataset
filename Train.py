# -*- coding: utf-8 -*-
"""

Program to train the model for Scene Segmentation Using the MovieScenes Dataset.

Created on Sat Mar  6 14:41:21 2021

@author: alaga
"""

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import os, glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from matplotlib import pyplot as plt



def read_files(filename):
    
    with open('./files/'+ filename, 'rb') as f:
        data = pkl.load(f)
    
    place = data['place']
    cast = data['cast']
    action = data['action']
    audio = data['audio']
    
    ground_truth = data['scene_transition_boundary_ground_truth']
    start_label = torch.tensor([0])
    ground_truth = torch.cat((start_label, ground_truth), dim = 0)
    
    return place, cast, action, audio, ground_truth



class custom_nn(nn.Module):
    
    def __init__(self, inter_dim = 128):
        super(custom_nn, self).__init__()
        
        self.layer1 = nn.Linear(2048, inter_dim)
        self.layer2 = nn.Linear(512, inter_dim)
        self.layer3 = nn.Linear(512, inter_dim)
        self.layer4 = nn.Linear(512, inter_dim)
        
        self.final_layer = nn.Linear(4*inter_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    
    def forward(self, X1, X2, X3, X4):
        out1 = self.relu(self.layer1(X1))
        out2 = self.relu(self.layer2(X2))
        out3 = self.relu(self.layer2(X3))
        out4 = self.relu(self.layer2(X4))
        
        out = torch.cat((out1, out2, out3, out4), dim=1)
        
        final_out = self.sigmoid(self.final_layer(out))
    
        return final_out        


class index_class(torch.utils.data.Dataset):
    
    def __init__(self, ids):
        'Initialization'
        self.list_IDs = ids

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        return ID



def main():
    
    data_dir = os.getcwd()
    file_list = glob.glob(os.path.join(data_dir, "tt*.pkl"))
    
    place_list, cast_list, action_list, audio_list, y = [], [], [], [], []
    
    for i in range(len(file_list)):
        place, cast, action, audio, ground_truth = read_files(file_list[i])
        
        place_list.append(place)
        cast_list.append(cast)
        action_list.append(action)
        audio_list.append(audio)
        y.append(ground_truth)
    
    place = torch.cat(place_list, dim=0)
    cast = torch.cat(cast_list, dim=0)
    action = torch.cat(action_list, dim=0)
    audio = torch.cat(audio_list, dim=0)
    y = torch.cat(y, dim=0)

    
    tr_ids, val_ids = train_test_split(range(0, place.shape[0]), test_size = 0.1, shuffle=False)
    training_set = index_class(tr_ids)
    validation_set = index_class(val_ids)
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)
    
    model = custom_nn()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    
    epochs = 150
    loss_func = nn.BCELoss()
    train_loss = []
    valid_loss = []
    
    for i in range(epochs):
        temp = 0
        actual_list, pred_list = [], []
        model.train()
        for j,ids in enumerate(train_loader):
            X1 = place[ids]
            X2 = cast[ids]
            X3 = action[ids]
            X4 = audio[ids]
            
            labels = y[ids]
            labels = labels.to(torch.float32)
            
            pred_labels = model(X1, X2, X3, X4)
            
            loss = loss_func(pred_labels.squeeze(1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            temp += loss.item()
            actual_list.append(labels.data.numpy())
            pred_list.append(pred_labels.squeeze(1).data.numpy())
        
        actual_list = np.concatenate(actual_list)
        pred_list = np.concatenate(pred_list)
        
        actual_list = [int(i) for i in actual_list]
        pred_list = [int(np.round(i)) for i in pred_list]
        
        train_loss.append(temp/len(train_loader))
        # print("Epoch " + str(i) + " training loss = ",temp/len(train_loader))
        print("Epoch " + str(i) + " Training Acuracy: ", balanced_accuracy_score(actual_list, pred_list))
        
        temp = 0
        actual_list, pred_list = [], []
        model.eval()
        with torch.no_grad():
            for j,ids in enumerate(val_loader):
                X1 = place[ids] 
                X2 = cast[ids]
                X3 = action[ids]
                X4 = audio[ids]
                
                labels = y[ids]
                labels = labels.to(torch.float32)
                
                pred_labels = model(X1, X2, X3, X4)
                
                loss = loss_func(pred_labels.squeeze(1), labels)
                temp += loss.item()
                
                actual_list.append(labels.data.numpy())
                pred_list.append(pred_labels.squeeze(1).data.numpy())
        
        actual_list = np.concatenate(actual_list)
        pred_list = np.concatenate(pred_list)
        
        actual_list = [int(i) for i in actual_list]
        pred_list = [int(np.round(i)) for i in pred_list]
        
        valid_loss.append(temp/len(val_loader))
        # print("Epoch " + str(i) + " validation loss = ",temp/len(val_loader))
        print("Validation Accuracy: ", balanced_accuracy_score(actual_list, pred_list))
        print("Confusion Matrix: ", confusion_matrix(actual_list, pred_list))


    epochs = range(150)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    data_dir = os.getcwd()
    torch.save(model, os.path.join(data_dir,"model"))
    
    
if __name__=='__main__':
    main()
