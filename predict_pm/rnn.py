import json
import codecs
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from process_data import process_data

class RecurrentNeuralNetwork(nn.Module):
    
    def __init__(self,rnn_layers,hidden_size,in_size,out_size,num_layers,dropout) -> None:
        super().__init__()
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size

        # Add LSTM models
        self.lstm = nn.GRU(self.input_size,self.hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,self.output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        batch_size = x[0]
        h_0 = torch.zeros(self.rnn_layers,batch_size,self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.rnn_layers,batch_size,self.hidden_size).requires_grad_()
        output, (h_n, c_n) = self.lstm(x,(h_0,c_0))
        output = self.ReLU(output)
        output = self.dense(self.hidden_size, self.out_size)
        return output.flatten()


class PMDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        #self.data = data
        #self.sequence_length = sequence_length
        #self.target_length = target_length
        self.x = x
        self.y = y

    def __len__(self):
        # This should return the length of all sequences in each datapoint
        # sequences = 0
        # for csv in self.data:
        #     total_seqs += len(csv) - self.sequence_length + 1
        return len(x)
    
    def __getitem__(self, pos):
        # Make this index the starting index
        # dp,idx = pos
        # csv = self.data[dp]
        # if len(csv) > idx + self.sequence_length + self.target_length:
        #     x = csv[idx:idx+self.sequence_length]
        #     y = csv[idx+self.sequence_length:idx+self.sequence_length+self.target_length]

        # else:
        #     # If index is too big pass the seqence until the end and then zeros?
        #     # Number of zeros is equal to the 
        #     if len(csv) <= idx + self.sequence_length:
        #         padding_length = (idx + self.sequence_length) - len(csv) + 1
        #         padding = np.zeros(padding_length)
        #         csv = torch.cat((padding, csv), 0)
        #         x = csv[idx:idx+self.sequence_length]
        #         y = np.zeros(self.target_length)
        #     # There are enough x values but not enough y values to predict
        #     else:
        #         x = csv[idx:idx+self.sequence_length]
        #         padding_length = (idx + self.sequence_length + self.target_length) - len(csv) + 1
        #         padding = np.zeros(padding_length)
        #         csv = torch.cat((padding, csv), 0)
        #         y = csv[idx+self.sequence_length:idx+self.sequence_length+self.target_length]
        return torch.Tensor(self.x[pos]), torch.Tensor(self.y[pos])
    

def train_model(epochs,lr,loss_function,optimizer,data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurrentNeuralNetwork().to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model()
        loss = loss_function()
    return



def read_data(data_path):
    data_txt = codecs.open(filename=data_path + "/train_data.json", mode="r", encoding='utf-8').read()
    data = json.loads(data_txt)
    data = torch.Tensor(data)
    target_txt = codecs.open(filename=data_path + "/target_data.json", mode="r", encoding='utf-8').read()
    target_data = json.loads(target_txt)
    target_data = torch.Tensor(target_data)
    val_index = int(0.8*len(data))
    test_index = int(0.9*len(data))
    X_train = data[:val_index]
    X_val = data[val_index:test_index]
    X_test = data[test_index:]
    y_train = target_data[:val_index]
    y_val = target_data[val_index:test_index]
    y_test = target_data[test_index:]
    train_dataset = PMDataset(X_train,y_train)
    val_dataset = PMDataset(X_val,y_val)
    test_dataset = PMDataset(X_test,y_test)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=4,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)
    return train_loader, val_loader, test_loader


def get_processed_data():
    X,y = process_data.split_all_files("pm_data/","processed_data/train_data",dp_length=48,pred_length=6)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    val_index = int(0.8*len(X))
    test_index = int(0.9*len(X))
    X_train = X[:val_index]
    X_val = X[val_index:test_index]
    X_test = X[test_index:]
    y_train = y[:val_index]
    y_val = y[val_index:test_index]
    y_test = y[test_index:]
    train_loader = PMDataset(X_train,y_train)
    val_loader = PMDataset(X_val,y_val)
    test_loader = PMDataset(X_test,y_test)
    return train_loader, val_loader, test_loader

