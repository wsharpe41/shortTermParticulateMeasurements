import json
import codecs
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self,data,sequence_length) -> None:
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)
    
    # INCORRECT FOR NOW
    def __getitem__(self, index):
        # Both of these need to be longer than 1 
        # Have index be the ending index
        if index > self.sequence_length:
            x = self.data[index-self.sequence_length:index]
        else:
            x = self.data[0]
        return x


def read_data(data_path):
    input_txt = codecs.open(filename=data_path + "/input_data.json", mode="r", encoding='utf-8').read()
    target_txt = codecs.open(filename=data_path + "/target_data.json", mode="r", encoding='utf-8').read()
    input_data = json.loads(input_txt)
    train_data = torch.Tensor(input_data)
    target_data = json.loads(target_txt)
    target_data = torch.Tensor(target_data)

    train_dataset = PMDataset(train_data)
    test_dataset = PMDataset(target_data)
    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=True)
    return input_data, target_data

# Need to make dataloaders and batch data before trying to train the RNN/GRU/LSTM
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
