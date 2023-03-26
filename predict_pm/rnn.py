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
        self.lstm = nn.GRU(self.input_size,self.hidden_size,num_layers=num_layers,dropout=dropout)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x):
        return


class PmDataset(Dataset):
    def __init__(self,X,y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X,y


def read_data(data_path):
    input_txt = codecs.open(filename=data_path + "/input_data.json", mode="r", encoding='utf-8').read()
    target_txt = codecs.open(filename=data_path + "/target_data.json", mode="r", encoding='utf-8').read()
    input_data = json.loads(input_txt)
    input_data = torch.Tensor(input_data)
    target_data = json.loads(target_txt)
    target_data = torch.Tensor(target_data)
    return input_data, target_data


