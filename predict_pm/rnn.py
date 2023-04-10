import json
import codecs
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError

class RNN(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout,l1=256,l2=128) -> None:
        super().__init__()
        #self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        # Add LSTM models
        self.lstm = nn.RNN(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,l1)
        self.dense2 = nn.Linear(l1,l2)
        self.dense3 = nn.Linear(l2,self.output_size)

        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        x= x.unsqueeze(2)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        #c_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        output, h_n = self.lstm(x,h_0)
        output = self.relu(output)
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.dense2(output)
        output = self.dense3(output)

        return output
    
    def train_model(self,epochs,lr,loss_function,data_loader,val_loader):
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loss = []
        val_loss = []
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
        num_batches = len(data_loader)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X)
                loss = loss_function(pred,y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() / num_batches
            print(f"Train Loss for epoch {i}: {epoch_loss}")
            train_loss.append(epoch_loss)
            
            for X,y in val_loader:
                pred = self(X)
                loss = loss_function(pred,y)
                epoch_loss += loss.item() / num_batches
            print(f"Val Loss for epoch {i}: {epoch_loss}")
            val_loss.append(epoch_loss)

        return train_loss, val_loss
    
    def test_model(self,loss_function,data_loader):
        # Just get the loss
        self.eval()
        test_loss = 0.0
        all_losses = []
        pred_v_actual = [[],[]]
        num_batches = len(data_loader)
        with torch.no_grad():
            for X,y in data_loader:
                pred = self(X)
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y)
                batch_loss=loss_function(pred,y).item()/num_batches
                test_loss += batch_loss
                all_losses.append(batch_loss)

        print(f"Avg Test Loss: {test_loss}")
        return test_loss, all_losses, pred_v_actual


class GRU(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout) -> None:
        super().__init__()
        #self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        # Add LSTM models
        self.lstm = nn.GRU(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,512)
        self.dense2 = nn.Linear(512,256)
        self.dense3 = nn.Linear(256,self.output_size)

        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        x= x.unsqueeze(2)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        #c_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        output, h_n = self.lstm(x,h_0)
        output = self.relu(output)
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.dense2(output)
        output = self.dense3(output)

        return output
    
    def train_model(self,epochs,lr,loss_function,data_loader,val_loader):
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loss = []
        val_loss = []
        optimizer = torch.optim.Adam(self.parameters(),lr)
        num_batches = len(data_loader)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X)
                loss = loss_function(pred,y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() / num_batches
            print(f"Train Loss for epoch {i}: {epoch_loss}")
            train_loss.append(epoch_loss)
            
            for X,y in val_loader:
                pred = self(X)
                loss = loss_function(pred,y)
                epoch_loss += loss.item() / num_batches
            print(f"Val Loss for epoch {i}: {epoch_loss}")
            val_loss.append(epoch_loss)

        return train_loss, val_loss
    
    
    def test_model(self,loss_function,data_loader):
        # Just get the loss
        self.eval()
        test_loss = 0.0
        all_losses = []
        pred_v_actual = [[],[]]
        num_batches = len(data_loader)
        with torch.no_grad():
            for X,y in data_loader:
                pred = self(X)
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y)
                batch_loss=loss_function(pred,y).item()/num_batches
                test_loss += batch_loss
                all_losses.append(batch_loss)

        print(f"Avg Test Loss: {test_loss}")
        return test_loss, all_losses, pred_v_actual
    


class LSTM(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout) -> None:
        super().__init__()
        #self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        # Add LSTM models
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,512)
        self.dense2 = nn.Linear(512,256)
        self.dense3 = nn.Linear(256,self.output_size)

        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        x= x.unsqueeze(2)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        #c_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        output, h_n = self.lstm(x,h_0)
        output = self.relu(output)
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.dense2(output)
        output = self.dense3(output)

        return output
    
    def train_model(self,epochs,lr,loss_function,data_loader,val_loader):
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loss = []
        val_loss = []
        optimizer = torch.optim.Adam(self.parameters(),lr)
        num_batches = len(data_loader)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X)
                loss = loss_function(pred,y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() / num_batches
            print(f"Train Loss for epoch {i}: {epoch_loss}")
            train_loss.append(epoch_loss)
            
            for X,y in val_loader:
                pred = self(X)
                loss = loss_function(pred,y)
                epoch_loss += loss.item() / num_batches
            print(f"Val Loss for epoch {i}: {epoch_loss}")
            val_loss.append(epoch_loss)
        return train_loss, val_loss
    
    
    def test_model(self,loss_function,data_loader):
        # Just get the loss
        self.eval()
        test_loss = 0.0
        all_losses = []
        pred_v_actual = [[],[]]
        num_batches = len(data_loader)
        with torch.no_grad():
            for X,y in data_loader:
                pred = self(X)
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y)
                batch_loss=loss_function(pred,y).item()/num_batches
                test_loss += batch_loss
                all_losses.append(batch_loss)

        print(f"Avg Test Loss: {test_loss}")
        return test_loss, all_losses, pred_v_actual

class PMDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, pos):
        return torch.Tensor(self.x[pos]), torch.Tensor(self.y[pos])


def read_data(data_path='./processed_data'):
    data_txt = codecs.open(filename=data_path + "/input_data.json", mode="r", encoding='utf-8').read()
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
    val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)
    return train_loader, val_loader, test_loader
