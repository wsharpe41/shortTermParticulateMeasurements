import json
import codecs
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Print which device is being used
print(f"Using {device} device")

class EarlyStopping:
    def __init__(self,patience,minimum_decrease) -> None:
        self.patience = patience
        self.minimum_decrease = minimum_decrease
        self.best_val_loss = float('inf')
        self.counter = 0
        
    # Stop early if the validation loss has not decreased by the minimum decrease for the patience number of epochs
    def stop_early(self,val_loss):
        if val_loss < self.best_val_loss - self.minimum_decrease:
            self.best_val_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_val_loss - self.minimum_decrease:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class RNN(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout=0.3,l1=512,l2=256,l3=128) -> None:
        super().__init__()
        #self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        # Add LSTM models
        self.rnn = nn.RNN(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,l1)
        self.dense2 = nn.Linear(l1,l2)
        self.dense3 = nn.Linear(l2,l3)
        self.dense4 = nn.Linear(l3,self.output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        #x= x.unsqueeze(2)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        #c_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_()
        output, h_n = self.rnn(x.to(device),h_0.to(device))

        output = self.relu(output)
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = self.dense3(output)
        output = self.relu(output)
        output = self.dense4(output)
        return output.to(device)
    
    def train_model(self,epochs,lr,optimizer,loss_function,data_loader,val_loader):
        print("Training")
        train_loss = []
        val_loss = []
        num_batches = len(data_loader)
        early_stopper = EarlyStopping(patience=20,minimum_decrease=0.001)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X.to(device))
                loss = loss_function(pred,y.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Train Loss for epoch {i}: {epoch_loss/num_batches}")
            train_loss.append(epoch_loss/num_batches)
            
            for X,y in val_loader:
                pred = self(X.to(device))
                loss = loss_function(pred,y.to(device))
                epoch_loss += loss.item()
            print(f"Val Loss for epoch {i}: {epoch_loss/num_batches}")
            val_loss.append(epoch_loss/num_batches)
            if early_stopper.stop_early(epoch_loss/num_batches):
                break
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
                pred = self(X.to(device))
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y.to(device))
                batch_loss=loss_function(pred,y).item()
                test_loss += batch_loss
            all_losses.append(batch_loss/num_batches)

        print(f"Avg Test Loss: {test_loss/num_batches}")
        return test_loss, all_losses, pred_v_actual


class GRU(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout,l1=512,l2=256,l3=128) -> None:
        super().__init__()
        #self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        self.attention = nn.MultiheadAttention(self.hidden_size,1,dropout=dropout).to(device)
        # Add LSTM models
        self.gru = nn.GRU(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True).to(device)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,l1).to(device)
        self.dense2 = nn.Linear(l1,l2).to(device)
        self.dense3 = nn.Linear(l2,l3).to(device)
        self.dense4 = nn.Linear(l3,self.output_size).to(device)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.r2score = R2Score().to(device)

    def forward(self,x):
        # Create architecture here
        x = x.to(device)
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).requires_grad_().to(device)
        output, h_n = self.gru(x.to(device),h_0.to(device))
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense4(output)
        return output.to(device)
    
    def train_model(self,epochs,lr,optimizer,loss_function,data_loader,val_loader):
        train_loss = []
        val_losses = []
        num_batches = len(data_loader)
        early_stopper = EarlyStopping(patience=20,minimum_decrease=0.001)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X.to(device))
                loss = loss_function(pred,y.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Train Loss for epoch {i}: {epoch_loss/num_batches}")
            train_loss.append(epoch_loss/num_batches)
            with torch.no_grad():
                epoch_val_loss = 0.
                for X,y in val_loader:
                    pred = self(X.to(device))
                    val_loss = loss_function(pred,y.to(device))
                    epoch_val_loss += val_loss.item()   
                print(f"Val Loss for epoch {i}: {epoch_val_loss/len(val_loader)}")
                val_losses.append(epoch_val_loss/len(val_loader))
                if early_stopper.stop_early(epoch_val_loss/len(val_loader)):
                    break
        return train_loss, val_losses
    
    def test_model(self,loss_function,data_loader):
        # Just get the loss
        loss_function = loss_function.to(device)
        self.eval()
        test_loss = 0.0
        all_losses = []
        pred_v_actual = [[],[]]
        num_batches = len(data_loader)
        with torch.no_grad():
            for X,y in data_loader:
                pred = self(X.to(device))
                pred = pred.to(device)
                y = y.to(device)
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y.to(device))
                batch_loss=loss_function(pred,y).item()
                test_loss += batch_loss
            all_losses.append(batch_loss/num_batches)

        print(f"Avg Test Loss: {test_loss/num_batches}")
        return test_loss, all_losses, pred_v_actual,test_loss/num_batches


class LSTM(nn.Module):
    
    def __init__(self,hidden_size,in_size,out_size,num_layers,dropout=0.3,l1=512,l2=256,l3=128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = in_size
        self.output_size = out_size
        self.num_layers = num_layers
        # Add LSTM models
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,num_layers=self.num_layers,dropout=dropout,batch_first=True,bidirectional=False).to(device)
        # Add dense layer
        self.dense = nn.Linear(self.hidden_size,l1).to(device)
        self.dense2 = nn.Linear(l1,l2).to(device)
        self.dense3 = nn.Linear(l2,l3).to(device)
        #self.dense4 = nn.Linear(l3,l4)
        self.dense5 = nn.Linear(l3,self.output_size).to(device)
        self.relu = nn.ReLU()

    def forward(self,x):
        # Create architecture here
        x = x.to(device)
        batch_size = x.shape[0]
        h_0 = (torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device),
        torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device))
        output, h_n = self.lstm(x,h_0)
        output = self.relu(output)
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)
        output = self.dense3(output)
        output = self.relu(output)
        output = self.dense5(output)
        return output
    
    def train_model(self,epochs,lr,optimizer,loss_function,data_loader,val_loader):
        train_loss = []
        val_loss = []
        num_batches = len(data_loader)
        early_stopper = EarlyStopping(patience=20,minimum_decrease=0.001)
        for i in range(epochs):
            epoch_loss = 0.0
            for X,y in data_loader:
                optimizer.zero_grad()
                pred = self(X.to(device))
                loss = loss_function(pred,y.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Train Loss for epoch {i}: {epoch_loss/num_batches}")
            train_loss.append(epoch_loss/num_batches)
            
            for X,y in val_loader:
                pred = self(X.to(device))
                loss = loss_function(pred,y.to(device))
                epoch_loss += loss.item()
            print(f"Val Loss for epoch {i}: {epoch_loss/num_batches}")
            val_loss.append(epoch_loss/num_batches)
            if early_stopper.stop_early(epoch_loss/num_batches):
                break
        return train_loss, val_loss
    
    def test_model(self,loss_function,data_loader):
        # Just get the loss
        loss_function.to(device)
        self.eval()
        test_loss = 0.0
        all_losses = []
        pred_v_actual = [[],[]]
        num_batches = len(data_loader)
        with torch.no_grad():
            for X,y in data_loader:
                pred = self(X.to(device))
                pred_v_actual[0].append(pred)
                pred_v_actual[1].append(y)
                batch_loss=loss_function(pred,y.to(device)).item()
                test_loss += batch_loss
                all_losses.append(batch_loss)
        print(f"Avg Test Loss: {test_loss/num_batches}")
        return test_loss, all_losses, pred_v_actual, test_loss/num_batches   


class PMDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, pos):
        return torch.Tensor(self.x[pos]), torch.Tensor(self.y[pos])


def read_data(data_path='../processed_data'):
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
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
    return train_loader, val_loader, test_loader


def read_dataset(data_path='../processed_data'):
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
    return train_dataset, val_dataset, test_dataset