from process_data import process_data
from predict_pm import rnn
from torchmetrics import MeanSquaredError
import torch

train_loader, val_loader, test_loader = rnn.read_data('processed_data')
epochs = 100
lr = 0.001
mse = MeanSquaredError()
model = rnn.RecurrentNeuralNetwork(
    hidden_size=64,
    in_size=1,
    out_size=6,
    num_layers=1,
    dropout=0.0
)

model.train_model(epochs,lr,mse,train_loader)
model.test_model(mse,val_loader)
