from process_data import process_data
from predict_pm import rnn
from torchmetrics import MeanSquaredError
import torch

train_loader, val_loader, test_loader = rnn.read_data('processed_data')
epochs = 50
lr = 0.001
mse = MeanSquaredError()
