from process_data import process_data
from predict_pm import rnn
from predict_pm import visualize_prediction
from torchmetrics import MeanSquaredError
import torch


# Need to allow access to unscaled data
X_scaler,y_scaler = process_data.split_all_files("pm_data/","processed_data",dp_length=72,pred_length=6)
train_loader, val_loader, test_loader = rnn.read_data('processed_data')
epochs =  15
lr = 0.0001
mse = MeanSquaredError()


model = rnn.GRU(
    hidden_size=20,
    in_size=1,
    out_size=6,
    num_layers=1,
    dropout=0.0
)

train_loss, val_loss = model.train_model(epochs,lr,mse,train_loader,val_loader)
test_loss, all_losses, pred_v_actual = model.test_model(mse,test_loader)
visualize_prediction.plot_average(pred_v_actual,y_scaler)
visualize_prediction.plot_loss(train_loss,val_loss)