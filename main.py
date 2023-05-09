from process_data import process_data
from predict_pm import rnn
from predict_pm import visualize_prediction
from predict_pm import performance
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_scaler,y_scaler = process_data.split_all_files("pm_data/2015_pm_data/","processed_data",dp_length=24*7,pred_length=24*3,measurement_index=7)
train_loader, val_loader, test_loader = rnn.read_data('processed_data')
epochs = 120
lr = 0.00001
mse = MeanSquaredError().to(device)
print("Training GRU Model...")
model = rnn.GRU(
      hidden_size=128,
      in_size=6,
      out_size=pred_length,
      num_layers=1,
      dropout=0.2
  )
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss, val_loss = model.train_model(epochs,lr,optimizer,mse,train_loader,val_loader)
test_loss, all_losses, pred_v_actual, avg_loss = model.test_model(mse,test_loader)
visualize_prediction.plot_average(pred_v_actual,y_scaler)
visualize_prediction.plot_loss(train_loss,val_loss)
performance.get_r2(pred_v_actual)
performance.get_rmse(pred_v_actual,y_scaler)
