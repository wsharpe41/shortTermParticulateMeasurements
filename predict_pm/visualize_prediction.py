import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_average(pred_v_actual,scaler):

    pred = pred_v_actual[0]
    y= pred_v_actual[1]
    # Unscale the data
    unscaled_preds = []
    unscaled_y = []

    # For each value in each batch, unscale for y and pred using enumerate
    for i,batch in enumerate(pred):
        for j,value in enumerate(batch):
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            value = np.average(value)
            unscaled_preds.append(value)

    for i,batch in enumerate(y):
        for j,value in enumerate(batch):
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            value = np.average(value)
            unscaled_y.append(value)


    
    plt.plot(unscaled_preds,color="red",label="Predictions",alpha=0.15)
    plt.plot(unscaled_y,color='blue',label="Actual",alpha=0.15)
    plt.title("Averaged Prediction vs Actual")
    plt.xlabel("Batch #")
    plt.ylabel("PM2.5")
    plt.legend(loc='upper right')
    plt.savefig("average_pred_vs_actual_three_day.png")
    
    # Plot residuals
    residuals = []
    for i in range(len(unscaled_preds)):
        residuals.append(unscaled_preds[i] - unscaled_y[i])
    plt.figure(2)
    plt.scatter(residuals,unscaled_y,color="red",label="Residuals",alpha=0.15)
    plt.title("Residuals v Actual")
    plt.xlabel("Residuals")
    plt.ylabel("Actual")
    plt.savefig("residuals_three_day.png")

# Plot train and validation loss
def plot_loss(train_loss,val_loss):
    plt.figure(3)
    plt.plot(train_loss,color='blue',label="Train Loss")
    plt.plot(val_loss,color='red',label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc='upper right')
    plt.savefig("train_val_loss_three_day.png")