import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.express as px

def plot_average(pred_v_actual,scaler):

    pred = pred_v_actual[0]
    y= pred_v_actual[1]
    # Unscale the data
    unscaled_preds = []
    unscaled_y = []

    # For each value in each batch, unscale for y and pred using enumerate
    for i,batch in enumerate(pred):
        for j,value in enumerate(batch):
            # Get to cpu first
            value = value.cpu()
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            value = np.average(value)
            unscaled_preds.append(value)

    for i,batch in enumerate(y):
        for j,value in enumerate(batch):
            value = value.cpu()
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            value = np.average(value)
            # Unpack value to get rid of array
            unscaled_y.append(value)

    plt.figure(2)
    plt.plot(unscaled_preds,color="red",label="Predictions",alpha=0.15)
    plt.plot(unscaled_y,color='blue',label="Actual",alpha=0.15)
    plt.title("Test Prediction vs Actual")
    plt.xlabel("Batch #")
    plt.ylabel("PM2.5 (ug/m^3)")
    plt.legend(loc='upper right')
    plt.savefig("average_pred_vs_actual_gru_test2.png")
    
    # Plot residuals
    residuals = []
    for i in range(len(unscaled_preds)):
        residuals.append(unscaled_preds[i] - unscaled_y[i])
    plt.figure(3)
    plt.scatter(residuals,unscaled_y,color="red",label="Residuals",alpha=0.15)
    plt.title("Model Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Actual PM 2.5 (ug/m^3)")
    plt.savefig("residuals_gru_test2.png")

# Plot train and validation loss
def plot_loss(train_loss,val_loss):
    plt.figure(4)
    plt.plot(train_loss,color='blue',label="Train Loss")
    plt.plot(val_loss,color='red',label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc='upper right')
    plt.savefig("train_val_loss_gru_test2.png")
    
    
def plot_performance_by_length(pred_v_actual,scaler):
    pred = pred_v_actual[0]
    y= pred_v_actual[1]
    # make a list of empty lists with length equal to the length of the first list in pred
    all_preds = [[] for i in range(len(pred[0][0]))]
    all_y = [[] for i in range(len(y[0][0]))]
    
    for i,batch in enumerate(pred):
        for j,value in enumerate(batch):
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # append to the list of lists
            for v in range(value.shape[1]):
                all_preds[v].append(value[0][v])

    for i,batch in enumerate(y):
        for j,value in enumerate(batch):
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            for v in range(value.shape[1]):
                all_y[v].append(value[0][v])

    # For each list in all_preds and all_y get the mean squared error
    mse_preds = []
    mse = torch.nn.MSELoss()
    for i in range(len(all_preds)):
        mse_preds.append(mse(torch.tensor(all_preds[i]),torch.tensor(all_y[i]))/len(pred[0]))
    # Change values in mse_preds to numpy
    mse_preds = [x.numpy() for x in mse_preds]
    print(f"MSE for each length: {mse_preds}")
    plt.figure(5)
    plt.plot(mse_preds,color='blue',label="MSE")
    plt.title("MSE vs Predicted Data Index")
    plt.xlabel("Predicted Point Index")
    plt.ylabel("MSE")
    plt.savefig("mse_by_index_6.png")
    return mse_preds

def plot_data_splitting(gru_loss):
    fig = px.scatter_3d(
        x=[inner_list[0] for inner_list in gru_loss],
        y=[inner_list[1] for inner_list in gru_loss],
        z=[inner_list[2] for inner_list in gru_loss],
        color=["GRU"] * len(gru_loss),
    )
    
    fig.update_layout(scene=dict(
        xaxis_title='Data Point Length',
        yaxis_title='Prediction Length',
        zaxis_title='MSE'),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        title="Loss by Data Point and Prediction Length for GRU and LSTM Models"
    )
    
    fig.write_html("gru_loss_test.html")
    print(f"GRU loss: {gru_loss}")
