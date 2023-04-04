import matplotlib.pyplot as plt
import numpy as np

def plot_average(pred_v_actual,scaler):
    pred = pred_v_actual[0]
    y= pred_v_actual[1]
    # Unscale the data
    
    # Remove any nonuniformity in the data


    y = scaler.inverse_transform(y.to_numpy().reshape(-1,1))
    pred = scaler.inverse_transform(pred.to_numpy().reshape(-1,1))
    # For each prediction, average the values
    

    # For each prediction, average each value
    for i in range(len(pred)):
        pred[i] = np.average(pred[i])
        y[i] = np.average(y[i])
    
    plt.plot(pred,color="red",label="Predictions")
    plt.plot(y,color='blue',label="Actual")
    plt.title("Averaged Prediction vs Actual")
    plt.xlabel("Pred #")
    plt.ylabel("Scaled PM2.5")
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("average_pred_vs_actual.png")
    