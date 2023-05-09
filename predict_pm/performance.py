from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error

def get_r2(pred_v_actual):
    # Get pred_v_actual to cpu
    pred = pred_v_actual[0]
    unscaled_preds = []
    unscaled_y = []

    actual= pred_v_actual[1]
    # Print shape of pred list
    for i,batch in enumerate(pred):
        for j,value in enumerate(batch):
            value = value.cpu()
            value = value.numpy()
            unscaled_preds.append(value)

    for i,batch in enumerate(actual):
        for j,value in enumerate(batch):
            value = value.cpu()
            value = value.numpy()
            unscaled_y.append(value)
    
    r2 = r2_score(unscaled_preds,unscaled_y)
    print(f"R2 Score: {r2}")
    return r2

def get_rmse(pred_v_actual,scaler):
    pred = pred_v_actual[0]
    actual= pred_v_actual[1]
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

    for i,batch in enumerate(actual):
        for j,value in enumerate(batch):
            value = value.cpu()
            value = value.numpy()
            value = scaler.inverse_transform(value.reshape(1,-1))
            # Average value
            value = np.average(value)
            # Unpack value to get rid of array
            unscaled_y.append(value)
            
    rmse = np.sqrt(mean_squared_error(unscaled_preds,unscaled_y))
    print(f"RMSE: {rmse}")
    return rmse