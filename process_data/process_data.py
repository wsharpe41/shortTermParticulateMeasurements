import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def split_csv_to_datapoints(path, dp_length,pred_length):
    # Read csv file
    data = pd.read_csv(path, header=None)
    # For row in data, split into X and y
    # X is a datapoint of length dp_length
    # y is the next pred_length values
    X = np.empty((0, dp_length))
    y = np.empty((0, pred_length))
    for i in range(len(data) - dp_length - pred_length):
        X = np.append(X, np.array([data.iloc[i:i + dp_length, 0]['sample_measurement']]), axis=0)
        y = np.append(y, np.array([data.iloc[i + dp_length:i + dp_length + pred_length, 0]['sample_measurement']]), axis=0)
    return X, y

def split_all_files():
    # Path to pmu_data
    path = 'pm_data/'
    dp_length = 50
    pred_length = 5
    # Create an empty np array
    X = np.empty((0, dp_length))
    y = np.empty((0, pred_length))
    # For each file in pmu_data, split into datapoints of length 100
    for file in os.listdir(path):
        datapoints,labels = split_csv_to_datapoints(path + file, dp_length,pred_length)
        # Append all datapoints to data
        X = np.append(X, datapoints, axis=0)
        y = np.append(y, labels, axis=0)
    scaler = StandardScaler()
    X = scaler.transform(X)
    y = scaler.transform(y)
    return X, y
