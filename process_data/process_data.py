import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import codecs

def split_csv_to_datapoints(path, dp_length,pred_length):
    # Read csv file
    data = pd.read_csv(path, header=None,skiprows=1)
    # For row in data, split into X and y
    # X is a datapoint of length dp_length
    # y is the next pred_length values
    X = np.empty((0, dp_length))
    y = np.empty((0, pred_length))
    correlations = []
    # Get sample_measurments column of data
    pm_measurments = data[7].to_numpy()
    # Iterate through pm_measurments and split into datapoints with length dp_length and labels with length pred_length with no overlap
    for i in range(0, len(pm_measurments) - dp_length - pred_length, dp_length):
        # Get datapoint
        datapoint = pm_measurments[i:i + dp_length]
        # Get label
        label = pm_measurments[i + dp_length:i + dp_length + pred_length]
          
        # Append datapoint and label to X and y
        X = np.append(X, [datapoint], axis=0)
        y = np.append(y, [label], axis=0)
        
        # Replace all negative values with 0 in x and y
        X[X < 0] = 0
        y[y < 0] = 0
    return X, y

def split_all_files(input_path,output_path,dp_length,pred_length):
    # Path to pmu_data
    # Create an empty np array
    X = np.empty((0, dp_length))
    y = np.empty((0, pred_length))
    # For each file in pmu_data, split into datapoints of length 100
    for file in os.listdir(input_path):
        print("Processing file: " + file)
        datapoints,labels = split_csv_to_datapoints(input_path + file, dp_length,pred_length)
        # get the averages of correlations
        # Append all datapoints to data
        X = np.append(X, datapoints, axis=0)
        y = np.append(y, labels, axis=0)    
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    write_input_and_output(X,y,output_path)  
    return X_scaler,y_scaler


def write_input_and_output(X,y,output_path):
    X = X.tolist()
    y = y.tolist()
    x_path = output_path + "/input_data.json"
    y_path = output_path + "/target_data.json"
    json.dump(X, codecs.open(x_path, 'w', encoding='utf-8'), indent=4, separators=(",",":"))
    json.dump(y,codecs.open(y_path, 'w', encoding='utf-8'),indent=4, separators=(",",":"))
    return


if __name__ == '__main__':
    X_scaler, y_scaler = split_all_files("pm_data/","processed_data",dp_length=48,pred_length=6)

