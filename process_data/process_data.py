import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import codecs
from process_data import outlier_detection

def split_csv_to_datapoints(path, dp_length,pred_length,average_stats,pos_index=[6,7],old_data_path="pm_data/2014_pm_data/"):
    # Read csv file
    data = pd.read_csv(path, header=None,skiprows=1)
    # Delete nan values from data
    data = data.dropna()
    # Print the fifth column of data
    # For row in data, split into X and y
    # X is a datapoint of length dp_length
    # y is the next pred_length values
    X = np.empty((0, dp_length,6))
    y = np.empty((0, pred_length))
    # Get sample_measurments column of data
    pm_measurments = data[8].to_numpy()
    # print the number of values in pm_measurments that are less than mdl
    mdl = data[9][0]
    # Replace all values in pm_measurments that are less than mdl with mdl
    #pm_measurments = pm_measurments[pm_measurments>0]
    
    
    long_lat = data.iloc[:, list(pos_index)].to_numpy()
    # Add method_code, poc, and mdl to long_lat
    #long_lat = np.column_stack((long_lat,data[5].to_numpy()))
    
    
    mean, max_count, std = get_annual_site_stats(old_data_path, path)    
    # Replace negative values with the mean 

    # Print percent of pm_measurements that are less than mdl
    
    pm_measurments = outlier_detection.detect_outliers(pm_measurments,mean,std,average_stats[0],average_stats[2],path)
    # Iterate through pm_measurments and split into datapoints with length dp_length and labels with length pred_length with no overlap
    for i in range(0, len(pm_measurments) - dp_length - pred_length, dp_length):
        # Get datapoint
        datapoint = pm_measurments[i:i + dp_length]
        # Replace all negative values with 0 in datapoint
        datapoint[datapoint < 0] = 0
        # Get the long and lat of the datapoint
        pos = long_lat[i:i + dp_length]
        # Stack the long and lat onto the datapoint
        datapoint = np.column_stack((datapoint,pos))
        
        # Stack the mean, max_count, and std onto the datapoint
        if mean is None:
            mean = np.ones((dp_length,1)) * average_stats[0]
            max_count = np.ones((dp_length,1)) * average_stats[1]
            std = np.ones((dp_length,1)) * average_stats[2]
        else:
            mean = np.ones((dp_length,1)) * mean
            max_count = np.ones((dp_length,1)) * max_count
            std = np.ones((dp_length,1)) * std
        # Stack the mean, max_count, and std onto the datapoint
        datapoint = np.column_stack((datapoint,mean))
        datapoint = np.column_stack((datapoint,max_count))
        datapoint = np.column_stack((datapoint,std))
        label = pm_measurments[i + dp_length:i + dp_length + pred_length]
        # Append datapoint and label to X and y
        X = np.append(X, [datapoint], axis=0)
        y = np.append(y, [label], axis=0)
        y[y < 0] = 0
    return X, y

def split_all_files(input_path,output_path,dp_length,pred_length,measurement_index=8,old_data_path="pm_data/2014_pm_data/"):
    print("Processing Data")
    # Path to pmu_data
    # Create an empty np array
    X = np.empty((0, dp_length,6))
    y = np.empty((0, pred_length))
    # For each file in pmu_data, split into datapoints of length 100
    average_stats = get_average_for_all_sites(old_data_path)
    
    
    for file in os.listdir(input_path):
        # if file is not a csv file, skip
        if file[-4:] != ".csv":
            continue
        datapoints,labels = split_csv_to_datapoints(input_path + file, dp_length,pred_length,average_stats)
        # Append all datapoints to data
        X = np.append(X, datapoints, axis=0)
        y = np.append(y, labels, axis=0)    
    X_scaler = StandardScaler()
    # Select the first value of the third dimension and reshape the array to have 2 dimensions
    temp = X[:, :, 0].reshape(-1, 1)

    # Fit a standard scaler to X and transform only the selected dimension
    temp_scaled = X_scaler.fit_transform(temp)

    # Reshape the scaled dimension back to its original shape
    temp_scaled = temp_scaled.reshape(X.shape[0], X.shape[1], 1)
    
    # Concatenate the scaled dimension with the other dimensions of the original array
    X_scaled = np.concatenate((temp_scaled, X[:, :, 1:]), axis=2)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    write_input_and_output(X_scaled,y,output_path)  
    return X_scaler,y_scaler


def write_input_and_output(X,y,output_path):
    X = X.tolist()
    y = y.tolist()
    x_path = output_path + "/input_data.json"
    y_path = output_path + "/target_data.json"
    json.dump(X, codecs.open(x_path, 'w', encoding='utf-8'), indent=4, separators=(",",":"))
    json.dump(y,codecs.open(y_path, 'w', encoding='utf-8'),indent=4, separators=(",",":"))
    return

def get_annual_site_stats(path,file):
    # If there is a file with name file in path, read it as dataframe
    all_files = os.listdir(path)
    # Get the name of file after the last /
    file = file.split("/")[-1]
    
    if file in all_files:
        data = pd.read_csv(path + file, header=None,skiprows=1)
        pm_measurments = data[8].to_numpy()
        # Get the mean, max, and std of the pm_measurments
        mean = np.mean(pm_measurments)
        max = np.max(pm_measurments)
        std = np.std(pm_measurments)
        return mean,max,std
    else:
        return None,None,None

def get_average_for_all_sites(path):
    means = []
    maxes = []
    stds = []
    all_files = os.listdir(path)
    for file in all_files:
        data = pd.read_csv(path + file, header=None,skiprows=1)
        pm_measurments = data[8].to_numpy()
        # Get the mean, max, and std of the pm_measurments
        means.append(np.mean(pm_measurments))
        maxes.append(np.max(pm_measurments))
        stds.append(np.std(pm_measurments))
    return (np.mean(means),np.mean(maxes),np.mean(stds))


if __name__ == '__main__':
    X_scaler, y_scaler = split_all_files("pm_data/","processed_data",dp_length=48,pred_length=6)

