import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        # Get correlation between datapoint and label
        corr = np.corrcoef(datapoint, label)[0, 1]
        # Append correlation to correlations
        correlations.append(corr)


    return X, y, np.mean(correlations)

def split_all_files():
    # Path to pmu_data
    path = 'pm_data/'
    dp_length = 24
    pred_length = 6
    # Create an empty np array
    X = np.empty((0, dp_length))
    y = np.empty((0, pred_length))
    correlations = []
    # For each file in pmu_data, split into datapoints of length 100
    for file in os.listdir(path):
        print(file)
        datapoints,labels,corr = split_csv_to_datapoints(path + file, dp_length,pred_length)
        # get the averages of correlations
        print(corr)
        correlations.append(corr)
        # Append all datapoints to data
        X = np.append(X, datapoints, axis=0)
        y = np.append(y, labels, axis=0)
    
    print(f"Average Correlation: {np.mean(correlations)}")
    print(correlations)
    plt.plot(correlations)
    # label plot
    plt.xlabel('Datapoint')
    plt.ylabel('Correlation')
    plt.title('Correlation between datapoint and next 6 hours')
    plt.show()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    # plot average of each item in correlations
    # For each item in correlations, get the mean    


   
    return X, y


if __name__ == '__main__':
    split_all_files()
