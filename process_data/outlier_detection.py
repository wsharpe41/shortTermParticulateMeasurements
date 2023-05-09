import matplotlib.pyplot as plt
import numpy as np
# Read in pm data and detect outliers
def detect_outliers(data,mean,std_dev,avg_mean,avg_std_dev,path=""):    
    if mean is None:
        mean = avg_mean
        std_dev = avg_std_dev
    j = []
    for i in range(len(data)):
        if data[i] < mean - 7 * std_dev or data[i] > mean + 7 * std_dev:
            # If the value is twice as big as the previous two values and the next two values, print it
            if i > 7 and i < len(data) - 7:
                if data[i] > 5* np.mean(data[i-7:i]):
                    j.append(i)

    
    # Plot the data with outliers in red 
    if len(j) > 7:
        plt.figure(1)
        plt.plot(data,color="blue",label="Data",alpha=0.5)
        # Plot the outliers as points in red
        plt.scatter(j,[data[i] for i in j],color="red",label="Outliers")
        plt.title("Data with Outliers")
        plt.xlabel("Index")
        plt.ylabel("PM 2.5 (ug/m^3)")
        plt.savefig("process_data/anomalies/data_with_outliers.png")

    data = np.delete(data,j)
    
    return data