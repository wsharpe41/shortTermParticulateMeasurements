from ray import tune
from matplotlib import pyplot as plt
import pandas as pd

path = "C:/Users/Will/ray_results/train_rnn_2023-04-16_17-58-28"
restored_tuner = tune.Tuner.restore(path)
result_grid = restored_tuner.get_results()

# For each result in result_grid, plot the mean_val_loss vs. training_iteration
plt.figure(1)

# show all rows in a dataframe
pd.set_option('display.max_rows', None)

for result in result_grid:
    print(result.metrics_dataframe['training_iteration'])


ax = None
for result in result_grid:
    if ax is None:
        ax = result.metrics_dataframe.plot("training_iteration", "mean_val_loss")
    else:
        result.metrics_dataframe.plot("training_iteration", "mean_val_loss", ax=ax)
ax.set_title("Mean Val Loss vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Loss")
