# Short Term Particulate Measurements
The EPA's IMPROVE and CSN networks are the USAs primary provider of accuract PM data. One drawback to these networks is that they require postprocessing of their data, which can take a few days time. Thus data is always a few days behind. This project uses RNN architectures to try to close that gap and predict real-time PM2.5 hourly concentrations based off of previously reported values.

# Data
Data was from a kaggle dataset showing hourly concentrations from 2014-2015 for all US monitoring sites (~8 million data points). High level statistics from 2014 were used and data from 2015 was used for training and testing. Outliers were detected and taken out based on validation accuracy increases.

# Models
All models were created with PyTorch and hyperparameters were tuned with Ray Tune. GRUs, LSTMs, and Vanilla RNNs were tested. Different prediction and data point lengths were also tested, in the end the best predictions came from using the past seven days to predict the next 3 days. The best performing model was a GRU with one recurrent layer and three dense layers trained for 100 epochs.

## Model Performance
The model was able to capture most of the variance in PM2.5, but had a hard time with extreme values. This makes sense since many of these events would be expected to be either a chance event (such as a truck driving by) or an annual event such as a forest fire. The past seven days of measurements would be unlikely to predict these scenarios.
