
# DEPRECATED: This file is not used in the final version of the project
# This was used to optimize the hyperparameters of the RNN model
 
train_dataset, val_dataset, test_dataset = read_dataset()

def train_rnn(config, checkpoint_dir=None):
    # Instantiate the RNN model with the hyperparameters from config
    model = GRU(config["hidden_size"], 1, 6, 
                config["num_layers"], config["dropout"], config["l1"], config["l2"],config["l3"],config["l4"],config["l5"],config["l6"],config["l7"],config["l8"])

    # Define the loss function and optimizer
    loss_function = MeanSquaredError()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=False)
    
    _, val_loss = model.train_model(config["epochs"], config["lr"], optimizer,loss_function, train_loader, val_loader)

    # Return the validation loss
    return {"mean_val_loss": np.mean(val_loss)}

search_space = {
    "hidden_size": tune.choice([10, 20, 64, 128]),
    'num_layers': 1,
    'dropout': tune.choice([0.1, 0.2, 0.3, 0.4]),
    'l1': tune.choice([64, 128, 256,512]),
    'l2': tune.choice([64, 128, 256,512]),
    'l3': tune.choice([64, 128, 256,512]),
    'l4': tune.choice([32, 64, 128,256]),
    'l5': tune.choice([32, 64, 128,256]),
    'l6': tune.choice([32, 64, 128,256]),
    'l7': tune.choice([16, 32, 64,128]),
    'l8': tune.choice([16, 32, 64,128]),
    'batch_size': tune.choice([32, 64,128]),
    'lr': tune.loguniform(1e-4, 1e-2),
    'epochs': 25
}

#hyperopt_search = ray.tune.suggest.HyperOptSearch(metric='mean_val_loss', mode='min')
scheduler = ASHAScheduler(time_attr='training_iteration', metric='mean_val_loss', mode='min', max_t=20, grace_period=1)
reporter = CLIReporter(metric_columns=['loss', 'mean_val_loss', 'training_iteration'])
tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_rnn),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=20,
        ),
        param_space=search_space,
    )

results = tuner.fit()
best_result = results.get_best_result(metric='mean_val_loss', mode='min')
print("Best trial config: {}".format(best_result.config))
print("Best trial final validation loss: {}".format(best_result.metrics["mean_val_loss"]))
