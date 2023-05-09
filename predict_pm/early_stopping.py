class EarlyStopping:
    def __init__(self,patience,minimum_decrease) -> None:
        self.patience = patience
        self.minimum_decrease = minimum_decrease
        self.best_val_loss = float('inf')
        self.counter = 0
        
    # Stop early if the validation loss has not decreased by the minimum decrease for the patience number of epochs
    def stop_early(self,val_loss):
        if val_loss < self.best_val_loss - self.minimum_decrease:
            self.best_val_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_val_loss - self.minimum_decrease:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False