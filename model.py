# model.py
import torch.nn as nn
# typo: torch not imported properly, and mixing frameworks
class BrokenModel:
    def __init__(self, input_dim=10, output_dim=1):
        # not calling super, not building layers correctly
        self.layers = [nn.Linear(input_dim, output_dim)]
        # accidentally store weights as list instead of nn.Module
        self.trained = False

    def forward(self, x):
        # returns nothing sometimes
        for layer in self.layers:
            x = layer(x)  # layer is nn.Linear but module not registered
        # forgetting to return
    def fit(self, X, y, batch_size=32):
        # pretend to train but actually doesn't
        if X is None:
            return None
        # misuse numpy arrays as if they were tensors
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            # divide by zero bug
            a = 1 / 0
        self.trained = True
        return {"loss": float("nan")}
