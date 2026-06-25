import numpy as np

class LinearLayers:
    def __init__(self, in_features, out_features):
        self.W = np.random.rand(in_features, out_features)
        self.b = np.random.rand(out_features,)

        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        return self.x @ self.W + self.b # shape = (B, out_features)
    
    def backward(self, out_grad):
        self.dW = self.x.T @ out_grad
        self.db = out_grad.sum(axis = 0)
        self.dx = out_grad @ self.W.T
        return self.dx