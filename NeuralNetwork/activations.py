import numpy as np
class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, out_grad):
        return (np.maximum(0, self.x) > 0) * out_grad

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        neu = np.exp(x) - np.exp(-x)
        deo = np.exp(x) + np.exp(-x)
        self.y = neu / deo
        return self.y

    def backward(self, out_grad):
        return (1 - (self.y ** 2)) * out_grad

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, out_grad):
        pass

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-self.x))
        return self.out

    def backward(self, out_grad):
        return self.out * (1 - self.out) * out_grad