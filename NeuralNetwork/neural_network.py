from linearlayers import LinearLayers

class NeuralNetwork:
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, n_layers: int):

        self.layers = [LinearLayers(in_features = in_features, out_features = hidden_dim)] # 1st layer

        for _ in range(n_layers - 2): # n-2 layers
            self.layers.append(
                LinearLayers(in_features = hidden_dim, out_features = hidden_dim)
            )

        # last layer
        self.layers.append(
            LinearLayers(in_features = hidden_dim, out_features = out_features)
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, out_grad):
        for layer in self.layers[::-1]:
            out_grad = layer.backward(out_grad)
        return out_grad
    
    def __call__(self, x):
        return self.forward(x)
    
    def param(self):
        self.param = []
        for layer in self.layers:
            self.param.append(layer.param.item())


if __name__ == "__main__":
    import numpy as np
    nn = NeuralNetwork(in_features = 3, hidden_dim = 10, out_features = 2, n_layers = 5)
    y_true = [0, 1]
    x = np.random.rand(1, 3)
    logits = nn(x) # forward pass execute ho jayega
    dL_dy = (-2 / len(y_true)) * (y_true - logits)
    grad = nn.backward(dL_dy) # forward pass execute ho jayega
    print(f"Gradients of weights Layer-1: {nn.layers[0].dW}")
    print(f"Gradients of bias for Layer-1: {nn.layers[0].db}")
    # nn.param()
    # print(nn.param)
    # epochs = 1000
    # lr = 0.0001
    # for i in range(epochs):
    #     logits = nn(x) # forward pass execute ho jayega
    #     mse_loss = np.mean((y_true - logits) ** 2)
    #     dL_dy = (-2 / len(y_true)) * (y_true - logits)

    #     nn.param
        
    #     if i % 100 == 0:
    #         print(f"Epoch = [{i}/{epochs}] | Loss = {mse_loss:.4f}")

