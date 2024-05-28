import numpy as np # type: ignore
import copy
from Optimization.Loss import BaseLoss
from Optimization.Optimizers import BaseOptimizer

class NeuralNetwork:
    def __init__(self, optimizer) -> None:
        self.data_layer = None
        self.optimizer  : BaseOptimizer = optimizer
        self.loss_layer : BaseLoss      = None
        self.layers = list()
        self.loss   = list()

    def forward(self):
        x, self.y = copy.deepcopy(self.data_layer.next())
        
        for layer in self.layers:
            x = layer.forward(x)
        
        return self.loss_layer.forward(
            predicted_label = x, 
            true_label      = copy.deepcopy(self.y)
        )
    
    def backward(self):
        y = self.loss_layer.backward(
            true_label = copy.deepcopy(self.y)
        )
        for layer in reversed(self.layers):
            y = layer.backward(y)
        
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for epoch in range(iterations):
            current_loss = self.forward()
            self.loss.append(current_loss)
            self.backward()
    
    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor