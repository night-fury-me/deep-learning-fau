import numpy as np # type: ignore
from Layers.Base import BaseLayer
from Optimization import Optimizers

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size)) # [5 x 3]
        self._optimizer: Optimizers.BaseOptimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        initial_bias = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate((input_tensor, initial_bias), axis = 1) # [9 x 5]
        return np.dot(self.input_tensor, self.weights)  # [9 x 5] (.) [5 x 3] = [9 x 3] 
    
    def backward(self, error_tensor):
        # error_tensor -> [9 x 3]
        weights = self.weights[:self.weights.shape[0]-1, :] # [4 x 3]: bias excluded
        gradient_input = np.dot(error_tensor, weights.T)    # [9 x 4]

        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor) # [5 x 3] 

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, 
                self._gradient_weights
            ) # [5 x 3]

        return gradient_input

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
