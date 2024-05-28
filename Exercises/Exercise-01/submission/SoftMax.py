import numpy as np # type: ignore

from Layers.Base import BaseLayer

class SoftMax(BaseLayer):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor):
        scaled_input = input_tensor - np.max(input_tensor, axis = 1, keepdims = True)
        exps = np.exp(scaled_input) 
        self.output_tensor = exps / np.sum(exps, axis = 1, keepdims = True)
        return self.output_tensor
    
    def backward(self, error_tensor):
        return self.output_tensor * (
            error_tensor - (error_tensor * self.output_tensor).sum(axis = 1).reshape(-1, 1))
