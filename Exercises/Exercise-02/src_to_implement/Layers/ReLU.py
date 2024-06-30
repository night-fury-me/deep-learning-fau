import numpy as np # type: ignore
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return self.input_tensor * (input_tensor > 0)
    
    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)