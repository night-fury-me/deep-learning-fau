from typing import Any
import numpy as np # type: ignore

class CrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        y, y_hat = label_tensor, prediction_tensor
        loss = -np.sum(y * np.log(y_hat + np.finfo(float).eps))
        return loss
    
    def backward(self, label_tensor):
        return -(label_tensor / (self.prediction_tensor + np.finfo(float).eps))