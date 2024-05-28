from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def calculate_update(self, weight_tensor, gradient_tensor): 
        pass

class Sgd(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor