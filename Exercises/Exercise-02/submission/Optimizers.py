import numpy as np #type: ignore
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
    

class SgdWithMomentum(BaseOptimizer):
    def __init__(self, learning_rate, momentum_rate) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v
    

class Adam(BaseOptimizer):
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 0
        self.v = None
        self.r = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        if self.r is None:
            self.r = np.zeros_like(weight_tensor)
        
        self.k = self.k + 1

        self.v = self.mu * self.v + (1.0 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1.0 - self.rho) * (gradient_tensor ** 2)
        
        v_hat = self.v / (1.0 - self.mu ** self.k)
        r_hat = self.r / (1.0 - self.rho ** self.k)

        return weight_tensor - self.learning_rate * (v_hat / (r_hat ** 0.5 + np.finfo(float).eps))