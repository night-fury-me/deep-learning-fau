from abc import ABC, abstractmethod
from .Initializers import BaseInitializer

class BaseLayer(ABC):
    def __init__(self) -> None:
        self.trainable = False
        self.wights = None

    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def backward(self, error_tensor):
        pass

class InitializableLayer(BaseLayer):
    @abstractmethod
    def initialize(self, weights_initializer: BaseInitializer, bias_initializer: BaseInitializer):
        pass
