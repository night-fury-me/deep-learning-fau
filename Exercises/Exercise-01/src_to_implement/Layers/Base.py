from abc import ABC, abstractmethod

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