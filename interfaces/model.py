from abc import ABC, abstractmethod

class Model(ABC):
    """Pattern to create machine learning models

    Args:
        ABC (_type_): _description_
    """
    
    def __init__(self, lr, batch_size, epochs) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
    @abstractmethod
    def create_model(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def opmizer(self):
        pass
    
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    