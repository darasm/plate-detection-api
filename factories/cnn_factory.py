from usecases import PlateDetection
from models import ConvolutionalNetwork

class CnnFactory:
    
    @staticmethod
    def create() -> PlateDetection:
        return PlateDetection(ConvolutionalNetwork(lr=0.001, batch_size=10, epochs=1))