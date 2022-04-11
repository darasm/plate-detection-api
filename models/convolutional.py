from interfaces import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class ConvolutionalNetwork(Model):
    """CNN model

    Args:
        Model (_type_): _description_
    """
    def __init__(self, lr, batch_size, epochs) -> None:
        self.model = Sequential()
        super().__init__(lr, batch_size, epochs)
    
    
    def create_model(self):
        """Building the convolutional model

        Returns:
            _type_: _description_
        """
        self.model.add(Conv2D(name="convolution2d_1", filters=32, kernel_size=(3, 3), activation='relu', input_shape=28))
        self.model.add(Conv2D(name="convolution2d_2", filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(name="maxpooling2d_1", pool_size=[2, 2]))
        self.model.add(Dropout(name="dropout_1", rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(name="dense_1", units=128, activation='relu'))
        self.model.add(Dropout(name="dropout_2", rate=0.5))
        self.model.add(Dense(name="dense_2", units=10, activation='softmax'))
        
        return self.model
    
    
    def opmizer(self, loss_function, opmizer, metrics):
        return self.model.compile(
            loss=loss_function, 
            optimizer=opmizer, 
            metrics=[metrics]
        )
    

    def train(self,input_train, target_train, input_test, target_test ):
        return self.model.fit(
            input_train, 
            target_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            verbose=1, 
            validation_data=(input_test, target_test)
        )
    
    
    def evaluate(self, input_test, target_test):
        return self.model.evaluate(input_test, target_test, verbose=0)
    
    
    def save_model(self) -> None:
        """Saving the weights and serializing with JSON

        Returns:
            _type_: _description_
        """
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
    
    