import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class Classifier:
    
    def __init__(self):
        self.classes_names = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses','human']
        self.model = self.create_model()
    
    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(50, 50, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(7))
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        return model
    
    def load_model(self, checkpoint_path = 'app/model/training_1/cp.ckpt'):
        self.model.load_weights(checkpoint_path)
        
    def predict(self, image):
        values = self.model.predict(image)
        print(values)
        prediction = self.classes_names[np.argmax(values)]
        return prediction

        