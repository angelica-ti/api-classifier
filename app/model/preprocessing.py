import tensorflow as tf
import numpy as np
import cv2 as cv

class Preprocessing:
    def __init__(self):
        pass
    
    def apply(self, bytes_image):
        image = np.array(np.frombuffer(bytes_image, dtype=np.uint8))
        img = cv.imdecode(image,cv.IMREAD_COLOR)
        img = cv.resize(img, (50,50), interpolation = cv.INTER_AREA)
        return tf.convert_to_tensor([tf.convert_to_tensor(img, dtype=tf.float32)])/255