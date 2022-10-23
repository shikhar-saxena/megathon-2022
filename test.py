import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

my_model = keras.models.load_model('./MoodNet.h5', compile=False)

def predict_mood(image_path):
    img = cv2.imread(image_path, cmpa=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, [1, 1, 48, 48])

    predictions = my_model.predict(img)

    predictions = np.argmax(predictions, axis=1)
    return predictions


if __name__ == '__main__':
    print(predict_mood('./test1.jpg'))
    print(predict_mood('./test2.jpeg'))
