import numpy as np
import cv2
import torch
from models.MoodNet import MoodNet

model = MoodNet()
model.load_state_dict(torch.load('moodnet.pth'))

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_image(path):
    img = cv2.imread(path)
    img = np.uint8(np.round(img))
    return img

def convert_to_rgb(img):
    img = np.copy(img)
    return img[..., ::-1]

def convert_to_grayscale(img):
     img = np.copy(img)
     img = np.uint8(np.round(np.dot(img, [0.299, 0.587, 0.114])))
     return img


def predict_mood(image_path):
    img = load_image(image_path)
    img = convert_to_rgb(img)
    img = convert_to_grayscale(img)

    img = cv2.resize(img, (48, 48))
    img = np.uint8(np.round(img))
    img = np.reshape(img, [1, 1, 48, 48])

    img = torch.from_numpy(img)
    img = img.float()

    classes = model(img)
    predict_mood = torch.argmax(classes, dim=1)
    
    return class_labels[predict_mood]


if __name__ == '__main__':
    print(predict_mood('test1.jpg'))
    print(predict_mood('test2.jpeg'))
    print(predict_mood('test3.jpg'))
    print(predict_mood('test4.png'))
