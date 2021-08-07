import tensorflow as tf
import cv2
import numpy as np
from createData import screenshot, preprocess
from time import sleep
import keyboard

save_path="savedmodel"
class_names = ['left', 'none', 'right']

model = tf.keras.models.load_model(save_path)

# used to predict the key to press from an image
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

# main loop
# takes screenshots and predicts the needed keypress
def play():
    print("press 's' to start")
    keyboard.wait("s")
    sleep(1)
    print("started")
    current = "none"
    while True:
        if keyboard.is_pressed("s"):
            if current != 'none':
                    keyboard.release(current)
            break
        img = screenshot("car (DEBUG)")
        img = preprocess(img)
        prediction, acc = predict(img)
        print(prediction, acc)
        if prediction != 'none':
            if prediction != current:
                if current != 'none':
                    keyboard.release(current)
                keyboard.press(prediction)
        else:
            if current != 'none':
                keyboard.release(current)
        current = prediction
    print("stopped")

if __name__ == '__main__':
    play()