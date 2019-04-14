# test_model.py

import numpy as np
import time
from alexnet import alexnet


import random

MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09   
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

train_data = np.load('training_data.npy')
shuffle(train_data)
test = train_data[-100:]
test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

def main():
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (160,120))

    prediction = model.predict([screen.reshape(160,120,1)])[0]
    print(prediction)

 
main()       










