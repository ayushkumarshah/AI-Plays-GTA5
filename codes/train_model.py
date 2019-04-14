# train_model.py

import numpy as np
from cnn_keras import cnn_keras

#CNN Model
model = cnn_keras()

model.compile(
    optimizer="Momentum",
    loss="categorical_crossentropy",
    metrics=['accuracy'])


train_data = np.load('training_data.npy')

train = train_data[:-1000]
test = train_data[-1000:]

train_x = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
train_y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.fit(
    x=train_x, 
    y=train_y,
    epochs=20,
    batch_size=64,
    validation_data= (test_x, test_y))

score = model.evaluate(
    x=test_x,
    y=test_y)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 







