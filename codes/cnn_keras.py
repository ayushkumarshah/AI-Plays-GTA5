import tensorflow as tf
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, BatchNormalization,ZeroPadding2D
from keras.models import Sequential

from keras import backend as K
K.tensorflow_backend._get_available_gpus()



def cnn_keras():
	model = Sequential()
	input_shape=(120, 160, 1)

	model.add(Conv2D(96, (11,11), strides=(4,4), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3, 3), strides=(2, 2)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	model.add(Conv2D(256, (5, 5),padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((3, 3), strides=(2, 2)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	model.add(Conv2D(384, (3, 3),padding='same'))
	model.add(Conv2D(384, (3, 3),padding='same'))
	model.add(Conv2D(256, (3, 3),padding='same'))
	model.add(MaxPooling2D((3, 3), strides=(2, 2)))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

	model.add(Dense(4096))
	model.add(Activation('tanh'))
	model.add(Dropout(rate=0.5))

	model.add(Dense(4096))
	model.add(Activation('tanh'))
	model.add(Dropout(rate=0.5))

	model.add(Dense(3))
	model.add(Activation('softmax'))

	model.summary()
	return model