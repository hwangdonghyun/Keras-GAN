from keras import backend as K
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt


def generator_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(64,64,3),filters=64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 2, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 4,  kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 8,  kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))
    model.add(Flatten())



    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(64*8*4*4))
    model.add(Reshape((4,4,64*8)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))


    #8*8*64*4
    model.add(UpSampling2D())
    model.add(Conv2D(64*4,kernel_size=5,padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64 * 2, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(3, kernel_size=4, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.summary()

def discremenator_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(64, 64, 3), filters=64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 2, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 4, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Conv2D(64 * 8, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("sigmoid"));
    model.summary()

generator_model()
discremenator_model()