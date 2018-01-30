import os
import glob
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras import optimizers, callbacks
from keras.applications import resnet50
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

target_size = (350,350)
batchsize = 100

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

def acc_top_n(y_true, probs, n=3):
    output = np.argsort(probs,axis=1)
    top_n = output[:,-n:]
    acc_top_n = []
    for i, val in enumerate(y):
        acc_top_n.append(val in top_n[i,:])
    acc_top_n = np.sum(acc_top_n) / float(len(acc_top_n))
    return acc_top_n

datagen = image.ImageDataGenerator(
    preprocessing_function=imagenet_utils.preprocess_input,
    horizontal_flip=True,
)

generator = datagen.flow_from_directory(
    PATH_TEST,
    batch_size=batchsize,
    target_size=target_size
)

best_model = 'ResNet50_FC_30_[700]_[0.5]_2e-05_1_0.81.hdf5'
model = load_model(PATH_MODELS+best_model)

probs = model.predict_generator(generator, workers=4, use_multiprocessing=True, verbose=1)

# x,y = generator.next()
# y = np.argmax(y, axis=1)
# probs = model.predict_on_batch(x)

print acc_top_n(y, probs, 1)
print acc_top_n(y, probs, 2)
print acc_top_n(y, probs, 3)
print acc_top_n(y, probs, 5)
