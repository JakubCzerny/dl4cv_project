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
batchsize = 64

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
    target_size=target_size,
    shuffle=True,
)
num = generator.n

best_model = 'ResNet50_classifiacation_0.82.hdf5'
model = load_model(PATH_MODELS+best_model)

probs = []
y = []
for idx in range(int(512 / float(batchsize))):
    print idx
    x_batch,y_batch = generator.next()
    probs_batch = model.predict_on_batch(x_batch)
    y.extend(y_batch)
    probs.extend(probs_batch)

probs = np.array(probs)
y = np.argmax(np.array(y), axis=1)

print acc_top_n(y, probs, 1)
print acc_top_n(y, probs, 2)
print acc_top_n(y, probs, 3)
print acc_top_n(y, probs, 5)
