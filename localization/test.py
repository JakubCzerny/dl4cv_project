from utils import extended_image as extended_image

import os
import glob
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from keras import backend as K
from keras.applications import resnet50
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import optimizers, callbacks

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print K.tensorflow_backend._get_available_gpus()

target_size = (350,350)

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

def loss_l2(y_true, y_pred):
    return K.mean(K.pow(np.subtract(y_true, y_pred),2))


model_name = 'models/trained/ResNet50_FC_localization_20_[50]_[0]_1e-05_6583.16.hdf5'
model = load_model(model_name, custom_objects={'loss_l2': loss_l2})

datagen = extended_image.ImageDataGenerator(
    preprocessing_function=imagenet_utils.preprocess_input,
)

generator = datagen.flow_from_directory(
    PATH_TEST,
    batch_size=3,
    target_size=target_size,
    class_mode='bboxes',
)

x,y = generator.next()

mean = [103.939, 116.779, 123.68]

for im, bbox in zip(x,y):
    im[:,:,0] += mean[0]
    im[:,:,1] += mean[1]
    im[:,:,2] += mean[2]
    im = im[..., ::-1]
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.rectangle( (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]), outline="red")
    im.show()

# y = np.argmax(y, axis=1)
# probs = model.predict_on_batch(x)
