from utils import extended_image as extended_image

import os
import glob
from copy import deepcopy
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
batch_size = 200

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

def loss_l2(y_true, y_pred):
    return K.mean(K.pow(np.subtract(y_true, y_pred),2))

smooth = 1
def IoU(y_true, y_pred):
    x11, y11, x12, y12 = tf.split(tf.clip_by_value(y_true,0,target_size[0]), 4, axis=1)
    x21, y21, x22, y22 = tf.split(tf.clip_by_value(y_pred,0,target_size[0]), 4, axis=1)
    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))
    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))
    inter_area = (xI2 - xI1 + smooth) * (yI2 - yI1 + smooth)

    bboxes1_area = (x12 - x11 + smooth) * (y12 - y11 + smooth)
    bboxes2_area = (x22 - x21 + smooth) * (y22 - y21 + smooth)
    tf.Print(bboxes1_area, [bboxes1_area])
    tf.Print(bboxes2_area, [bboxes2_area])
    return inter_area / ((bboxes1_area + tf.transpose(bboxes2_area)) - inter_area)

model_name = 'models/trained/ResNet50_localization_0.58.hdf5'
model = load_model(model_name, custom_objects={'loss_l2': loss_l2, 'IoU': IoU})

datagen = extended_image.ImageDataGenerator(
    preprocessing_function=imagenet_utils.preprocess_input,
)

generator = datagen.flow_from_directory(
    PATH_TEST,
    batch_size=batch_size,
    target_size=target_size,
    shuffle=True,
    class_mode='bboxes',
)

x,y = generator.next()
y_pred = model.predict_on_batch(x)
print model.evaluate(x,y,batch_size)
mean = [103.939, 116.779, 123.68]

imgs = []
imgs2 = []

for im, bbox, bbox_pred in zip(x,y,y_pred):
    im[:,:,0] += mean[0]
    im[:,:,1] += mean[1]
    im[:,:,2] += mean[2]
    im = im[..., ::-1]
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    imgs2.append(deepcopy(im))

    draw_bbox = ImageDraw.Draw(im)
    draw_bbox.rectangle( (bbox[0], bbox[1], bbox[2], bbox[3]), outline="red")

    draw_pred_bbox = ImageDraw.Draw(im)
    draw_pred_bbox.rectangle( (bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]), outline="green")

    print '350x350', (bbox[0], bbox[1], bbox[2], bbox[3]), (bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3])
    imgs.append(im)
