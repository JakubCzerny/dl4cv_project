'''
    Author: Jakub Czerny
    Email: jakub-czerny@outlook.com
    Deep Learning for Computer Vision
    Python Version: 2.7
'''

from utils import extended_image as extended_image

import os
import glob
import numpy as np
from pathlib import Path
from keras import backend as K
from keras.applications import resnet50
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D
from keras import optimizers, callbacks

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print K.tensorflow_backend._get_available_gpus()

target_size = (350,350)

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

# It's nothing else than sum square of errors
def loss_l2(y_true, y_pred):
    y_true = tf.clip_by_value(y_true,0,target_size[0])
    y_pred = tf.clip_by_value(y_pred,0,target_size[0])
    return K.mean(K.pow(np.subtract(y_true, y_pred),2))

# Intersetion over Union as the metric of goodness of the model
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


def train(batchsize, epochs, l_nodes, l_dropouts, l_rate, momentum):
    model_name = 'ResNet50_FC_localization_'+str(epochs)+'_'+str(l_nodes)+'_'+str(l_dropouts)+'_'+str(l_rate)+'_'
    print "Training:",model_name

    datagen = extended_image.ImageDataGenerator(
        preprocessing_function=imagenet_utils.preprocess_input,
    )

    '''
        I had to modify Keras datagenerator by adding `class_mode` to load also the bounding boxes coordinates
    '''

    train_generator = datagen.flow_from_directory(
        PATH_TRAIN,
        batch_size=batchsize,
        target_size=target_size,
        class_mode='bboxes'
    )

    test_generator = datagen.flow_from_directory(
        PATH_TEST,
        batch_size=batchsize,
        target_size=target_size,
        class_mode='bboxes'
    )

    num_train = train_generator.n
    num_test = test_generator.n


    # Load the best classifier
    file_name = 'ResNet50_classifiacation_0.82.hdf5'
    base_model = load_model(PATH_MODELS+file_name)

    # Remove 4 last laters which are part of classification network
    for i in range(4):
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    #x = Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding="same")(x)
    #x = Conv2D(filters=256,kernel_size=(2,2),strides=(1,1),padding="same")(x)
    x = GlobalAveragePooling2D()(x)

    for nodes, dropout in zip(l_nodes, l_dropouts):
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(4,activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=optimizers.RMSprop(lr=l_rate), loss=loss_l2, metrics=[IoU])

    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=num_train/batchsize,
            validation_data=test_generator,
            validation_steps=num_test/batchsize,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=[
                callbacks.ModelCheckpoint(filepath=PATH_MODELS + model_name+'{val_IoU:.2f}.hdf5',
                                          monitor='val_IoU', mode='max', save_best_only=True, save_weights_only=False),
                callbacks.EarlyStopping(monitor='val_loss', patience=3),
            ],
        )
    except KeyboardInterrupt:
        pass

    score, accuracy = model.evaluate_generator(
        test_generator,
        steps=num_test/batchsize,
        use_multiprocessing=True
    )

    print "Final Accuracy: {:.2f}%".format(accuracy * 100   )


l_rates = [1e-4,5e-4]
l_nodes = [[128],[256],[512]]
dropouts = [[0.5]]
epochs = 30
batch_size = 128

for lr in l_rates:
    for nodes in l_nodes:
        for dropout in dropouts:
            train(batch_size,epochs,nodes,dropout,lr,0.95)

l_rates = [1e-4,5e-4]
l_nodes = [[128]*2,[256]*2,[512]*2]
dropouts = [[0.5]*2,[0.25]*2]
epochs = 30
batch_size = 128

for lr in l_rates:
    for nodes in l_nodes:
        for dropout in dropouts:
            train(batch_size,epochs,nodes,dropout,lr,0.95)
