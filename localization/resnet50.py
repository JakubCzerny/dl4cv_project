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
        horizontal_flip=True,
        shear_range=0.05
    )

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

    file_name = 'ResNet50_FC_30_[700]_[0.5]_2e-05_1_0.81.hdf5'
    base_model = load_model(PATH_MODELS+file_name)

    # Remove 4 last laters which are part of classification network
    for i in range(4):
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)

    for nodes, dropout in zip(l_nodes, l_dropouts):
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(4, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=optimizers.SGD(lr=l_rate), loss=loss_l2, metrics=[IoU])

    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=num_train/batchsize*2.0,
            validation_data=test_generator,
            validation_steps=num_test/batchsize,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=[
                callbacks.ModelCheckpoint(filepath=PATH_MODELS + model_name+'{IoU:.2f}.hdf5',
                                          monitor='IoU', mode='max', save_best_only=True, save_weights_only=False),
                callbacks.EarlyStopping(monitor='IoU', patience=3),
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


l_rates = [5e-5, 5e-4, 1e-4, 2e-5]
l_nodes = [[100],[200],[500],[50],[1000]]
dropouts = [[0],[0.25],[0.5]]
epochs = 20

for lr in l_rates:
    for nodes in l_nodes:
        for dropouts in dropouts:
            train(128,epochs,nodes,dropouts,lr,0.95)


# l_rates = [1e-3, 5e-4, 1e-5]
# l_nodes = [[100,100],[200,200],[500,500]]
# dropouts = [[0,0],[0.5,0.5],[0,0.5]]
# epochs = 20
#
# for lr in l_rates:
#     for nodes in l_nodes:
#         for dropouts in dropouts:
#             train(200,epochs,nodes,dropouts,lr,0.95)
