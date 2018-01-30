import glob
from pathlib import Path
import os
from keras import backend as K
from keras.applications import resnet50
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import optimizers, callbacks

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print K.tensorflow_backend._get_available_gpus()

target_size = (350,350)

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

def train(batchsize, epochs, l_rate, momentum):
    model_name = 'ResNet50_TUNE_CONV_'+str(epochs)+'_'+str(l_rate)+'_'
    print "Training:",model_name

    datagen = image.ImageDataGenerator(
        preprocessing_function=imagenet_utils.preprocess_input,
        horizontal_flip=True,
        shear_range=0.1,
        rotation_range=10
    )

    train_generator = datagen.flow_from_directory(
        PATH_TRAIN,
        batch_size=batchsize,
        target_size=target_size
    )

    test_generator = datagen.flow_from_directory(
        PATH_TEST,
        batch_size=batchsize,
        target_size=target_size
    )

    num_train = train_generator.n
    num_test = test_generator.n

    best_model = 'ResNet50_FC_30_[500]_[0.5]_2e-05_1_0.81.hdf5'
    model = load_model(PATH_MODELS+best_model)

    # To keep trainable last conv module use :147
    for layer in model.layers[:153]:
        layer.trainable = False

    model = Model(inputs=model.input, outputs=model.output)
    model.compile(optimizer=optimizers.RMSprop(lr=l_rate), loss='categorical_crossentropy', metrics=['acc'])

    print model.summary()
    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=num_train/batchsize*2.0,
            validation_data=test_generator,
            validation_steps=num_test/batchsize,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=[
                callbacks.ModelCheckpoint(filepath=PATH_MODELS + model_name+'{val_acc:.2f}.hdf5',
                                          monitor='val_acc', save_best_only=True, save_weights_only=False),
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

    print "Final Accuracy: {:.2f}%".format(accuracy * 100)


l_rates = [1e-5, 5e-4, 1e-4]
epochs = 10
batch_size = 30

for lr in l_rates:
    train(batch_size,epochs,lr,0.95)
