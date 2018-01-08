import glob
from pathlib import Path
import os
from keras import backend as K
from keras.applications import vgg16
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import optimizers, callbacks

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print K.tensorflow_backend._get_available_gpus()

model_name = 'vgg16__FC'
target_size = (224,224)

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'model/trained/'

def train(batchsize, epochs, l_nodes, l_dropouts, l_rate, momentum):
    datagen = image.ImageDataGenerator(
        preprocessing_function=imagenet_utils.preprocess_input
        # horizontal_flip=True,
        # shear_range=0.15
    )

    print os.getcwd()

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

    base_model = vgg16.VGG16(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for nodes, dropout in zip(l_nodes, l_dropouts):
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(120, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:19]:
        layer.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=l_rate), loss='categorical_crossentropy', metrics=['acc'])

    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=num_train/batchsize,
            validation_data=test_generator,
            validation_steps=num_test/batchsize,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=[
                callbacks.ModelCheckpoint(filepath=PATH_MODELS + model_name+'{val_acc:.2f}.hdf5',
                                          monitor='val_acc', save_best_only=True, save_weights_only=False),
                callbacks.EarlyStopping(monitor='val_loss', patience=2),
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


train(200,5,[100,100],[0,0],0.003,0.95)
