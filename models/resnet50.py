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

def train(batchsize, epochs, l_nodes, l_dropouts, l_rate, momentum, modules_to_drop):
    model_name = 'ResNet50_FC_'+str(epochs)+'_'+str(l_nodes)+'_'+str(l_dropouts)+'_'+str(l_rate)+'_'+str(modules_to_drop)+'_'
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

    if modules_to_drop == 0:
        layers_to_drop = 0
    elif modules_to_drop == 1:
        layers_to_drop = 11
    elif modules_to_drop == 2:
        layers_to_drop = 21
    elif modules_to_drop == 3:
        layers_to_drop = 33


    base_model = resnet50.ResNet50(include_top=False, weights='imagenet')

    # To keep trainable last conv module use :147, 153, 131, 121
    for layer in base_model.layers[:147]:
        layer.trainable = False

    for i in range(layers_to_drop):
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

    x = base_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    for nodes, dropout in zip(l_nodes, l_dropouts):
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(120, activation='softmax', kernel_regularizer=regularizers.l2(0.02))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=optimizers.RMSprop(lr=l_rate), loss='categorical_crossentropy', metrics=['acc'])
    print model.summary()

    for i,l in enumerate(model.layers):
        print i,l.trainable,l

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
