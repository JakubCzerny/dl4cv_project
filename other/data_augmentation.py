import glob
from pathlib import Path
import os
from keras import backend as K
from keras.applications import resnet50
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, Conv1D, Reshape
from keras import optimizers, callbacks, regularizers

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(K.tensorflow_backend._get_available_gpus())

target_size = (224,224)

PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
PATH_MODELS = 'models/trained/'

def train(batchsize, epochs, l_nodes, l_dropout, l_rate, momentum, modules_to_drop, l_decay):
    model_name = 'ResNet50_FC_Ep_'+str(epochs)+'_N_'+str(l_nodes)+'_Decay_'+str(l_decay)+'_LR_'+str(l_rate)+'_DO_'+str(l_dropout)+'_'
    print("Training:",model_name)

    datagen = image.ImageDataGenerator(
        preprocessing_function=imagenet_utils.preprocess_input,
        horizontal_flip=True,
        shear_range=0.15,
        rotation_range=40,
        zoom_range=[0.5, 2],
	fill_mode='nearest'
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

    # To keep trainable last conv module use :147
    for layer in base_model.layers[:159]:
        layer.trainable = False

    for i in range(layers_to_drop):
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []

    x = base_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)

    #x = Dense(l_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x)
    #x = K.reshape(x,(1,-1,l_nodes,1))
    #x = Conv2D(10,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.1))(x)
    #x = GlobalAveragePooling2D()(x)
    x = Dense(l_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x)
    #x = Dense(l_nodes, activation='relu')(x)
    x = Dropout(l_dropout)(x)

    predictions = Dense(120, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=optimizers.Adam(lr=l_rate,decay=l_decay), loss='categorical_crossentropy', metrics=['acc'])

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

    print("Final Accuracy: {:.2f}%".format(accuracy * 100))


l_rates = [1e-3]
#l_rates = [5e-5]
#l_nodes = [[100],[200],[500],[1000]]
l_nodes = [500]
#l_nodes = [200]
#dropouts = [[0],[0.25],[0.5]]
#dropouts = [0]
dropouts = [0.5]
epochs = 10
modules_to_drop = [0]
#batch_sizes = [200,256]
batch_size = [256]
decays = [0.05]
#decays = [0]

for lr in l_rates:
    for nodes in l_nodes:
        for decay in decays:
            for dropout in dropouts:
                train(256,epochs,nodes,dropout,lr,0.95,0,decay)


# l_rates = [1e-3, 5e-4, 1e-5]
# l_nodes = [[100,100],[200,200],[500,500]]
# dropouts = [[0,0],[0.5,0.5],[0,0.5]]
# epochs = 20
#
# for lr in l_rates:
#     for nodes in l_nodes:
#         for dropouts in dropouts:
#             train(200,epochs,nodes,dropouts,lr,0.95)

