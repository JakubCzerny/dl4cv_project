import glob
from pathlib import Path
import os
from keras import backend as K
from keras.applications import resnet50
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
from PIL import Image

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(K.tensorflow_backend._get_available_gpus())

target_size = (224,224)

data_path = 'data/test'
PATH_MODELS = 'models/trained/ResNet50_FC_Ep_10_N_200_Decay_0_LR_5e-05_BS_200_0.71.hdf5'
output_path = 'models/output/'

datagen = image.ImageDataGenerator(preprocessing_function=imagenet_utils.preprocess_input,)

generator = datagen.flow_from_directory(
    data_path,
    batch_size=500,
    target_size=(224,224),
)

model = load_model(PATH_MODELS)
x,y = generator.next()

y = np.argmax(y, axis=1)
probs = model.predict_on_batch(x)
y_pred = np.argmax(probs, axis=1)
mapping = dict((l,k) for (k,l) in generator.class_indices.items())

print(sum(y==y_pred)/float(len(y)))

for idx, (im, t_class, p_class) in enumerate(zip(x,y,y_pred)):
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.68
    i = Image.fromarray(im[:,:,::-1].astype(np.uint8))
    if t_class == p_class:
        i.save(output_path+'correct/'+str(mapping[t_class])+'_'+str(mapping[p_class])+'_'+str(idx)+'.jpg')
    else:
        i.save(output_path+'incorrect/'+str(mapping[t_class])+'_'+str(mapping[p_class])+'_'+str(idx)+'.jpg')
