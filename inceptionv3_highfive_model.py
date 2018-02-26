from keras.applications.resnet50 import ResNet50
from keras import activations
from keras.preprocessing import image
from keras.layers import Input,Flatten,Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


def model_inceptionv3(height,width,depth,classes):
    #Built Inception with imagenet pre-trained weights
    model_inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(width,height,depth))
    model_inceptionv3.summary()

    """
    #avoid input routinely error based on backend
    img_rows,img_cols = 299,299

    if K.image_data_format() == 'channels_first':
        input_crop = input_crop.reshape(input_crop.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        input_crop = input_crop.reshape(input_crop.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    """


    #Use InceptionV3
    output_inceptionv3 = model_inceptionv3.output

    glob_avrg_pool = GlobalAveragePooling2D()(output_inceptionv3)

    fc_1 = Dense(1024, activation='elu', name='fc_1')(glob_avrg_pool)

    # and a logistic layer 
    predictions = Dense(classes, activation='softmax', name='predictions')(fc_1)

    #New trainable model
    model = Model(inputs=model_inceptionv3.input, output= predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model_inceptionv3.layers:
        layer.trainable = False


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
