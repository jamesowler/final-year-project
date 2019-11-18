from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Reshape, Permute, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from keras import losses
from keras.utils import plot_model

from params import params

def model_2d_u_net(params):
    '''
    2D U-net implementation using Keras with tensorflow as the backend
    '''

    kernal_size = (3, 3)
    img_rows, img_columns = params['patch_size'], params['patch_size']

    inputs = Input((img_rows, img_columns, params['n_channels']))

    ####
    # Encoding branch - down-sampling
    ####

    # Block 0: 32-64
    conv0 = Conv2D(32, kernal_size, padding='same', name='conv0_0')(inputs)
    bn = BatchNormalization()(conv0)
    act = Activation('relu')(bn)
    conv0 = Conv2D(64, kernal_size, padding='same', name='conv0_1')(act)
    bn = BatchNormalization()(conv0)
    act0 = Activation('relu')(bn)

    # Pool 0 - 2x2
    pool0 = MaxPooling2D((2, 2))(act0)


    # Block 1: 64-128
    conv1 = Conv2D(64, kernal_size, padding='same', name='conv1_0')(pool0)
    bn = BatchNormalization()(conv1)
    act = Activation('relu')(bn)
    conv1 = Conv2D(128, kernal_size, padding='same', name='conv1_1')(act)
    bn = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn)

    # Pool 1 - 2x2
    pool1 = MaxPooling2D((2,2))(act1)


    # Block 2: 128-256
    conv2 = Conv2D(128, kernal_size, padding='same', name='conv2_0')(pool1)
    bn = BatchNormalization()(conv2)
    act = Activation('relu')(bn)
    conv2 = Conv2D(256, kernal_size, padding='same', name='conv2_1')(act)
    bn = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn)

    # Pool 2 - 2x2
    pool2 = MaxPooling2D((2,2))(act2)

    #### Base layers
    # Block 3: 256-256
    conv3 = Conv2D(256, kernal_size, padding='same', name='conv3_0')(pool2)
    bn = BatchNormalization()(conv3)
    act = Activation('relu')(bn)
    conv3 = Conv2D(256, kernal_size, padding='same', name='conv3_1')(act)
    bn = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn)

    ####
    # Decoding branch - up-sampling
    ####

    # up 0 - 2x2
    up0 = concatenate([UpSampling2D(size=(2,2))(act3), (act2)], axis=-1)

    # Block 4: 256-128
    conv4 = Conv2D(256, kernal_size, padding='same', name='conv4_0')(up0)
    bn = BatchNormalization()(conv4)
    act = Activation('relu')(bn)
    conv4 = Conv2D(128, kernal_size, padding='same', name='conv4_1')(act)
    bn = BatchNormalization()(conv4)
    act = Activation('relu')(bn)


    # up 1 - 2x2
    up1 = concatenate([UpSampling2D(size=(2,2))(act), act1], axis=-1)

    # Block 5: 128-64
    conv5 = Conv2D(128, kernal_size, padding='same', name='conv5_0')(up1)
    bn = BatchNormalization()(conv5)
    act = Activation('relu')(bn)
    conv5 = Conv2D(64, kernal_size, padding='same', name='conv5_1')(act)
    bn = BatchNormalization()(conv5)
    act = Activation('relu')(bn)


    # up 2 - 2x2
    up2 = concatenate([UpSampling2D(size=(2,2))(act), act0], axis=-1)

    # Block 6: 64-32
    conv6 = Conv2D(64, kernal_size, padding='same', name='conv6_0')(up2)
    bn = BatchNormalization()(conv6)
    act = Activation('relu')(bn)
    conv6 = Conv2D(32, kernal_size, padding='same', name='conv6_1')(act)
    bn = BatchNormalization()(conv6)
    act = Activation('relu')(bn)


    ### Output layer:
    conv7 = Conv2D(params['n_classes'], (1, 1), padding='same', name='conv7')(act)
    flat_1 = Reshape((params['patch_size']*params['patch_size'], params['n_classes']))(conv7)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    # compile model with binary cross-entropy loss and Adam optimiser
    model = Model(inputs=[inputs], outputs=[act_last])
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(lr=float(params['learning_rate'])))

    return model


def model_2d_u_net_shallow(params):
    '''
    2D U-net implementation using Keras with tensorflow as the backend - fewer layers (with only 2 max pooling operations) - more suited to patch base segmentation approach
    '''

    kernal_size = (3, 3)
    img_rows, img_columns = params['patch_size'], params['patch_size']

    inputs = Input((img_rows, img_columns, params['n_channels']))

    ####
    # Encoding branch - down-sampling
    ####

    # Block 0: 32-64
    conv0 = Conv2D(32, kernal_size, padding='same', name='conv0_0')(inputs)
    bn = BatchNormalization()(conv0)
    act = Activation('relu')(bn)
    conv0 = Conv2D(64, kernal_size, padding='same', name='conv0_1')(act)
    bn = BatchNormalization()(conv0)
    act0 = Activation('relu')(bn)

    # Pool 0 - 2x2
    pool0 = MaxPooling2D((2, 2))(act0)


    # Block 1: 64-128
    conv1 = Conv2D(64, kernal_size, padding='same', name='conv1_0')(pool0)
    bn = BatchNormalization()(conv1)
    act = Activation('relu')(bn)
    conv1 = Conv2D(128, kernal_size, padding='same', name='conv1_1')(act)
    bn = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn)

    # Pool 1 - 2x2
    pool1 = MaxPooling2D((2,2))(act1)

    #### Base layers
    # Block 3: 128-128
    conv3 = Conv2D(128, kernal_size, padding='same', name='conv3_0')(pool1)
    bn = BatchNormalization()(conv3)
    act = Activation('relu')(bn)
    conv3 = Conv2D(128, kernal_size, padding='same', name='conv3_1')(act)
    bn = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn)

    ####
    # Decoding branch - up-sampling
    ####

    # up 0 - 2x2
    up0 = concatenate([UpSampling2D(size=(2,2))(act3), (act1)], axis=-1)

    # Block 4: 128-64
    conv4 = Conv2D(128, kernal_size, padding='same', name='conv4_0')(up0)
    bn = BatchNormalization()(conv4)
    act = Activation('relu')(bn)
    conv4 = Conv2D(64, kernal_size, padding='same', name='conv4_1')(act)
    bn = BatchNormalization()(conv4)
    act = Activation('relu')(bn)


    # up 1 - 2x2
    up1 = concatenate([UpSampling2D(size=(2,2))(act), act0], axis=-1)

    # Block 5: 64-32
    conv5 = Conv2D(64, kernal_size, padding='same', name='conv5_0')(up1)
    bn = BatchNormalization()(conv5)
    act = Activation('relu')(bn)
    conv5 = Conv2D(32, kernal_size, padding='same', name='conv5_1')(act)
    bn = BatchNormalization()(conv5)
    act = Activation('relu')(bn)


    ### Output layer:
    conv7 = Conv2D(params['n_classes'], (1, 1), padding='same', name='conv7')(act)
    flat_1 = Reshape((params['patch_size']*params['patch_size'], params['n_classes']))(conv7)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    # compile model with binary cross-entropy loss and Adam optimiser
    model = Model(inputs=[inputs], outputs=[act_last])
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(lr=float(params['learning_rate'])))

    return model


def model_2d_u_net_shallow_dropout(params):
    
    '''
    Architecture taken from paper: 'Retina Blood Vessel Segmentation Using A U-Net Based Convolutional Nerual Network'
    '''

    kernal_size = (3, 3)
    img_rows, img_columns = params['patch_size'], params['patch_size']

    inputs = Input((img_rows, img_columns, params['n_channels']))

    ####
    # Encoding branch - down-sampling
    ####

    # Block 0: 32
    conv0 = Conv2D(32, kernal_size, padding='same', name='conv0_0')(inputs)
    act = Activation('relu')(conv0)
    do = Dropout(0.2)(act)
    conv1 = Conv2D(32, kernal_size, padding='same', name='conv0_1')(do)
    act0 = Activation('relu')(conv1)

    # Max pooling 32 -> 16
    pool0 = MaxPooling2D((2,2))(act1)

    # Block 1: 64
    conv2 = Conv2D(64, kernal_size, padding='same', name='conv1_0')(pool0)
    act = Activation('relu')(conv2)
    do = Dropout(0.2)(act)
    conv3 = Conv2D(64, kernal_size, padding='same', name='conv1_1')(do)
    act1 = Activation('relu')(conv3)

    # Max pooling 16 -> 8
    pool1 = MaxPooling2D((2,2))(act1)

    # Block 2: 128
    conv4 = Conv2D(128, kernal_size, padding='same', name= 'conv2_0')(pool1)
    act = Activation('relu')(conv4)
    do = Dropout(0.2)(act)
    conv5 = Conv2D(128, kernal_size, padding='same', name='conv2_1')(do)
    act2 = Activation('relu')(conv4)

    # Up-sampling 8 -> 16
    up0 = concatenate([UpSampling2D((2,2))(act2), act1], axis=-1)

    # Block 3: 64
    conv6 = Conv2D(64, kernal_size, padding='same', name= 'conv3_0')(up0)
    act = Activation('relu')(conv6)
    do = Dropout(0.2)(act)
    conv7 = Conv2D(64, kernal_size, padding='same', name= 'conv3_1')(do)
    act3 = Activation('relu')(conv7)

    # Up-sampling 16 -> 32
    up1 = concatenate([UpSampling2D((2,2))(act3), act0], axis=-1)

    # Block 4: 32
    conv8 = Conv2D(32, kernal_size, padding='same', name= 'conv4_0')(up1)
    act = Activation('relu')(conv8)
    do = Dropout(0.2)(act)
    conv9 = Conv2D(32, kernal_size, padding='same', name= 'conv4_0')(up1)
    act = Activation('relu')(conv8)

    ### Output layer:
    conv7 = Conv2D(params['n_classes'], (1, 1), padding='same', name='conv7')(act)
    flat_1 = Reshape((params['patch_size']*params['patch_size'], params['n_classes']))(conv7)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    # compile model with binary cross-entropy loss and Adam optimiser
    model = Model(inputs=[inputs], outputs=[act_last])
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(lr=float(params['learning_rate'])))

    return model


if __name__ == '__main__':
    model = model_2d_u_net(params)
    # plot_model(model, './model_diagram.png', show_shapes=True)
    print(model.summary())

