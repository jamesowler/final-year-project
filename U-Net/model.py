from keras.layers import Input, Conv2D, Deconvolution2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Reshape, Permute, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from keras import losses
from keras.utils import plot_model

from params import params
import loss_funcs

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
    loss_mathod = getattr(loss_funcs, params['loss_method'])
    model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

    return model


def model_2d_u_net_full(params):
    '''
    2D U-net implementation using Keras with tensorflow as the backend
    '''

    kernal_size = (3, 3)
    img_rows, img_columns = params['image_size_x'], params['image_size_y']

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
    flat_1 = Reshape((params['image_size_x']*params['image_size_y'], params['n_classes']))(conv7)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    # compile model with binary cross-entropy loss and Adam optimiser
    model = Model(inputs=[inputs], outputs=[act_last])
    loss_mathod = getattr(loss_funcs, params['loss_method'])
    model.compile(loss=loss_mathod, optimizer=Adam(lr=float(params['learning_rate'])))

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
    loss_mathod = getattr(loss_funcs, params['loss_method'])
    model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

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
    pool0 = MaxPooling2D((2,2))(act0)

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
    conv9 = Conv2D(32, kernal_size, padding='same', name= 'conv4_1')(up1)
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


def fully_conv_expri_network(params):

    ''' 
    Implementation of a fully convolutional network inspired by the network seen in Jiang et al. 
    '''
    kernal_size = (3, 3)
    img_rows, img_columns = params['patch_size'], params['patch_size']
    inputs = Input((img_rows, img_columns, params['n_channels']))

    # 8 down convs ========================================
    conv1 = Conv2D(10, kernal_size, padding='same', name='conv1')(inputs)
    bn = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn)

    # Residual block 1:
    res1_1 = Conv2D(10, kernal_size, padding='same', name='res1_1')(act1)
    bn = BatchNormalization()(res1_1)
    act = Activation('relu')(bn)
    res1_2 = Conv2D(10, kernal_size, padding='same', name='res1_2')(act)
    bn = BatchNormalization()(res1_2)
    act2 = Activation('relu')(bn)
    
    res_out_1 = concatenate([act1, act2], axis=-1)  # to concat 4

    # 16 down convs ========================================
    conv_down_1 = Conv2D(20, kernal_size, strides=2, padding='same', name='conv_down_1')(res_out_1)
    bn = BatchNormalization()(conv_down_1)
    act3 = Activation('relu')(bn)

    # Residual block 2:
    res2_1 = Conv2D(20, kernal_size, padding='same', name='res2_1')(act3)
    bn = BatchNormalization()(res2_1)
    act = Activation('relu')(bn)
    res2_2 = Conv2D(20, kernal_size, padding='same', name='res2_2')(act)
    bn = BatchNormalization()(res2_2)
    act4 = Activation('relu')(bn)

    res_out_2 = concatenate([act3, act4], axis=-1)  # to concat 3

    # 32 down convs ========================================
    conv_down_2 = Conv2D(40, kernal_size, strides=2, padding='same', name='conv_down_2')(res_out_2)
    bn = BatchNormalization()(conv_down_2)
    act5 = Activation('relu')(bn)

    # Residual block 3:
    res3_1 = Conv2D(40, kernal_size, padding='same', name='res3_1')(act5)
    bn = BatchNormalization()(res3_1)
    act = Activation('relu')(bn)
    res3_2 = Conv2D(40, kernal_size, padding='same', name='res3_2')(act)
    bn = BatchNormalization()(res3_2)
    act6 = Activation('relu')(bn)

    res_out_3 = concatenate([act5, act6], axis=-1)  # to concat 2

    # 64 down convs ========================================
    conv_down_3 = Conv2D(80, kernal_size, strides=2, padding='same', name='conv_down_3')(res_out_3)
    bn = BatchNormalization()(conv_down_3)
    act7 = Activation('relu')(bn)

    # Residual block 4:
    res4_1 = Conv2D(80, kernal_size, padding='same', name='res4_1')(act7)
    bn = BatchNormalization()(res4_1)
    act = Activation('relu')(bn)
    res4_2 = Conv2D(80, kernal_size, padding='same', name='res4_2')(act)
    bn = BatchNormalization()(res4_2)
    act8 = Activation('relu')(bn)

    res_out_4 = concatenate([act7, act8], axis=-1)  # to concat 1

    # 128 down convs ========================================
    conv_down_4 = Conv2D(160, kernal_size, strides=2, padding='same', name='conv_down_4')(res_out_4)
    bn = BatchNormalization()(conv_down_4)
    act9 = Activation('relu')(bn)

    # Residual block 5:
    res5_1 = Conv2D(160, kernal_size, padding='same', name='res5_1')(act9)
    bn = BatchNormalization()(res5_1)
    act = Activation('relu')(bn)
    res5_2 = Conv2D(160, kernal_size, padding='same', name='res5_2')(act)
    bn = BatchNormalization()(res5_2)
    act10 = Activation('relu')(bn)

    res_out_5 = concatenate([act9, act10], axis=-1)

    # 64 up convs ========================================
    deconv_1 = Deconvolution2D(80, kernal_size, subsample=(2,2), padding='same', name='deconv_1')(res_out_5)
    bn = BatchNormalization()(deconv_1)
    act11 = Activation('relu')(bn)

    concat_1 = concatenate([act11, res_out_4], axis=-1)

    # Resblock 6:
    res6_1 = Conv2D(80, kernal_size, padding='same', name='res6_1')(concat_1)
    bn = BatchNormalization()(res6_1)
    act = Activation('relu')(bn)
    res6_2 = Conv2D(80, kernal_size, padding='same', name='res6_2')(act)
    bn = BatchNormalization()(res6_2)
    act12 = Activation('relu')(bn)

    res_out_6 = concatenate([concat_1, act12], axis=-1)

    # 32 down convs ========================================
    deconv_2 = Deconvolution2D(40, kernal_size, subsample=(2,2), padding='same', name='deconv_2')(res_out_6)
    bn = BatchNormalization()(deconv_2)
    act13 = Activation('relu')(bn)

    concat_2 = concatenate([act13, res_out_3], axis=-1)

    # Resblock 7:
    res7_1 = Conv2D(40, kernal_size, padding='same', name='res7_1')(concat_2)
    bn = BatchNormalization()(res7_1)
    act = Activation('relu')(bn)
    res7_2 = Conv2D(40, kernal_size, padding='same', name='res7_2')(act)
    bn = BatchNormalization()(res7_2)
    act14 = Activation('relu')(bn)

    res_out_7 = concatenate([concat_2, act14], axis=-1)

    # 16 down convs ========================================
    deconv_3 = Deconvolution2D(20, kernal_size, subsample=(2,2), padding='same', name='deconv_3')(res_out_7)
    bn = BatchNormalization()(deconv_3)
    act15 = Activation('relu')(bn)

    concat_3 = concatenate([act15, res_out_2], axis=-1)

    # Resblock 8:
    res8_1 = Conv2D(20, kernal_size, padding='same', name='res8_1')(concat_3)
    bn = BatchNormalization()(res8_1)
    act = Activation('relu')(bn)
    res8_2 = Conv2D(20, kernal_size, padding='same', name='res8_2')(act)
    bn = BatchNormalization()(res8_2)
    act16 = Activation('relu')(bn)

    res_out_8 = concatenate([concat_3, act16], axis=-1)

     # 8 down convs ========================================
    deconv_4 = Deconvolution2D(10, kernal_size, subsample=(2,2), padding='same', name='deconv_4')(res_out_8)
    bn = BatchNormalization()(deconv_4)
    act17 = Activation('relu')(bn)

    concat_4 = concatenate([act17, res_out_1], axis=-1)

    # Resblock 9:
    res9_1 = Conv2D(10, kernal_size, padding='same', name='res9_1')(concat_4)
    bn = BatchNormalization()(res9_1)
    act = Activation('relu')(bn)
    res9_2 = Conv2D(10, kernal_size, padding='same', name='res9_2')(act)
    bn = BatchNormalization()(res9_2)
    act18 = Activation('relu')(bn)

    res_out_9 = concatenate([concat_4, act18], axis=-1)
    
    ######## Final output layers ########
    conv_out = Conv2D(params['n_classes'], (1, 1), padding='same', name='conv_out')(res_out_9)
    flat_1 = Reshape((params['patch_size']*params['patch_size'], params['n_classes']))(conv_out)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    model = Model(inputs=[inputs], outputs=[act_last])

    loss_mathod = getattr(loss_funcs, params['loss_method'])
    model.compile(loss=loss_mathod, optimizer=Adam(lr=float(params['learning_rate'])))

    return model


def fully_conv_expri_network_full(params):

    ''' 
    Implementation of a fully convolutional network inspired by the network seen in Jiang et al. 
    '''
    kernal_size = (3, 3)
    img_rows, img_columns = params['image_size_x'], params['image_size_y']
    inputs = Input((img_rows, img_columns, params['n_channels']))

    # 8 down convs ========================================
    conv1 = Conv2D(10, kernal_size, padding='same', name='conv1')(inputs)
    bn = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn)

    # Residual block 1:
    res1_1 = Conv2D(10, kernal_size, padding='same', name='res1_1')(act1)
    bn = BatchNormalization()(res1_1)
    act = Activation('relu')(bn)
    res1_2 = Conv2D(10, kernal_size, padding='same', name='res1_2')(act)
    bn = BatchNormalization()(res1_2)
    act2 = Activation('relu')(bn)
    
    res_out_1 = concatenate([act1, act2], axis=-1)  # to concat 4

    # 16 down convs ========================================
    conv_down_1 = Conv2D(20, kernal_size, strides=2, padding='same', name='conv_down_1')(res_out_1)
    bn = BatchNormalization()(conv_down_1)
    act3 = Activation('relu')(bn)

    # Residual block 2:
    res2_1 = Conv2D(20, kernal_size, padding='same', name='res2_1')(act3)
    bn = BatchNormalization()(res2_1)
    act = Activation('relu')(bn)
    res2_2 = Conv2D(20, kernal_size, padding='same', name='res2_2')(act)
    bn = BatchNormalization()(res2_2)
    act4 = Activation('relu')(bn)

    res_out_2 = concatenate([act3, act4], axis=-1)  # to concat 3

    # 32 down convs ========================================
    conv_down_2 = Conv2D(40, kernal_size, strides=2, padding='same', name='conv_down_2')(res_out_2)
    bn = BatchNormalization()(conv_down_2)
    act5 = Activation('relu')(bn)

    # Residual block 3:
    res3_1 = Conv2D(40, kernal_size, padding='same', name='res3_1')(act5)
    bn = BatchNormalization()(res3_1)
    act = Activation('relu')(bn)
    res3_2 = Conv2D(40, kernal_size, padding='same', name='res3_2')(act)
    bn = BatchNormalization()(res3_2)
    act6 = Activation('relu')(bn)

    res_out_3 = concatenate([act5, act6], axis=-1)  # to concat 2

    # 64 down convs ========================================
    conv_down_3 = Conv2D(80, kernal_size, strides=2, padding='same', name='conv_down_3')(res_out_3)
    bn = BatchNormalization()(conv_down_3)
    act7 = Activation('relu')(bn)

    # Residual block 4:
    res4_1 = Conv2D(80, kernal_size, padding='same', name='res4_1')(act7)
    bn = BatchNormalization()(res4_1)
    act = Activation('relu')(bn)
    res4_2 = Conv2D(80, kernal_size, padding='same', name='res4_2')(act)
    bn = BatchNormalization()(res4_2)
    act8 = Activation('relu')(bn)

    res_out_4 = concatenate([act7, act8], axis=-1)  # to concat 1

    # 128 down convs ========================================
    conv_down_4 = Conv2D(160, kernal_size, strides=2, padding='same', name='conv_down_4')(res_out_4)
    bn = BatchNormalization()(conv_down_4)
    act9 = Activation('relu')(bn)

    # Residual block 5:
    res5_1 = Conv2D(160, kernal_size, padding='same', name='res5_1')(act9)
    bn = BatchNormalization()(res5_1)
    act = Activation('relu')(bn)
    res5_2 = Conv2D(160, kernal_size, padding='same', name='res5_2')(act)
    bn = BatchNormalization()(res5_2)
    act10 = Activation('relu')(bn)

    res_out_5 = concatenate([act9, act10], axis=-1)

    # 64 up convs ========================================
    deconv_1 = Deconvolution2D(80, kernal_size, subsample=(2,2), padding='same', name='deconv_1')(res_out_5)
    bn = BatchNormalization()(deconv_1)
    act11 = Activation('relu')(bn)

    concat_1 = concatenate([act11, res_out_4], axis=-1)

    # Resblock 6:
    res6_1 = Conv2D(80, kernal_size, padding='same', name='res6_1')(concat_1)
    bn = BatchNormalization()(res6_1)
    act = Activation('relu')(bn)
    res6_2 = Conv2D(80, kernal_size, padding='same', name='res6_2')(act)
    bn = BatchNormalization()(res6_2)
    act12 = Activation('relu')(bn)

    res_out_6 = concatenate([concat_1, act12], axis=-1)

    # 32 down convs ========================================
    deconv_2 = Deconvolution2D(40, kernal_size, subsample=(2,2), padding='same', name='deconv_2')(res_out_6)
    bn = BatchNormalization()(deconv_2)
    act13 = Activation('relu')(bn)

    concat_2 = concatenate([act13, res_out_3], axis=-1)

    # Resblock 7:
    res7_1 = Conv2D(40, kernal_size, padding='same', name='res7_1')(concat_2)
    bn = BatchNormalization()(res7_1)
    act = Activation('relu')(bn)
    res7_2 = Conv2D(40, kernal_size, padding='same', name='res7_2')(act)
    bn = BatchNormalization()(res7_2)
    act14 = Activation('relu')(bn)

    res_out_7 = concatenate([concat_2, act14], axis=-1)

    # 16 down convs ========================================
    deconv_3 = Deconvolution2D(20, kernal_size, subsample=(2,2), padding='same', name='deconv_3')(res_out_7)
    bn = BatchNormalization()(deconv_3)
    act15 = Activation('relu')(bn)

    concat_3 = concatenate([act15, res_out_2], axis=-1)

    # Resblock 8:
    res8_1 = Conv2D(20, kernal_size, padding='same', name='res8_1')(concat_3)
    bn = BatchNormalization()(res8_1)
    act = Activation('relu')(bn)
    res8_2 = Conv2D(20, kernal_size, padding='same', name='res8_2')(act)
    bn = BatchNormalization()(res8_2)
    act16 = Activation('relu')(bn)

    res_out_8 = concatenate([concat_3, act16], axis=-1)

     # 8 down convs ========================================
    deconv_4 = Deconvolution2D(10, kernal_size, subsample=(2,2), padding='same', name='deconv_4')(res_out_8)
    bn = BatchNormalization()(deconv_4)
    act17 = Activation('relu')(bn)

    concat_4 = concatenate([act17, res_out_1], axis=-1)

    # Resblock 9:
    res9_1 = Conv2D(10, kernal_size, padding='same', name='res9_1')(concat_4)
    bn = BatchNormalization()(res9_1)
    act = Activation('relu')(bn)
    res9_2 = Conv2D(10, kernal_size, padding='same', name='res9_2')(act)
    bn = BatchNormalization()(res9_2)
    act18 = Activation('relu')(bn)

    res_out_9 = concatenate([concat_4, act18], axis=-1)
    
    ######## Final output layers ########
    conv_out = Conv2D(params['n_classes'], (1, 1), padding='same', name='conv_out')(res_out_9)
    flat_1 = Reshape((params['image_size_x']*params['image_size_y'], params['n_classes']))(conv_out)
    flat_2 = Permute((1,2))(flat_1)
    act_last = Activation('sigmoid')(flat_2)

    model = Model(inputs=[inputs], outputs=[act_last])

    loss_mathod = getattr(loss_funcs, params['loss_method'])
    model.compile(loss=loss_mathod, optimizer=Adam(lr=float(params['learning_rate'])))

    return model


if __name__ == '__main__':
    model = fully_conv_expri_network_full(params)
    # plot_model(model, './model_diagram.png', show_shapes=True)
    print(model.summary())

