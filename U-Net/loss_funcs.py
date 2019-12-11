import keras.backend as K
import numpy as np
from params import params

def bce(y_true, y_pred):
    loss = K.binary_crossentropy(y_true, y_pred)
    return loass

def weighted_bce(y_true, y_pred):
    '''
    Unbalanced class segmentation - weighted towards vessel class
    '''
    weights = (y_true * float(params['loss_weight'])) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

