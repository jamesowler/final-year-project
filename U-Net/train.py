import os
import time
import glob
import random

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from params import params
from data_utils import load_batch
from model import model_2d_u_net

def train():

    start_training = time.time()

    files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4\*')
    training_files = [i for i in files if 'training.png' in i]
    seg_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'
    
    # load in model
    # model = model_2d_u_net(params)
    model = load_model('./model.h5')

    # begin training
    for e in range(1, params['n_epochs'] + 1):
        
        random.shuffle(training_files)

        X, Y = load_batch(training_files, seg_dir, params['image_size_x'], params['image_size_y'], 1)

        seed = np.random.randint(0, 100)
        data_gen_args = dict(rotation_range=5.,
                             width_shift_range=0.5,
                             height_shift_range=0.5,
                             zoom_range=0.5,
                             fill_mode='constant',
                             horizontal_flip=True,
                             vertical_flip=True,
                             cval=0)

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(Y, augment=True, seed=seed)

        image_generator = image_datagen.flow(X, seed=seed)
        mask_generator = mask_datagen.flow(Y, seed=seed)
        train_generator = zip(image_generator, mask_generator)

        batches = 0

        print('Epoch: ', e)
        for X_batch, Y_batch in train_generator:
            Y_out = Y_batch.reshape((len(training_files), params['image_size_x'] * params['image_size_y'], 1))
            l1 = model.train_on_batch(X_batch, Y_out)
            batches =+ 1
            print(l1)     
            # insert logic about batch size later on
            if batches >= 1:
                break
    
    model.save('./model.h5')


if __name__ == '__main__':
    train()
        



