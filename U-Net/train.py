import os
import time
import glob
import random

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from params import params
from data_utils import load_batch, load_batch_patch_training
from model import model_2d_u_net, model_2d_u_net_shallow, model_2d_u_net_shallow_dropout

def train(params):

    start_training = time.time()

    files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4\*')
    training_files = [i for i in files if 'training.png' in i]
    seg_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'
    
    # load in model
    model = model_2d_u_net_shallow(params)
    # model = load_model('./model.h5')

    loss = []

    # begin training
    for e in range(1, params['n_epochs'] + 1):
        
        random.shuffle(training_files)

        X, Y = load_batch(training_files, seg_dir, params['image_size_x'], params['image_size_y'], 1, params['patch_size'])

        seed = np.random.randint(0, 10000)
        data_gen_args = dict(rotation_range=5.,
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
            Y_out = Y_batch.reshape((len(training_files), params['patch_size'] * params['patch_size'], 1))
            l1 = model.train_on_batch(X_batch, Y_out)
            loss.append(l1)
            batches =+ 1
            print(l1)     
            # insert logic about batch size later on
            if batches >= 1:
                break
        
        if e % 500 == 0:
            model.save(f'./patch_model_{str(e)}.h5')

    with open('./losses.txt', 'w') as f:
        for i in loss:
            f.write(str(i) + ',')


def train_from_data_dir(params):
    '''
    Train model using dataset of patches
    '''

    imgs_dir = os.path.join(params['data_dir'], 'imgs')
    segs_dir = os.path.join(params['data_dir'], 'segs')

    # list images in dataset directory
    imgs = os.listdir(imgs_dir)

    # train - validation split
    imgs_rand = imgs
    random.shuffle(imgs_rand)
    val_index_split = int(len(imgs_rand)*params['val_proportion'])
    train_imgs = imgs_rand[val_index_split:]
    val_imgs = imgs_rand[:val_index_split]

    n_batches_train = np.ceil(len(train_imgs)/params['batch_size']).astype('int')
    n_batches_val = np.ceil(len(val_imgs)/params['batch_size']).astype('int')

    for e in range(1, params['n_epochs'] + 1):

        print('Epoch: ', e)

        # randomize order of training images
        train_imgs_rand = train_imgs
        random.shuffle(train_imgs_rand)

        losses = []
        
        for b in range(n_batches_train):
            if (b+1)*params['batch_size'] < len(train_imgs_rand):
                batch_train_imgs = train_imgs_rand[b*params['batch_size']:(b+1)*params['batch_size']]
            else:
                batch_train_imgs = train_imgs_rand[b*params['batch_size']:]

            X, Y = load_batch_patch_training(batch_train_imgs, os.path.join(params['data_dir'], 'imgs'), os.path.join(params['data_dir'], 'segs'), params['patch_size'])
            seed = np.random.randint(0, 10000)
            data_gen_args = dict(rotation_range=5.,
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

            for X_aug, Y_aug in train_generator:
                Y_out = Y_aug.reshape(len(batch_train_imgs), params['patch_size'] * params['patch_size'], 1)
                loss = model.train_on_batch(X_aug, Y_out)
                losses.append(loss)
                print(loss)

                break
        
        for b in range(n_batches_val):
            pass
            
        if e % 1 == 0:
            model.save_weights(f'./patch_model_{str(e)}.h5')
    

def plot_losses(filename):
    
    pass

def train_patch():
    model = model_2d_u_net_shallow(params)
    model.load_weights(r'C:\Users\James\Projects\final-year-project\patch_model_20.h5')
    train_from_data_dir(params)


if __name__ == '__main__':

    model = model_2d_u_net(params)
    train(params)