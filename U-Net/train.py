import os
import time
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam

from params import params
from data_utils import load_batch_patch_training, guass_noise
from model import model_2d_u_net, model_2d_u_net_shallow, model_2d_u_net_full
from test_model import MultiResUnet_shallow, MultiResUnet
import loss_funcs


def train_from_data_dir(params, model):
    '''
    Trains model using dataset of patches

    For pre-processing testing - define seeds for data shuffling and augmentation

    '''

    imgs_dir = os.path.join(params['data_dir'], 'imgs')
    segs_dir = os.path.join(params['data_dir'], 'segs')

    # list images in dataset directory
    imgs = os.listdir(imgs_dir)

    # train - validation split
    imgs_rand = imgs

    # SEED
    random.Random(21).shuffle(imgs_rand)
    val_index_split = int(len(imgs_rand)*params['val_proportion'])
    train_imgs = imgs_rand[val_index_split:]
    val_imgs = imgs_rand[:val_index_split]

    n_batches_train = np.ceil(len(train_imgs)/params['batch_size']).astype('int')
    n_batches_val = np.ceil(len(val_imgs)/params['batch_size']).astype('int')

    losses = []
    accuracies = []
    val_accs = []
    val_loss = []
    
    for e in range(1, params['n_epochs'] + 1):

        print('Epoch: ', e)

        # randomize order of training images
        train_imgs_rand = train_imgs
        
        # SEED
        random.Random(e).shuffle(train_imgs_rand)

        val_freq = int(n_batches_train/2)

        epoch_loss = []
        epoch_acc = []

        for batch_num, b in enumerate(range(n_batches_train)):
            if (b+1)*params['batch_size'] < len(train_imgs_rand):
                batch_train_imgs = train_imgs_rand[b*params['batch_size']:(b+1)*params['batch_size']]
            else:
                batch_train_imgs = train_imgs_rand[b*params['batch_size']:]

            X, Y = load_batch_patch_training(batch_train_imgs, os.path.join(params['data_dir'], 'imgs'), os.path.join(params['data_dir'], 'segs'), params['patch_size'])
            
            # SEED
            seed = e + batch_num

            # preprocessing_function=guass_noise,
            data_gen_args_img = dict(rotation_range=5.,
                                fill_mode='constant',
                                horizontal_flip=True,
                                vertical_flip=True,
                                cval=0)
            data_gen_args_seg = dict(rotation_range=5.,
                                fill_mode='constant',
                                horizontal_flip=True,
                                vertical_flip=True,
                                cval=0)

            image_datagen = ImageDataGenerator(**data_gen_args_img)
            mask_datagen = ImageDataGenerator(**data_gen_args_seg)

            image_datagen.fit(X, augment=True, seed=seed)
            mask_datagen.fit(Y, augment=True, seed=seed)

            image_generator = image_datagen.flow(X, seed=seed)
            mask_generator = mask_datagen.flow(Y, seed=seed)
            train_generator = zip(image_generator, mask_generator)

            for X_aug, Y_aug in train_generator:
                Y_out = Y_aug.reshape(len(batch_train_imgs), params['patch_size'] * params['patch_size'], 1)
                history = model.train_on_batch(X_aug, Y_out)
                losses.append(history[0])
                accuracies.append(history[1])
                break

            # Custom training progress indicator 
            end = '' if batch_num < n_batches_train - 1 else '\n'
            print("\r{}/{} - ".format(batch_num + 1, n_batches_train) + "{:.4f}".format(history[0]) + ', ' + "{:.4f}".format(history[1]), end = end)
            epoch_loss.append(history[0])
            epoch_acc.append(history[1])
        
        # Calc validation metrics
        eval_losses = []
        eval_accuracies = []
        for b in range(n_batches_val):
            if (b+1)*params['batch_size'] < len(val_imgs):
                batch_val_imgs = val_imgs[b*params['batch_size']:(b+1)*params['batch_size']]
            else:
                batch_val_imgs = val_imgs[b*params['batch_size']:]
            # load data
            X, Y = load_batch_patch_training(batch_val_imgs, os.path.join(params['data_dir'], 'imgs'), os.path.join(params['data_dir'], 'segs'),params['patch_size'])
            Y_out = Y.reshape(len(batch_val_imgs), params['patch_size'] * params['patch_size'], 1)
            # evaluate model on validation data
            eval_score = model.evaluate(X, Y_out, verbose=0)
            eval_losses.append(eval_score[0])
            eval_accuracies.append(eval_score[1])

        print('validation loss: {:.4f}'.format(np.mean(eval_losses)))
        print('validation accuracy: {:.4f}'.format(np.mean(eval_accuracies)))
        val_loss.append(np.mean(eval_losses))
        val_accs.append(np.mean(eval_accuracies))
        
        if e % 1 == 0:
            model.save_weights(f'./patch_model_{str(e)}.h5')
        
        print('Average loss: {:.4f} \nAverage accuracy: {:.4f} \n'.format(np.mean(epoch_loss), np.mean(epoch_acc)))

    # save metrics into text files
    with open('./losses.txt', 'w') as f:
        for i in losses:
            f.write(str(i) + '\n')

    with open('./accuries.txt', 'w') as f:
        for i in accuracies:
            f.write(str(i) + '\n')

    with open('./val_losses.txt', 'w') as f:
        for i in val_loss:
            f.write(str(i) + '\n')

    with open('./val_accuries.txt', 'w') as f:
        for i in val_accs:
            f.write(str(i) + '\n')


def plot_losses():
    with open('losses.txt') as f:
        losses = [x.strip() for x in f]
    
    plt.scatter(losses, range(len(losses)))
    plt.show()

##### Training functions

def train_patch(weights=None, shallow=None):
    if shallow:
        model = model_2d_u_net_shallow(params)
        if weights:
            model.load_weights(weights)
        train_from_data_dir(params, model)
    else:
        model = model_2d_u_net(params)
        if weights:
            model.load_weights(weights)
        train_from_data_dir(params, model)

def train_full(weights=None):
    model = model_2d_u_net_full(params)
    if weights:
        model.load_weights(weights)
    train(params, model)

def train_multi_res_unet_shallow(full=None, weights=None):
    '''
    Train shallow MultiResUnet model

    Full = None : trains model using patches (size defined in params script)

    Full = True : trains model using full resolution images
    '''
    if not full:
        model = MultiResUnet_shallow(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

        if weights:
            model.load_weights(weights)

        train_from_data_dir(params, model)

    if full:
        model = MultiResUnet_shallow(params['image_size_x'], params['image_size_y'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

        if weights:
            model.load_weights(weights)

        train(params, model)

def train_multi_res_unet(full=None, weights=None):
    '''
    Train MultiResUnet model

    Full = None : trains model using patches (size defined in params script)

    Full = True : trains model using full resolution images
    '''
    if not full:
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

        if weights:
            model.load_weights(weights)

        train_from_data_dir(params, model)

    if full:
        model = MultiResUnet(params['image_size_x'], params['image_size_y'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))

        if weights:
            model.load_weights(weights)

        train(params, model)


if __name__ == '__main__':

    if params['model'] == 'MultiResUnet':
        train_multi_res_unet(weights=params['weights'])

    if params['model'] == 'MultiResUnet_full':
        train_multi_res_unet(full=True, weights=params['weights'])
    
    if params['model'] == 'unet_full':
        train_full(weights=params['weights'])
    
    if params['model'] == 'unet':
        train_patch(weights=params['weights'])

    if params['model'] == 'unet_shallow':
        train_patch(shallow=True, weights=params['weights'])

    if params['model'] == 'MultiResUnet_shallow':
        train_multi_res_unet_shallow(weights=params['weights'])