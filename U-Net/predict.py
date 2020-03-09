import os
import glob
import shutil

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import keras.backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from params import params
from data_utils import test_batch, unblockshaped
from test import multi_test

import loss_funcs
from params import params
from model import model_2d_u_net_shallow, model_2d_u_net, model_2d_u_net_full
from test_model import MultiResUnet_shallow, MultiResUnet


def patch_inferance(filename, model_file, save_dir, model_type):

    '''
    Predicts output segmentation for each patch - joins patches together to create final segmentation map
    '''

    if model_type == 'unet':
        model = model_2d_u_net(params)

    if model_type == 'unet_shallow':
        model = model_2d_u_net_shallow(params)

    if model_type == 'MultiResUnet':
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)

    if model_type == 'MultiResUnet_shallow':
        model = MultiResUnet_shallow(params['patch_size'], params['patch_size'], 1)
    
    model.load_weights(model_file)

    y_full = cv2.imread(filename, 0)
    width, height = y_full.shape
    params['image_size_x'] = int(np.ceil(width/params['patch_size']))*params['patch_size']
    params['image_size_y'] = int(np.ceil(height/params['patch_size']))*params['patch_size']
    params['n_patches'] = int((params['image_size_x']*params['image_size_y'])/(params['patch_size']*params['patch_size']))

    y_batch = test_batch(filename, params['image_size_x'], params['image_size_y'], params['n_channels'], params['patch_size'])

    y_pred = model.predict_on_batch(y_batch)
    y_pred = np.reshape(y_pred, (params['n_patches'], params['patch_size'], params['patch_size'], 1))
    print(y_pred.shape)
    y_pred_full = unblockshaped(y_pred[:, :, :, 0], params['image_size_x'], params['image_size_y'])

    y_pred_full = y_pred_full[:width, :height]
    Y_pred_full_eval = np.copy(y_pred_full)

    y_pred_full[y_pred_full < 0.5] = 0
    y_pred_full[y_pred_full >= 0.5] = 1
    plt.imshow(y_pred_full, cmap='gray')

    #TODO 
    file_basename = os.path.basename(filename)
    outfile_default = os.path.join(save_dir, file_basename.split(".")[0] + '-seg.png')
    outfile_default_eval = os.path.join(save_dir, file_basename.split(".")[0] + '-seg-eval.png')

    # save final segmentation mask and segmenation probabilities
    plt.imsave(outfile_default, y_pred_full)
    plt.imsave(outfile_default_eval, Y_pred_full_eval)

def multi_predict(model_file, outputdir, prepro, mode='drive'):
    if mode == 'drive':
        files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-' + prepro + '\*')

    if mode == 'chase_db1':
        files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\CHASE_DB1\imgs-' + prepro + '\*')

    test_files = [i for i in files if 'test' in i]

    for i in test_files:
        patch_inferance(i, model_file, outputdir, params['model'])

        # clear session to try and avoid memory leakage when calling function in a loop
        K.clear_session()

def multi_epoch_pred(first_epoch=1, final_epoch=16, step=1, full=None, mode='chase_db1'):

    '''
    Output aucs and accs to results.txt file for multiple classiers (saved at different epochs in he training process)
    '''

    results_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-4'
    
    accs = []
    aucs = []
    epochs = []

    # obtain predicted segmentation masks
    for epoch in range(first_epoch, final_epoch + 1, step):
        epochs.append(epoch)

        if not full:
            multi_predict(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', results_dir, 'n4-clahe', mode=mode)

        if full:
            multi_predict_full(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', results_dir)

        # calc stats
        auc, acc = multi_test(results_dir, 'seg.png', 'seg-eval.png', epoch_num=epoch, mode=mode)
        accs.append(acc)
        aucs.append(auc)

    with open('./results.txt', 'w') as f:
        for i, j in zip(accs, aucs):
            f.write(str(j) + ',')
            f.write(str(i))
            f.write('\n')

def pred_prepro():
    '''
    Returns text file of results for each model type and preprocessing method
    '''

    for pre_pro in ['clahe', 'green', 'n4', 'n4-clahe']:

        results_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-4'
        models = ['unet_shallow', 'unet']
        # models = ['unet_shallow']

        model_folders = [r'C:\Users\James\Desktop\seg_test\processed_data_testing\\' + pre_pro + r'\Shallow_Unet', r'C:\Users\James\Desktop\seg_test\processed_data_testing\\' + pre_pro + r'\Unet']

        # model_folders = [r'C:\Users\James\Desktop\seg_test\processed_data_testing\\' + pre_pro + r'\Shallow_Unet']

        for model, folder in zip(models, model_folders):
            params['model'] = model
            accs = []
            aucs = []
            for epoch in range(1, 16):
                print(pre_pro, model, str(epoch))
                multi_predict(folder + r'\patch_model_' + str(epoch) + '.h5', results_dir, pre_pro)
                auc, acc = multi_test(results_dir, 'seg.png', 'seg-eval.png', epoch_num=epoch)
                accs.append(acc)
                aucs.append(auc)

            with open('./accs.txt', 'w') as f:
                for i in accs:
                    f.write(str(i))
                    f.write('\n')

            with open('./aucs.txt', 'w') as f:
                for i in aucs:
                    f.write(str(i))
                    f.write('\n')

            shutil.copy('./accs.txt', os.path.join(folder, 'accs.txt'))  
            shutil.copy('./aucs.txt', os.path.join(folder, 'aucs.txt'))

def pred_prepro_chase():
    # TODO - impliment chase_db1 inference 
    for pre_pro in ['clahe', 'green', 'n4', 'n4-clahe']:

        results_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-4'
        models = ['unet_shallow', 'unet']
        # models = ['unet_shallow']

        folder = r'C:\Users\James\Desktop\seg_test\processed_data_testing\CHASE_DB1\\' + pre_pro

        accs = []
        aucs = []
        for epoch in range(1, 16):
            print(pre_pro, str(epoch))
            multi_predict(folder + r'\patch_model_' + str(epoch) + '.h5', results_dir, pre_pro, mode='chase_db1')
            auc, acc = multi_test(results_dir, 'seg.png', 'seg-eval.png', epoch_num=epoch, mode='chase_db1')
            accs.append(acc)
            aucs.append(auc)

        with open('./accs.txt', 'w') as f:
            for i in accs:
                f.write(str(i))
                f.write('\n')

        with open('./aucs.txt', 'w') as f:
            for i in aucs:
                f.write(str(i))
                f.write('\n')

        shutil.copy('./accs.txt', os.path.join(folder, 'accs.txt'))  
        shutil.copy('./aucs.txt', os.path.join(folder, 'aucs.txt'))


if __name__ == '__main__':

    # multi_predict(r'C:\Users\James\Desktop\seg_test\processed_data_testing\CHASE_DB1\clahe\patch_model_2.h5', r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-4', 'n4-clahe', mode='chase_db1')
    # multi_test(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-4', 'seg', 'seg-eval', mode='chase_db1')
    multi_epoch_pred()
    # pred_prepro_chase()