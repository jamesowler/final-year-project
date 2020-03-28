import os
import glob
import shutil
import json

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
from model import model_2d_u_net_shallow, model_2d_u_net, model_2d_u_net_full, fully_conv_expri_network
from test_model import MultiResUnet_shallow, MultiResUnet


def patch_inferance(filename, model_file, save_dir, model_type):

    '''
    Predicts output segmentation for each patch - joins patches together to create final segmentation map
    '''

    if model_type == 'unet':
        model = model_2d_u_net(params)

    if model_type == 'unet_shallow':
        model = model_2d_u_net_shallow(params)
    
    if model_type == 'multi_res_testing':
        model = fully_conv_expri_network(params)

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

def multi_epoch_pred(first_epoch=1, final_epoch=15, step=1, full=None, mode='drive'):

    '''
    Output aucs and accs to results.txt file for multiple classiers (saved at different epochs in he training process)
    '''

    output_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-3'
    multi_epoch_results = {}

    # obtain predicted segmentation masks
    for epoch in range(first_epoch, final_epoch + 1, step):

        if not full:
            multi_predict(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', output_dir, 'n4-clahe', mode=mode)

        if full:
            multi_predict_full(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', output_dir)

        # calc stats
        aucs, accs, sens, specs = multi_test(output_dir, 'seg.png', 'seg-eval.png', mode=mode)

        results = {}
        results['accs'] = accs
        results['accs_mean'] = round(np.mean(accs), 4)
        results['aucs'] = aucs
        results['aucs_mean'] = round(np.mean(aucs), 4)
        results['sens'] = sens
        results['sens_mean'] = round(np.mean(sens), 4)
        results['specs'] = specs
        results['specs_mean'] = round(np.mean(specs), 4)

        multi_epoch_results[f'Epoch {str(epoch)}'] = results

    with open('multi_epoch_results.json', 'w') as json_file:
        json.dump(multi_epoch_results, json_file, indent=4, sort_keys=True)   

def pred_prepro_drive():
    '''
    Returns text file of results for each model type and preprocessing method
    '''

    for pre_pro in ['clahe', 'green', 'n4', 'n4-clahe']:

        folder = r'C:\Users\James\Desktop\seg_test\processed_data_testing\DRIVE-128\\' + pre_pro
        output_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-3'
        model = 'unet'
        model_folders = r'C:\Users\James\Desktop\seg_test\processed_data_testing\\' + pre_pro + r'\Unet'

        params['model'] = model

        multi_epoch_results = {}

        for epoch in range(1, 3):
            print(pre_pro, model, str(epoch))
            multi_predict(folder + r'\patch_model_' + str(epoch) + '.h5', output_dir, pre_pro)
            aucs, accs, sens, specs = multi_test(output_dir, 'seg.png', 'seg-eval.png', mode='drive')
            results = {}
            results['accs'] = accs
            results['accs_mean'] = round(np.mean(accs), 4)
            results['aucs'] = aucs
            results['aucs_mean'] = round(np.mean(aucs), 4)
            results['sens'] = sens
            results['sens_mean'] = round(np.mean(sens), 4)
            results['specs'] = specs
            results['specs_mean'] = round(np.mean(specs), 4)

            multi_epoch_results[f'Epoch {str(epoch)}'] = results
        
        
        with open('multi_epoch_results.json', 'w') as json_file:
            json.dump(multi_epoch_results, json_file, indent=4, sort_keys=True)        

        shutil.move('./multi_epoch_results.json', os.path.join(folder, 'multi_epoch_results.json'))

def pred_prepro_chase():

    for pre_pro in ['clahe', 'green', 'n4', 'n4-clahe']:

        multi_epoch_results = {}
        output_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\CHASE_DB1'
        folder = r'C:\Users\James\Desktop\seg_test\processed_data_testing\CHASE_DB1-128\\' + pre_pro
        
        for epoch in range(1,4):

            multi_predict(folder + r'\patch_model_' + str(epoch) + '.h5', output_dir, pre_pro, mode='chase_db1')
            aucs, accs, sens, specs = multi_test(output_dir, 'seg.png', 'seg-eval.png', mode='chase_db1')

            results = {}
            results['accs'] = accs
            results['accs_mean'] = round(np.mean(accs), 4)
            results['aucs'] = aucs
            results['aucs_mean'] = round(np.mean(aucs), 4)
            results['sens'] = sens
            results['sens_mean'] = round(np.mean(sens), 4)
            results['specs'] = specs
            results['specs_mean'] = round(np.mean(specs), 4)

            multi_epoch_results[f'Epoch {str(epoch)}'] = results

            
        with open('multi_epoch_results.json', 'w') as json_file:
            json.dump(multi_epoch_results, json_file, indent=4, sort_keys=True)


        shutil.move('./multi_epoch_results.json', os.path.join(folder, 'multi_epoch_results.json'))

def pred_prepro_stare(pre_pro='clahe'):
    models_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing\STARE-128'
    imgs_dir = r'C:\Users\James\Projects\final-year-project\data\STARE'
    epoch_num = '3'
    image_nums = [str(x) + '.png' for x in range(1, 21)]

    output_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\STARE'

    for img in image_nums:
        
        img_model_path = os.path.join(models_dir, img)
        input_img = imgs_dir + f'\\imgs-{pre_pro}\\{img}'
        mask_img = imgs_dir + f'\\masks\\{img}'
        fov_img_name = img.split('.')[0] + '.gif'
        fov_img = imgs_dir + f'\\fov_masks\\{fov_img_name}'
        img_model_path_prepro = os.path.join(img_model_path, pre_pro)
        img_model_path_prepro_model = os.path.join(img_model_path_prepro, f'patch_model_{epoch_num}.h5')

        patch_inferance(input_img, img_model_path_prepro_model, output_dir, 'unet')
        K.clear_session()

    aucs, accs, sens, specs = multi_test(output_dir, 'seg.png', 'seg-eval.png', mode='stare')

    results = {}
    results['accs'] = accs
    results['accs_mean'] = round(np.mean(accs), 4)
    results['aucs'] = aucs
    results['aucs_mean'] = round(np.mean(aucs), 4)
    results['sens'] = sens
    results['sens_mean'] = round(np.mean(sens), 4)
    results['specs'] = specs
    results['specs_mean'] = round(np.mean(specs), 4)
    
    with open('results.txt', 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

    results_folder = os.path.join(models_dir, pre_pro + '_results')

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    
    shutil.move('results.txt', os.path.join(results_folder, 'results.txt'))


if __name__ == '__main__':

    # for i in ['clahe', 'green', 'n4', 'n4-clahe']:
    #     pred_prepro_stare(pre_pro=i)

    # pred_prepro_chase()
    # pred_prepro_drive()
    multi_epoch_pred(first_epoch=1, final_epoch=2)