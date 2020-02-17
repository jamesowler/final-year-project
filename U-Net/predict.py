import os
import glob

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
    height, width = y_full.shape
    
    y_batch = test_batch(filename, params['image_size_x'], params['image_size_y'], params['n_channels'], params['patch_size'])

    y_pred = model.predict_on_batch(y_batch)
    y_pred = np.reshape(y_pred, (params['n_patches'], params['patch_size'], params['patch_size'], 1))
    print(y_pred.shape)
    y_pred_full = unblockshaped(y_pred[:, :, :, 0], params['image_size_x'], params['image_size_y'])

    Y_pred_full_eval = np.copy(y_pred_full)

    y_pred_full[y_pred_full < 0.5] = 0
    y_pred_full[y_pred_full >= 0.5] = 1
    plt.imshow(y_pred_full, cmap='gray')
    # plt.show()

    #TODO 
    file_basename = os.path.basename(filename)
    outfile_default = os.path.join(save_dir, file_basename.split(".")[0] + '-seg.png')
    outfile_default_eval = os.path.join(save_dir, file_basename.split(".")[0] + '-seg-eval.png')
    
    y_pred_full = cv2.resize(y_pred_full, (width, height), interpolation=cv2.INTER_NEAREST)
    Y_pred_full_eval = cv2.resize(Y_pred_full_eval, (width, height), interpolation=cv2.INTER_NEAREST)

    # save final segmentation mask and segmenation probabilities
    plt.imsave(outfile_default, y_pred_full)
    plt.imsave(outfile_default_eval, Y_pred_full_eval)



def predict(params, filename, model_name, save_dir):

    '''
    Predicts segmentation mask of retinal blood vessels given an image of the retina
    '''

    # img_data = Image.open(filename).convert('LA')
    # width, height = img_data.size
    # img_data = img_data.resize((params['image_size_x'], params['image_size_y']), resample=Image.BILINEAR)
    
    x = np.zeros((1, params['image_size_x'], params['image_size_y'], 1))
    img_data_org = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    width, height = img_data_org.shape
    img_data = cv2.resize(img_data_org, (params['image_size_x'], params['image_size_y']), interpolation = cv2.INTER_LINEAR)


    # load and use model to predict vessels
    x[0, :, :, 0] = img_data
    x = (x - np.min(x))/np.ptp(x)

    print(x.shape)
    model = MultiResUnet(params['image_size_x'], params['image_size_y'], 1)
    model.load_weights(model_name)

    # predict output 
    y_pred = model.predict(x)

    # reshape output back into image
    y_pred = np.reshape(y_pred, (params['image_size_x'], params['image_size_y']))
    # segmentation probability map
    Y_pred_full_eval = np.copy(y_pred) 
    # final tresholded segmentaiton
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    
    y_pred_resized = cv2.resize(y_pred, (height, width), interpolation = cv2.INTER_NEAREST)
    Y_pred_full_eval = cv2.resize(Y_pred_full_eval, (height, width), interpolation = cv2.INTER_LINEAR)

    file_basename = os.path.basename(filename)
    outfile_default = os.path.join(save_dir, file_basename.split(".")[0] + '-seg.png')
    outfile_default_eval = os.path.join(save_dir, file_basename.split(".")[0] + '-seg-eval.png')

    # save final segmentation mask and segmenation probabilities
    plt.imsave(outfile_default, y_pred_resized)
    plt.imsave(outfile_default_eval, Y_pred_full_eval)

    # # Need to display the result of imshow on an axis
    # fig, ax = plt.subplots()
    # ax.imshow(img_data_org, cmap='gray')
    # ax.imshow(y_pred_resized, alpha=0.15)
    # ax.set_axis_off()
    # plt.savefig(outfile_default_seg_combined, dpi=200)

    return y_pred_resized


def multi_predict(model_file, outputdir):
    files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-' + params['preprocessing'] + '\*')
    test_files = [i for i in files if 'test.png' in i]

    for i in test_files:
        patch_inferance(i, model_file, outputdir, params['model'])

        # clear session to try and avoid memory leakage when calling function in a loop
        K.clear_session()

def multi_predict_full(model_file, outputdir):
    files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-' + params['preprocessing'][1:] + '\*')
    test_files = [i for i in files if 'test.png' in i]

    for i in test_files:
        predict(params, i, model_file, outputdir)

        # clear session to try and avoid memory leakage when calling function in a loop
        K.clear_session()

if __name__ == '__main__':

    def multi_epoch_pred(first_epoch=1, final_epoch=15, step=1, full=None):

        '''
        Output aucs and accs to results.txt file for multiple classiers (saved at different epochs in he training process)
        '''

        results_dir = r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-2'
        
        accs = []
        aucs = []
        epochs = []

        # obtain predicted segmentation masks
        for epoch in range(first_epoch, final_epoch + 1, step):
            epochs.append(epoch)

            if not full:
                multi_predict(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', results_dir)

            if full:
                multi_predict_full(r'C:\Users\James\Projects\final-year-project\patch_model_' + str(epoch) + '.h5', results_dir)

            # calc stats
            auc, acc = multi_test(results_dir, 'seg.png', 'seg-eval.png', epoch_num=epoch)
            accs.append(acc)
            aucs.append(auc)

        with open('./results.txt', 'w') as f:
            for i, j, k in zip(accs, aucs, epochs):
                f.write(str(k) + '- ')
                f.write('AUC: ' + str(j) + ' ')
                f.write('Accuracy: ' + str(i))
                f.write('\n')

    
    multi_epoch_pred(first_epoch=1, final_epoch=1, step=1, full=None)
    
    # multi_epoch_pred(first_epoch=50, final_epoch=250, step=50, full=True)

    # patch_inferance(r'C:\Users\James\Desktop\seg_test\0a74c92e287c-preprocessed.png',  r'C:\Users\James\Projects\final-year-project\U-Net\models\best\patch_model_64_16_shallow_unet.h5', r'C:\Users\James\Desktop\seg_test', 'unet_shallow')