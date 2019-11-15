import os
import glob

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

from params import params
from data_utils import test_batch, unblockshaped


def patch_inferance(filename, model_file, save_dir):

    '''
    Predicts output segmentation for each patch - joins patches together to create final segmentation map
    '''

    model = load_model(model_file)
    y_batch = test_batch(filename, params['image_size_x'], params['image_size_y'],
params['n_channels'], params['patch_size'])
    # print(y_batch.shape)

    y_pred = model.predict_on_batch(y_batch)
    y_pred = np.reshape(y_pred, (params['n_patches'], params['patch_size'], params['patch_size'], 1))
    print(y_pred.shape)
    y_pred_full = unblockshaped(y_pred[:, :, :, 0], params['image_size_x'], params['image_size_y'])

    y_pred_full[y_pred_full < 0.5] = 0
    y_pred_full[y_pred_full > 0.5] = 1
    plt.imshow(y_pred_full, cmap='gray')
    # plt.show()

    #TODO 
    file_basename = os.path.basename(filename)
    outfile_default = os.path.join(save_dir, file_basename.split(".")[0] + '-seg.png')
    plt.imsave(outfile_default, y_pred_full)


def predict(params, filename, model_name, outfile=None):

    '''
    Predicts segmentation mask of retinal blood vessels given an image of the retina
    '''


    # img_data = Image.open(filename).convert('LA')
    # width, height = img_data.size
    # img_data = img_data.resize((params['image_size_x'], params['image_size_y']), resample=Image.BILINEAR)
    
    x = np.zeros((1, params['image_size_x'], params['image_size_y'], 1), dtype='float32')

    img_data_org = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    width, height = img_data_org.shape
    img_data = cv2.resize(img_data_org, (params['image_size_x'], params['image_size_y']), interpolation = cv2.INTER_LINEAR)


    # load and use model to predict vessels
    x[0, :, :, 0] = img_data
    print(x.shape)
    model = load_model(model_name)
    # print(model.summary())
    y_pred = model.predict(x)

    # reshape output back into image
    y_pred = np.reshape(y_pred, (params['image_size_x'], params['image_size_y']))
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred > 0.5] = 1

    y_pred_resized = cv2.resize(y_pred, (height, width), interpolation = cv2.INTER_NEAREST)

    if not outfile:
        outfile_default = filename.split('.')[0] + '-seg.png'
        outfile_default_seg_combined = filename.split('.')[0] + '-seg-overlay.png'
        plt.imsave(outfile_default, y_pred_resized)

        # Need to display the result of imshow on an axis
        fig, ax = plt.subplots()
        ax.imshow(img_data_org, cmap='gray')
        ax.imshow(y_pred_resized, alpha=0.15)
        ax.set_axis_off()
        plt.savefig(outfile_default_seg_combined, dpi=200)

    else:
        plt.imsave(outfile, y_pred_resized)


    return y_pred_resized


def multi_predict():
    files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-clahe\*')
    test_files = [i for i in files if 'test' in i]

    for i in test_files:
        patch_inferance(i, r'C:\Users\James\Projects\final-year-project\patch_model_2500.h5',r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-5')



if __name__ == '__main__':

    # predict(params, r'C:\Users\James\Projects\final-year-project\data\STARE\imgs\17.tif', '.\model.h5')

    multi_predict()

    # patch_inferance(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-2\01_test.png', r'C:\Users\James\Projects\final-year-project\patch_model_4000.h5')

