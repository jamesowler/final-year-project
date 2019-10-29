import os
from logging import log
import glob

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from params import params

def load_batch(img_names, segs_dir, xshape, yshape, n_channels):
    
    # create empty arrays with desired resampled size - for batch
    X = np.zeros((len(img_names), xshape, yshape, n_channels), dtype='float32')
    Y = np.zeros((len(img_names), xshape, yshape, 1), dtype='float32')

    # for n subj in the batch
    x = np.zeros((xshape, yshape), dtype='float32')
    y = np.zeros((xshape, yshape), dtype='float32')

    for n, img in enumerate(img_names):
        
        img_id = os.path.basename(img)[0:2]

        seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')
        log(1, seg_name)

        img_data = Image.open(img).convert('LA')
        seg_data = Image.open(seg_name).convert('LA')

        # reshape data:
        img_data = img_data.resize((xshape, yshape), resample=Image.BILINEAR)
        seg_data = seg_data.resize((xshape, yshape), resample=Image.BILINEAR)

        x[:, :] = np.array(img_data)[:, :, 0]
        # print(np.array(seg_data)[:, :, 0].shape)
        y[:, :] = np.array(seg_data)[:, :, 0]

        # y_flat = y.reshape(xshape * yshape)

        # convert to binary mask
        y[y > 0] = 1

        X[n, :, :, 0] = x[:, :]
        Y[n, :, :, 0] = y[:, :]

    return X, Y


if __name__ == '__main__':

    def testing():
        files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4\*')
        test_files = [i for i in files if 'test.png' in i]
        seg_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'

        X, Y = load_batch(test_files, seg_dir, params['image_size_x'], params['image_size_y'], 1)
        plt.imshow(Y[0, :, :, 0], cmap='gray')
        # plt.imshow(Y[0, :, :, 0], alpha=0.5)
        plt.show()


    testing()