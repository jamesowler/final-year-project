import os
from logging import log
import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d

from params import params

def load_batch(img_names, segs_dir, xshape, yshape, n_channels, patch_size):
    
    # create empty arrays with desired resampled size - for batch
    X = np.zeros((len(img_names), patch_size, patch_size, n_channels), dtype='float32')
    Y = np.zeros((len(img_names), patch_size, patch_size, 1), dtype='float32')

    X_full = np.zeros((len(img_names), xshape, yshape, n_channels), dtype='float32')
    Y_full = np.zeros((len(img_names), xshape, yshape, 1), dtype='float32')

    # for n subj in the batch
    x = np.zeros((xshape, yshape), dtype='float32')
    y = np.zeros((xshape, yshape), dtype='float32')

    rand_ints = []

    # half of patch size
    half_patch = int(patch_size/2)

    for n, img in enumerate(img_names):
        
        img_id = os.path.basename(img)[0:2]

        seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')
        log(1, seg_name)

        img_data = Image.open(img).convert('LA')
        seg_data = Image.open(seg_name).convert('LA')

        # reshape data:
        img_data = img_data.resize((xshape, yshape), resample=Image.BILINEAR)
        seg_data = seg_data.resize((xshape, yshape), resample=Image.NEAREST)

        x[:, :] = np.array(img_data)[:, :, 0]

        # normalise data [0, 1]
        # x = (x - np.min(x))/np.ptp(x)

        y[:, :] = np.array(seg_data)[:, :, 0]
        # convert 255 to 1
        y[y == 255] = 1

        # extract random patch
        rand_int = np.random.randint(half_patch, high=(xshape - half_patch), size=2)
        rand_ints.append(rand_int)
        x_patch = x[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]
        y_patch = y[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]

        X[n, :, :, 0] = x_patch[:, :]
        Y[n, :, :, 0] = y_patch[:, :]

        # full image to display where patch is in image
        X_full[n, :, :, 0] = x[:, :]
        Y_full[n, :, :, 0] = y[:, :]

    return X, Y #, X_full, Y_full, rand_ints


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def test_batch(img_name, xshape, yshape, n_channels, patch_size):

    n_patches = int((xshape*yshape)/(patch_size*patch_size))
    img_data = Image.open(img_name).convert('LA')
    img_data = img_data.resize((xshape, yshape), resample=Image.BILINEAR)
    
    # normalise data [0, 1]
    img_data = np.array(img_data)
    img_data = (img_data - np.min(img_data))/np.ptp(img_data)

    image = np.zeros((xshape, yshape), dtype='float32')
    patches_final = np.zeros((n_patches, patch_size, patch_size, n_channels), dtype='float32')


    image[:, :] = np.array(img_data)[:, :, 0]

    patches = blockshaped(image, patch_size, patch_size)

    patches_final[:, :, :, 0] = patches

    return patches_final


if __name__ == '__main__':

    def testing():
        files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-clahe\*')
        test_files = [i for i in files if 'test.png' in i]
        seg_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'
        
        patch_size = 64
        half_patch = int(patch_size/2)

        X, Y, X_full, Y_full, rand_ints = load_batch(test_files, seg_dir, params['image_size_x'], params['image_size_y'], 1, params['patch_size'])
        plt.figure(1)
        plt.imshow(X[0, :, :, 0], cmap='gray')
        plt.imshow(Y[0, :, :, 0], alpha=0.2)

        plt.figure(2)
        X_full_patch = np.zeros(X_full.shape)
        X_full_patch[0, (rand_ints[0][0] - half_patch):(rand_ints[0][0] + half_patch), (rand_ints[0][1] - half_patch):(rand_ints[0][1] + half_patch), 0] += 255
        print(np.max(X_full_patch))
        plt.imshow(X_full_patch[0, :, :, 0])
        plt.imshow(X_full[0, :, :, 0], cmap='gray', alpha=0.8)


        plt.show()

    def test_inference_loading():
        test_batch(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4\01_test.png', params['image_size_x'], params['image_size_y'], 1, params['patch_size'])

    # testing()
    test_inference_loading()