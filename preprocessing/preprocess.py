import os
import argparse
import time

from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2


def preprocessing(image_name, n4=True, save_dir=None, circle_crop=False):
    '''
    Python implementation of the N4 bias field correction algorithm
    '''

    img_directory = os.path.dirname(image_name)
    img_basename = os.path.basename(image_name)

    # Load image data
    img = cv2.imread(image_name)

    # select green channel from the image (gives best blood consrast)
    img_data_green = img[:, :, 1].astype('float32')

    if n4:
        # Implementation of N4 bias field correction using simpleITK
        sitk_img = sitk.GetImageFromArray(img_data_green)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetNumberOfThreads(8)
        mask_img = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        final_img = sitk.GetArrayFromImage(corrector.Execute(sitk_img, mask_img))
        prefix = ''
    else:
        final_img = img_data_green
        prefix = '-non'

    if save_dir:
        plt.imsave(save_dir + '/' + img_basename.split('.')[0] + f'{prefix}-processed.png', final_img, cmap='gray')
    
    else:
        plt.imsave(img_directory + '/' + img_basename.split('.')[0] + f'{prefix}-processed.png', final_img, cmap='gray')



if __name__ == '__main__':
    preprocessing('/home/james/Downloads/DR-data/Data/DRIVE/test/images/13_test.tif', save_dir=None, n4=False)
