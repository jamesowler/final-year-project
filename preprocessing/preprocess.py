import os
import argparse
import time
import glob

from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2


def preprocessing(image_name, n4=True, save_dir=None, circle_crop=False):
    '''
    Python wrapper for the N4 bias field correction algorithm
    '''

    img_directory = os.path.dirname(image_name)
    img_basename = os.path.basename(image_name)

    # Load image data
    img = cv2.imread(image_name)

    # select green channel from the image (gives best blood consrast)
    img_data_green = img[:, :, 1].astype('float32')

    if n4:
        # Implementation of N4 bias field correction using simpleITK
        print('Processing...')
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


def contrast_enhancement(image_name):
    img = cv2.imread(image_name, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    cv2.imwrite(image_name, cl1)

def extract_channel(image_name):
    img = cv2.imread(image_name)
    plt.imsave(image_name, img[:, :, 0], cmap='gray')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description= 'Preprocessing for retinal images')
    # parser.add_argument('file_path', help='Input file path of image file you want to process')
    # parser.add_argument('-n4', required=False, action='store_true', help='Use N4 bias field correction')
    # parser.add_argument('--output_dir', '-o', help='Path of directory to save the image to')
    # args = parser.parse_args()

    # preprocessing(args.file_path, save_dir=args.output_dir, n4=args.n4)

    extract_channel(r'C:\Users\James\Documents\Uni\Final Project\content\pre-processing-examples\colour - Copy.png')
