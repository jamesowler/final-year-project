import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_dataset(input_dir, ouput_dir):
    '''
    Creates dataset of image patches - 96x96px
    '''

    img_dir = os.path.join(ouput_dir, 'imgs')
    seg_dir = os.path.join(ouput_dir, 'segs')

    if not os.path.exists(ouput_dir):
        os.mkdir(ouput_dir)
        os.mkdir(img_dir)
        os.mkdir(seg_dir)

    files = glob.glob(input_dir + '\*')
    training_files = [i for i in files if 'training.png' in i]
    segs_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'


    for n, img in enumerate(training_files):

        x = np.zeros((576, 576), dtype='float32')
        y = np.zeros((576, 576), dtype='float32')

        # load in image and segmentation
        img_id = os.path.basename(img)[0:2]
        seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')
        img_data = Image.open(img).convert('LA')
        seg_data = Image.open(seg_name).convert('LA')
        
        # reshape data:
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        seg_data = seg_data.resize((576, 576), resample=Image.NEAREST)
        x[:, :] = np.array(img_data)[:, :, 0]

        # normalise data [0, 1]
        x = (x - np.min(x))/np.ptp(x)

        y[:, :] = np.array(seg_data)[:, :, 0]

        # convert 255 to 1 - for segmentation 
        y[y == 255] = 1

        for n in range(int(576/48)-1):
            for j in range(int(576/48)-1):
                # extract random patch
                x_patch = x[0 + 48*n:96 + 48*n, 0 + 48*j:96 + 48*j]
                y_patch = y[0 + 48*n:96 + 48*n, 0 + 48*j:96 + 48*j]

                # save patches
                img_save_name = os.path.join(img_dir, f'{img_id}-{n}{j}-img.png')
                seg_save_name = os.path.join(seg_dir, f'{img_id}-{n}{j}-seg.png')
                plt.imsave(img_save_name, x_patch, cmap='gray')
                plt.imsave(seg_save_name, y_patch, cmap='gray')

def create_image_img_folder(input_dir, ouput_dir, patch_size, n_patches):
    '''
    Creates dataset of image patches
    '''
    files = glob.glob(input_dir + '\*')
    training_files = [i for i in files if 'training.png' in i]
    segs_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'

    img_dir = os.path.join(ouput_dir, 'imgs')
    seg_dir = os.path.join(ouput_dir, 'segs')

    half_patch = int(patch_size/2)

    if not os.path.exists(ouput_dir):
        os.mkdir(ouput_dir)
        os.mkdir(img_dir)
        os.mkdir(seg_dir)

    for n, img in enumerate(training_files):

        x = np.zeros((576, 576), dtype='float32')
        y = np.zeros((576, 576), dtype='float32')

        # load in image and segmentation
        img_id = os.path.basename(img)[0:2]
        seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')
        img_data = Image.open(img).convert('LA')
        seg_data = Image.open(seg_name).convert('LA')
        
        # reshape data:
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        seg_data = seg_data.resize((576, 576), resample=Image.NEAREST)
        x[:, :] = np.array(img_data)[:, :, 0]

        # normalise data [0, 1]
        x = (x - np.min(x))/np.ptp(x)

        y[:, :] = np.array(seg_data)[:, :, 0]

        # convert 255 to 1
        y[y == 255] = 1

        for n_patch in range(n_patches):

            # extract random patch
            rand_int = np.random.randint(half_patch, high=(576 - half_patch), size=2)
            x_patch = x[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]
            y_patch = y[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]   

            # save patches
            img_save_name = os.path.join(img_dir, f'{img_id}-{n_patch}-img.png')
            seg_save_name = os.path.join(seg_dir, f'{img_id}-{n_patch}-seg.png')
            plt.imsave(img_save_name, x_patch, cmap='gray')
            plt.imsave(seg_save_name, y_patch, cmap='gray')


if __name__ == '__main__':
    # create_dataset(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-clahe-n4', r'C:\Users\James\Projects\final-year-project\data\drive_patches\clahe-n4-96' )
    create_image_img_folder(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-clahe', r'C:\Users\James\Projects\final-year-project\data\drive_patches\clahe-96', 96, 6000)
