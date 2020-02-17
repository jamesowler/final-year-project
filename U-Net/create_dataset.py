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

def create_image_img_folder(ouput_dir, patch_size, n_patches):
    '''
    Creates dataset of image patches
    '''

    # segs
    segs_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'

    # green channel
    files_1 = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-green' + '\*')
    training_files_1 = [i for i in files_1 if 'training.png' in i]
    os.mkdir(ouput_dir + 'green')
    img_dir_1 = os.path.join(ouput_dir + 'green', 'imgs')
    seg_dir_1 = os.path.join(ouput_dir + 'green', 'segs')
    os.mkdir(img_dir_1)
    os.mkdir(seg_dir_1)

    # n4
    files_2 = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4' + '\*')
    training_files_2 = [i for i in files_2 if 'training.png' in i]
    os.mkdir(ouput_dir + 'n4')
    img_dir_2 = os.path.join(ouput_dir + 'n4', 'imgs')
    seg_dir_2 = os.path.join(ouput_dir + 'n4', 'segs')
    os.mkdir(img_dir_2)
    os.mkdir(seg_dir_2)

    # n4_plus_clahe
    files_3 = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4-clahe' + '\*')
    training_files_3 = [i for i in files_3 if 'training.png' in i]
    os.mkdir(ouput_dir + 'n4-clahe')
    img_dir_3 = os.path.join(ouput_dir + 'n4-clahe', 'imgs')
    seg_dir_3 = os.path.join(ouput_dir + 'n4-clahe', 'segs')
    os.mkdir(img_dir_3)
    os.mkdir(seg_dir_3)

    # clahe
    files_4 = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-clahe' + '\*')
    training_files_4 = [i for i in files_4 if 'training.png' in i]
    os.mkdir(ouput_dir + 'clahe')
    img_dir_4 = os.path.join(ouput_dir + 'clahe', 'imgs')
    seg_dir_4 = os.path.join(ouput_dir + 'clahe', 'segs')
    os.mkdir(img_dir_4)
    os.mkdir(seg_dir_4)

    half_patch = int(patch_size/2)

    n = 0

    for img_1, img_2, img_3, img_4 in zip(training_files_1, training_files_2, training_files_3, training_files_4):

        n += 1
        print(f'\r {n}/20 extracting patches')

        x_1 = np.zeros((576, 576), dtype='float32')
        x_2 = np.zeros((576, 576), dtype='float32')
        x_3 = np.zeros((576, 576), dtype='float32')
        x_4 = np.zeros((576, 576), dtype='float32')

        y = np.zeros((576, 576), dtype='float32')

        # load in image and segmentation
        img_id = os.path.basename(img_1)[0:2]
        seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')
        seg_data = Image.open(seg_name).convert('LA')
        seg_data = seg_data.resize((576, 576), resample=Image.NEAREST)
        # process seg data
        y[:, :] = np.array(seg_data)[:, :, 0]
        # convert 255 to 1
        y[y == 255] = 1

        # 1
        img_data = Image.open(img_1).convert('LA')
        # reshape data:
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        x_1[:, :] = np.array(img_data)[:, :, 0]
        # normalise data [0, 1]
        x_1 = (x_1 - np.min(x_1))/np.ptp(x_1)

        # 2
        img_data = Image.open(img_2).convert('LA')
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        x_2[:, :] = np.array(img_data)[:, :, 0]
        x_2 = (x_2 - np.min(x_2))/np.ptp(x_2)

        # 3
        img_data = Image.open(img_3).convert('LA')
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        x_3[:, :] = np.array(img_data)[:, :, 0]
        x_3 = (x_3 - np.min(x_3))/np.ptp(x_3)

        # 4
        img_data = Image.open(img_4).convert('LA')
        img_data = img_data.resize((576, 576), resample=Image.BILINEAR)
        x_4[:, :] = np.array(img_data)[:, :, 0]
        x_4 = (x_4 - np.min(x_4))/np.ptp(x_4)

        for n_patch in range(n_patches):
            # extract random patch
            rand_int = np.random.randint(half_patch, high=(576 - half_patch), size=2)
            
            # for each type of preprocessed image
            x_patch_1 = x_1[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]
            x_patch_2 = x_2[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]
            x_patch_3 = x_3[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]
            x_patch_4 = x_4[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]

            y_patch = y[(rand_int[0] - half_patch):(rand_int[0] + half_patch), (rand_int[1] - half_patch):(rand_int[1] + half_patch)]   

            # save patches 1
            img_save_name = os.path.join(img_dir_1, f'{img_id}-{n_patch}-img.png')
            seg_save_name = os.path.join(seg_dir_1, f'{img_id}-{n_patch}-seg.png')
            plt.imsave(img_save_name, x_patch_1, cmap='gray')
            plt.imsave(seg_save_name, y_patch, cmap='gray')
            # save patches 2
            img_save_name = os.path.join(img_dir_2, f'{img_id}-{n_patch}-img.png')
            seg_save_name = os.path.join(seg_dir_2, f'{img_id}-{n_patch}-seg.png')
            plt.imsave(img_save_name, x_patch_2, cmap='gray')
            plt.imsave(seg_save_name, y_patch, cmap='gray')
            # save patches 3
            img_save_name = os.path.join(img_dir_3, f'{img_id}-{n_patch}-img.png')
            seg_save_name = os.path.join(seg_dir_3, f'{img_id}-{n_patch}-seg.png')
            plt.imsave(img_save_name, x_patch_3, cmap='gray')
            plt.imsave(seg_save_name, y_patch, cmap='gray')
            # save patches 4
            img_save_name = os.path.join(img_dir_4, f'{img_id}-{n_patch}-img.png')
            seg_save_name = os.path.join(seg_dir_4, f'{img_id}-{n_patch}-seg.png')
            plt.imsave(img_save_name, x_patch_4, cmap='gray')
            plt.imsave(seg_save_name, y_patch, cmap='gray')

if __name__ == '__main__':
    create_image_img_folder(r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive-', 64, 5000)