import os
import shutil
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

def create_image_img_folder(ouput_dir, patch_size, n_patches, mode='drive'):
    '''
    Creates dataset of image patches
    '''

    # segs
    if mode == 'drive':
        segs_dir = r'C:\Users\James\Projects\final-year-project\data\DRIVE\masks'
        data = 'DRIVE'

    if mode == 'chase_db1':
        segs_dir = r'C:\Users\James\Projects\final-year-project\data\CHASE_DB1\masks'
        data = 'CHASE_DB1'

    if mode == 'stare':
        segs_dir = r'C:\Users\James\Projects\final-year-project\data\STARE\masks'
        data = 'STARE'

    # green channel
    files_1 = glob.glob(f'C:\\Users\\James\\Projects\\final-year-project\\data\\{data}\\imgs-green' + '\*')
    if not mode == 'stare':
        training_files_1 = files_1 # [i for i in files_1 if 'training.png' in i]
    else:
        training_files_1 = files_1
    os.mkdir(ouput_dir + 'green')
    img_dir_1 = os.path.join(ouput_dir + 'green', 'imgs')
    seg_dir_1 = os.path.join(ouput_dir + 'green', 'segs')
    os.mkdir(img_dir_1)
    os.mkdir(seg_dir_1)

    # n4
    files_2 = glob.glob(f'C:\\Users\\James\\Projects\\final-year-project\\data\\{data}\\imgs-n4' + '\*')
    if not mode == 'stare':
        training_files_2 = files_2 # [i for i in files_2 if 'training.png' in i]
    else:
        training_files_2 = files_2
    os.mkdir(ouput_dir + 'n4')
    img_dir_2 = os.path.join(ouput_dir + 'n4', 'imgs')
    seg_dir_2 = os.path.join(ouput_dir + 'n4', 'segs')
    os.mkdir(img_dir_2)
    os.mkdir(seg_dir_2)

    # n4_plus_clahe
    files_3 = glob.glob(f'C:\\Users\\James\\Projects\\final-year-project\\data\\{data}\\imgs-n4-clahe' + '\*')
    if not mode == 'stare':
        training_files_3 = files_3 # [i for i in files_3 if 'training.png' in i]
    else:
        training_files_3 = files_3
    os.mkdir(ouput_dir + 'n4-clahe')
    img_dir_3 = os.path.join(ouput_dir + 'n4-clahe', 'imgs')
    seg_dir_3 = os.path.join(ouput_dir + 'n4-clahe', 'segs')
    os.mkdir(img_dir_3)
    os.mkdir(seg_dir_3)

    # clahe
    files_4 = glob.glob(f'C:\\Users\\James\\Projects\\final-year-project\\data\\{data}\\imgs-clahe' + '\*')
    if not mode == 'stare':
        training_files_4 = files_4 # [i for i in files_4 if 'training.png' in i]
    else:
        training_files_4 = files_4
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

        # load in image and segmentation
        if mode == 'drive':
            img_id = os.path.basename(img_1)[0:2]
            seg_name = os.path.join(segs_dir, img_id + '_manual1.gif')

        if mode == 'chase_db1':
            img_id = os.path.basename(img_1)[0:3]
            seg_name = os.path.join(segs_dir, f'Image_{img_id}_1stHO.png')
        
        if mode == 'stare':
            img_id = os.path.basename(img_1)
            seg_name = os.path.join(segs_dir, img_id)

        seg_data = Image.open(seg_name).convert('LA')

        # process seg data
        y = np.array(seg_data)[:, :, 0]

        height, width = np.shape(y)[0], np.shape(y)[1]
        
        # convert 255 to 1
        y[y == 255] = 1

        # 1
        img_data = Image.open(img_1).convert('LA')
        # reshape data:
        x_1 = np.array(img_data)[:, :, 0]
        # normalise data [0, 1]
        x_1 = (x_1 - np.min(x_1))/np.ptp(x_1)

        # 2
        img_data = Image.open(img_2).convert('LA')
        x_2 = np.array(img_data)[:, :, 0]
        x_2 = (x_2 - np.min(x_2))/np.ptp(x_2)

        # 3
        img_data = Image.open(img_3).convert('LA')
        x_3 = np.array(img_data)[:, :, 0]
        x_3 = (x_3 - np.min(x_3))/np.ptp(x_3)

        # 4
        img_data = Image.open(img_4).convert('LA')
        x_4 = np.array(img_data)[:, :, 0]
        x_4 = (x_4 - np.min(x_4))/np.ptp(x_4)

        for n_patch in range(n_patches):
            # extract random patch
            rand_int = [np.random.randint(half_patch, high=(height - half_patch)), np.random.randint(half_patch, high=(width - half_patch))]
            
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

def rename_chase_files(input_dir):
    for i in os.listdir(input_dir)[0:20]:
        img_id = i[6:9]
        new_name = img_id + '_training.png'
        old_path = os.path.join(input_dir, i)
        new_path = os.path.join(input_dir, new_name)
        shutil.move(old_path, new_path)

    for i in os.listdir(input_dir)[20:]:
        img_id = i[6:9]
        new_name = img_id + '_testing.png'
        old_path = os.path.join(input_dir, i)
        new_path = os.path.join(input_dir, new_name)
        shutil.move(old_path, new_path)

if __name__ == '__main__':
    create_image_img_folder(r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive_crossval_128-', 128, 3000, mode='drive')