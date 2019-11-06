import os
import random
import shutil
import glob


example_dir_name = r'C:\Users\James\Projects\final-year-project\Wei-Sam\data\normal\drive_images'

def randomize():

    dir_names = [r'C:\Users\James\Projects\final-year-project\Wei-Sam\data\normal\drive_images',
    r'C:\Users\James\Projects\final-year-project\Wei-Sam\data\n4\drive_n4_images',
    r'C:\Users\James\Projects\final-year-project\Wei-Sam\data\clahe\drive_n4_plus_clahe_images']

    l = [random.randint(0,20) for i in range(5)]

    for rand_n, cross_val in enumerate(range(5)):

        for dir_name in dir_names:
            
            imgs = glob.glob(dir_name + '\*.png')

            if len(imgs) == 0:
                imgs = glob.glob(dir_name + '\*.tif')

            # shuffle files - same for each image sets - different for each cross validation set
            random.Random(l[rand_n]).shuffle(imgs)

            new_dir_name = dir_name + '_' + str(cross_val)
            os.mkdir(dir_name + '_' + str(cross_val))
            
            for n, img in enumerate(imgs[0:round((len(imgs)/2))]):
                ni = n + 1
                org_img_id = os.path.basename(img)[0:2]
                test_or_training = os.path.basename(img).split('.')[0].split('_')[1]

                shutil.copy(dir_name + f'\{org_img_id}_manual1.gif', new_dir_name + f'\{ni:02d}_manual1.gif')
                shutil.copy(dir_name + f'\{org_img_id}_{test_or_training}_mask.gif', new_dir_name + f'\{ni:02d}_test_mask.gif')
                shutil.copy(img, new_dir_name + f'\{ni:02d}_test.png')        
                
                print(img)

            for n, img in enumerate(imgs[(round((len(imgs)/2))):len(imgs)]):
                org_img_id = os.path.basename(img)[0:2]
                test_or_training = os.path.basename(img).split('.')[0].split('_')[1]

                ni = n + round((len(imgs)/2)) + 1
                shutil.copy(dir_name + f'\{org_img_id}_manual1.gif', new_dir_name + f'\{ni:02d}_manual1.gif')
                shutil.copy(dir_name + f'\{org_img_id}_{test_or_training}_mask.gif', new_dir_name + f'\{ni:02d}_training_mask.gif')

                shutil.copy(img, new_dir_name + f'\{ni:02d}_training.png')                    

randomize()


