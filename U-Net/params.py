params = {}
params['image_size_x'] = 576
params['image_size_y'] = 576

# full image training #####
params['full_img_n_epochs'] = 250

# patch training params #####
params['patch_size'] = 96
params['preprocessing'] = '\clahe'
params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\drive_patches' + params['preprocessing'] + '-' + str(params['patch_size'])
params['n_epochs'] = 10
params['batch_size'] = 25
params['n_patches'] = int((params['image_size_x']*params['image_size_y'])/(params['patch_size']*params['patch_size']))

params['n_channels'] = 1
params['n_classes'] = 1

#### Training options
params['learning_rate'] = 2e-4
params['loss_method'] = 'bce' #bce or weight_bce
params['loss_weight'] = 5

'''
Model options:

- MultiResUnet          : train MultiResUnet with patches
- MultiResUnet_full     : train MultiResUnet with full resolution images
- MultiResUnet_shallow  : train shallow MultiResUnet with patches

- unet                  : train unet with patches
- unet_full             : train unet with full resolution images
- unet_shallow          : trian shallow unet with patches

'''

params['model'] = 'MultiResUnet'
params['weights'] = None # None or r'C:\Users\James\Projects\final-year-project\patch_model_25.h5'
params['val_proportion'] = 0.05