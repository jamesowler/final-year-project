params = {}

# patch training params #####
params['patch_size'] = 128
params['preprocessing'] = 'n4-clahe'
params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive_128-' + params['preprocessing']
params['n_epochs'] = 3
params['batch_size'] = 25

params['n_channels'] = 1
params['n_classes'] = 1

#### Training options
params['learning_rate'] = 5e-3
params['loss_method'] = 'bce' #bce or weighted_bce
params['loss_weight'] = 4

'''
Model options:

- MultiResUnet          : train MultiResUnet with patches
- MultiResUnet_full     : train MultiResUnet with full resolution images
- MultiResUnet_shallow  : train shallow MultiResUnet with patches

- unet                  : train unet with patches
- unet_full             : train unet with full resolution images
- unet_shallow          : trian shallow unet with patches

'''


params['model'] = 'unet'
params['weights'] =  r'C:\Users\James\Projects\final-year-project\initial_weights_unet.h5'
params['val_proportion'] = 0  # 0.4 for chase