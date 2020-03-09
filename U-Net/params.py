params = {}

# patch training params #####
params['patch_size'] = 128
params['preprocessing'] = 'clahe'
params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\chase_db1_128-' + params['preprocessing']
params['n_epochs'] = 15
params['batch_size'] = 25

params['n_channels'] = 1
params['n_classes'] = 1

#### Training options
params['learning_rate'] = 1e-4
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


params['model'] = 'unet'
params['weights'] = r'C:\Users\James\Projects\final-year-project\initial_weights_unet.h5'
params['val_proportion'] = 0.0