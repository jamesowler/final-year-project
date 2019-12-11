params = {}
params['image_size_x'] = 576
params['image_size_y'] = 576
params['patch_size'] = 64
params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\drive_patches\clahe-64-large'

params['n_patches'] = int((params['image_size_x']*params['image_size_y'])/(params['patch_size']*params['patch_size']))

params['n_channels'] = 1
params['n_classes'] = 1

params['learning_rate'] = 1e-6
params['loss_method'] = 'weighted_bce'
params['loss_weight'] = 3

params['model'] = 'model_2d_u_net_shallow'

params['n_epochs'] = 15
params['batch_size'] = 30
params['val_proportion'] = 0.1