params = {}
params['image_size_x'] = 576
params['image_size_y'] = 576
params['patch_size'] = 32
params['n_patches'] = int((params['image_size_x']*params['image_size_y'])/(params['patch_size']*params['patch_size']))

params['n_channels'] = 1
params['n_classes'] = 1

params['learning_rate'] = 5e-5

params['n_epochs'] = 15000
params['batch_size'] = 25