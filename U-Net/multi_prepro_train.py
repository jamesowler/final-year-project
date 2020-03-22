import shutil
import os
import keras.backend as K
from keras.optimizers import Adam

from train import train_from_data_dir, train_from_data_dir_stare
from model import model_2d_u_net, model_2d_u_net_shallow
from test_model import MultiResUnet
import loss_funcs
from params import params


def copy_files(prepro_method):
    
    save_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing\DRIVE-128'
    
    model_files = [r'C:\Users\James\Projects\final-year-project\patch_model_{}.h5'.format(str(i)) for i in range(1, params['n_epochs'] + 1)]

    for i in model_files:
        file_name = os.path.basename(i)
        save_location = save_dir + f'\{prepro_method}' + f'\{file_name}'
        shutil.copy(i, save_location)
    
    shutil.copy(r'C:\Users\James\Projects\final-year-project\losses.txt', save_dir + f'\{prepro_method}' + r'\losses.txt')
    shutil.copy(r'C:\Users\James\Projects\final-year-project\accuries.txt', save_dir + f'\{prepro_method}' + r'\accuracies.txt')

def copy_files_stare(prepro_method, img_id):
    
    save_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing\STARE-128'
    save_dir = os.path.join(save_dir, img_id)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(os.path.join(save_dir, prepro_method)):
        os.mkdir(os.path.join(save_dir, prepro_method))
    
    model_files = [r'C:\Users\James\Projects\final-year-project\patch_model_{}.h5'.format(str(i)) for i in range(1, params['n_epochs'] + 1)]

    for i in model_files:
        file_name = os.path.basename(i)
        save_location = save_dir + f'\{prepro_method}' + f'\{file_name}'
        shutil.copy(i, save_location)
    
    shutil.copy(r'C:\Users\James\Projects\final-year-project\losses.txt', save_dir + f'\{prepro_method}' + r'\losses.txt')
    shutil.copy(r'C:\Users\James\Projects\final-year-project\accuries.txt', save_dir + f'\{prepro_method}' + r'\accuracies.txt')

def multi_train(mode='unet'):
    '''
    Trains models for different preprocessing methods - copies models and results over to new directory
    '''

    training_data = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive_128-'

    # train green
    print('Training green channel images')
    if mode == 'resunet':
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
        params['weights'] = r'C:\Users\James\Projects\final-year-project\initial_weights_multiresunet.h5'
    elif mode == 'unet':
        model = model_2d_u_net(params)
    elif mode == 'unet_shallow':
        model = model_2d_u_net_shallow(params)

    model.load_weights(params['weights'])
    params['preprocessing'] = 'green'
    params['data_dir'] = training_data + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('green')
    K.clear_session()

    # train n4
    print('Training n4 images')
    if mode == 'resunet':
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
    elif mode == 'unet':
        model = model_2d_u_net(params)
    elif mode == 'unet_shallow':
        model = model_2d_u_net_shallow(params)

    model.load_weights(params['weights'])
    params['preprocessing'] = 'n4'
    params['data_dir'] = training_data + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('n4')
    K.clear_session()

    # train clahe
    print('Training clahe images')
    if mode == 'resunet':
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
    elif mode == 'unet':
        model = model_2d_u_net(params)
    elif mode == 'unet_shallow':
        model = model_2d_u_net_shallow(params)

    model.load_weights(params['weights'])
    params['preprocessing'] = 'clahe'
    params['data_dir'] = training_data + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('clahe')
    K.clear_session()

    # train n4_clahe
    print('Training n4-clahe images')
    if mode == 'resunet':
        model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
        loss_mathod = getattr(loss_funcs, params['loss_method'])
        model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
    elif mode == 'unet':
        model = model_2d_u_net(params)
    elif mode == 'unet_shallow':
        model = model_2d_u_net_shallow(params)
        
    model.load_weights(params['weights'])
    params['preprocessing'] = 'n4-clahe'
    params['data_dir'] = training_data + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('n4-clahe')

def multi_train_stare(mode='unet'):
    '''
    Leave-one-out training for the STARE dataset - for each preprocessing method - 5 epochs each

    '''
    training_data = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\stare_128-'

    img_id_filters = [str(x) + '.png' for x in range(1,21)]

    for img_id_filter in img_id_filters:
        print(img_id_filter)
        # train green
        print('Training green channel images')
        if mode == 'resunet':
            model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
            loss_mathod = getattr(loss_funcs, params['loss_method'])
            model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
            params['weights'] = r'C:\Users\James\Projects\final-year-project\initial_weights_multiresunet.h5'
        elif mode == 'unet':
            model = model_2d_u_net(params)
        elif mode == 'unet_shallow':
            model = model_2d_u_net_shallow(params)

        model.load_weights(params['weights'])
        params['preprocessing'] = 'green'
        params['data_dir'] = training_data + params['preprocessing']
        train_from_data_dir_stare(params, model, img_id_filter)
        copy_files_stare('green', img_id_filter)
        K.clear_session()

        # train n4
        print('Training n4 images')
        if mode == 'resunet':
            model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
            loss_mathod = getattr(loss_funcs, params['loss_method'])
            model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
        elif mode == 'unet':
            model = model_2d_u_net(params)
        elif mode == 'unet_shallow':
            model = model_2d_u_net_shallow(params)

        model.load_weights(params['weights'])
        params['preprocessing'] = 'n4'
        params['data_dir'] = training_data + params['preprocessing']
        train_from_data_dir_stare(params, model, img_id_filter)
        copy_files_stare('n4', img_id_filter)
        K.clear_session()

        # train clahe
        print('Training clahe images')
        if mode == 'resunet':
            model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
            loss_mathod = getattr(loss_funcs, params['loss_method'])
            model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
        elif mode == 'unet':
            model = model_2d_u_net(params)
        elif mode == 'unet_shallow':
            model = model_2d_u_net_shallow(params)

        model.load_weights(params['weights'])
        params['preprocessing'] = 'clahe'
        params['data_dir'] = training_data + params['preprocessing']
        train_from_data_dir_stare(params, model, img_id_filter)
        copy_files_stare('clahe', img_id_filter)
        K.clear_session()

        # train n4_clahe
        print('Training n4-clahe images')
        if mode == 'resunet':
            model = MultiResUnet(params['patch_size'], params['patch_size'], 1)
            loss_mathod = getattr(loss_funcs, params['loss_method'])
            model.compile(loss=loss_mathod, metrics=['accuracy'], optimizer=Adam(lr=float(params['learning_rate'])))
        elif mode == 'unet':
            model = model_2d_u_net(params)
        elif mode == 'unet_shallow':
            model = model_2d_u_net_shallow(params)
            
        model.load_weights(params['weights'])
        params['preprocessing'] = 'n4-clahe'
        params['data_dir'] = training_data + params['preprocessing']
        train_from_data_dir_stare(params, model, img_id_filter)
        copy_files_stare('n4-clahe', img_id_filter)

if __name__ == '__main__':
    multi_train_stare(mode='unet')