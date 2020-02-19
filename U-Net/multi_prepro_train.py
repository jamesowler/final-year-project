import shutil
import os
import keras.backend as K

from train import train_from_data_dir
from model import model_2d_u_net, model_2d_u_net_shallow
from params import params


def copy_files(prepro_method):
    
    save_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing'
    
    model_files = [r'C:\Users\James\Projects\final-year-project\patch_model_{}.h5'.format(str(i)) for i in range(1, params['n_epochs'] + 1)]

    for i in model_files:
        file_name = os.path.basename(i)
        save_location = save_dir + f'\{prepro_method}' + f'\{file_name}'
        shutil.copy(i, save_location)
    
    shutil.copy(r'C:\Users\James\Projects\final-year-project\losses.txt', save_dir + f'\{prepro_method}' + r'\losses.txt')
    shutil.copy(r'C:\Users\James\Projects\final-year-project\accuries.txt', save_dir + f'\{prepro_method}' + r'\accuracies.txt')


def multi_train():
    '''
    Trains models for different preprocessing methods - copies models and results over to new directory
    '''

    # # train green
    # model = model_2d_u_net_shallow(params)
    # model.load_weights(params['weights'])
    # params['preprocessing'] = 'green'
    # params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive-' + params['preprocessing']
    # train_from_data_dir(params, model)
    # copy_files('green')
    # K.clear_session()

    # train n4
    model = model_2d_u_net_shallow(params)
    model.load_weights(params['weights'])
    params['preprocessing'] = 'n4'
    params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive-' + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('n4')
    K.clear_session()

    # train clahe
    model = model_2d_u_net_shallow(params)
    model.load_weights(params['weights'])
    params['preprocessing'] = 'clahe'
    params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive-' + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('clahe')
    K.clear_session()

    # train n4_clahe
    model = model_2d_u_net_shallow(params)
    model.load_weights(params['weights'])
    params['preprocessing'] = 'n4_clahe'
    params['data_dir'] = r'C:\Users\James\Projects\final-year-project\data\pre-processing-test\drive-' + params['preprocessing']
    train_from_data_dir(params, model)
    copy_files('n4_clahe')


if __name__ == '__main__':
    multi_train()