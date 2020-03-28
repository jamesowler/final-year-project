import shutil
import os
import keras.backend as K
from keras.optimizers import Adam

from sklearn.model_selection import ShuffleSplit

from train import train_from_data_dir, train_from_data_dir_stare
from model import model_2d_u_net, model_2d_u_net_shallow
from test_model import MultiResUnet
import loss_funcs
from params import params


def copy_files(prepro_method, mode='drive'):
    
    if mode == 'chase_db1':
        save_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing\CHASE_DB1-128'
    
    if mode == 'drive':
        save_dir = r'C:\Users\James\Desktop\seg_test\processed_data_testing\DRIVE-128'
    
    model_files = [r'C:\Users\James\Projects\final-year-project\patch_model_{}.h5'.format(str(i)) for i in range(1, params['n_epochs'] + 1)]

    for i in model_files:
        file_name = os.path.basename(i)
        save_location = save_dir + f'\{prepro_method}' + f'\{file_name}'
        shutil.copy(i, save_location)
    
    shutil.copy(r'C:\Users\James\Projects\final-year-project\losses.txt', save_dir + f'\{prepro_method}' + r'\losses.txt')
    shutil.copy(r'C:\Users\James\Projects\final-year-project\accuries.txt', save_dir + f'\{prepro_method}' + r'\accuracies.txt')

def copy_files_cross_val(prepro_method, img_id, data='STARE'):
    
    save_dir = f'C:\\Users\\James\\Desktop\\seg_test\\processed_data_testing\\{data}-128'
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

def multi_train(mode='unet', data='drive'):
    '''
    Trains models for different preprocessing methods - copies models and results over to new directory
    '''

    training_data = f'C:\\Users\\James\\Projects\\final-year-project\\data\\pre-processing-test\\{data}_128-'

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
    copy_files('green', mode=data)
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
    copy_files('n4', mode=data)
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
    copy_files('clahe', mode=data)
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
    copy_files('n4-clahe', mode=data)

def cross_val_multi_train(mode='unet', data='drive_crossval'):
    '''
    Leave-one-out training for the STARE dataset - for each preprocessing method - 5 epochs each

    '''

    training_data = f'C:\\Users\\James\\Projects\\final-year-project\\data\\pre-processing-test\\{data}_128-'

    if data == 'stare':
        img_id_filters = [str(x) + '.png' for x in range(1,21)]

    if data == 'drive_crossval':
        img_id_filters = []
        img_ids = [x for x in range(1,41)]
        rs = ShuffleSplit(n_splits=5, test_size=0.5)
        for i in rs.split(img_ids):
            string_fold = []
            for j in i[0]:
                string_fold.append(f'{j:02d}')
            img_id_filters.append(string_fold)
    
    if data == 'chasse_db1_crossval':
        img_id_filters = []
        img_ids = [x for x in range(1,29)]
        rs = ShuffleSplit(n_splits=5, test_size=0.27)
        for i in rs.split(img_ids):
            string_fold = []
            for j in i[0]:
                string_fold.append(f'{j:02d}')
            img_id_filters.append(string_fold)
    
    with open('testing_imgs_for_each_fold.txt', 'w') as f:
        f.write(str(img_id_filters))
    shutil.move('testing_imgs_for_each_fold.txt', os.path.join(r'C:\Users\James\Desktop\seg_test\processed_data_testing\DRIVE-128', 'testing_imgs_for_each_fold.txt'))

    for n, img_id_filter in enumerate(img_id_filters):
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
        if data == 'stare':
            copy_files_cross_val('green', img_id_filter)
        if data == 'drive_crossval':
            copy_files_cross_val('green', str(n), data='DRIVE')
        if data == 'chase_db1_crossval':
            pass
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
        if data == 'stare':
            copy_files_cross_val('n4', img_id_filter)
        if data == 'drive_crossval':
            copy_files_cross_val('n4', str(n), data='DRIVE')
        if data == 'chase_db1_crossval':
            pass
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
        if data == 'stare':
            copy_files_cross_val('clahe', img_id_filter)
        if data == 'drive_crossval':
            copy_files_cross_val('clahe', str(n), data='DRIVE')
        if data == 'chase_db1_crossval':
            pass
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
        if data == 'stare':
            copy_files_cross_val('n4-clahe', img_id_filter)
        if data == 'drive_crossval':
            copy_files_cross_val('n4-clahe', str(n), data='DRIVE')
        if data == 'chase_db1_crossval':
            pass

if __name__ == '__main__':
    cross_val_multi_train(mode='unet')