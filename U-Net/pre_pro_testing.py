import matplotlib.pyplot as plt
import numpy as np

bin_size = 1000

def plot_losses(file_1, file_2=None, file_3=None, file_4=None):

    losses_1 = load_data(file_1)
    
    if file_2:
        losses_2 = load_data(file_2)
    if file_3:
        losses_3 = load_data(file_3)
    if file_4:
        losses_4 = load_data(file_4)

    plt.figure(1)
    plt.plot(losses_1[0:150], label='n4_clahe')
    if file_2:
        plt.plot(losses_2[0:150], 'g', label='green_chan')
    if file_3:
        plt.plot(losses_3[0:150], 'r', label='n4')
    if file_4:
        plt.plot(losses_4[0:150], 'y', label='clahe')
    plt.legend()
    plt.xlabel('Batch number')
    plt.ylabel('Loss')

    plt.figure(2)
    plt.plot(losses_1, label='n4_clahe')
    if file_2:
        plt.plot(losses_2[0:150], 'g', label='green_chan')
    if file_3:
        plt.plot(losses_3[0:150], 'r', label='n4')
    if file_4:
        plt.plot(losses_4[0:150], 'y', label='clahe')
    plt.legend()
    plt.xlabel('Batch number')
    plt.ylabel('Loss')

    plt.show()

def load_data(fname):
    with open(fname) as f:
        losses = [float(x.strip()) for x in f]
    return losses

def find_means(data, bin_size):
    '''
    Inputs
    data: array object of data
    bin_size: distribution size to calculate median from

    Returns: numpy array of median values
    '''
    means = []
    for i in range(0, len(data), bin_size):
        means.append(np.mean(data[i:i+bin_size]))
    return np.array(means)

def scatter_with_median(fname):
    '''
    Overlays average median loss 
    '''
    data = load_data(fname)
    plt.scatter(range(len(data)), data, label='n4_clahe', s=0.1, alpha=0.5)
    means = find_means(data, bin_size)
    # plot median values - points lie in the center of the bin
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), means, c='r')
    plt.ylim(0.07, 0.2)
    plt.title('Mean training loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.show()

def plot_means(fname1, fname2, fname3, fname4):
    data = load_data(fname1)
    d_1 = find_means(load_data(fname1), bin_size)
    d_2 = find_means(load_data(fname2), bin_size)
    d_3 = find_means(load_data(fname3), bin_size)
    d_4 = find_means(load_data(fname4), bin_size)
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), d_1, c='r', label='clahe')
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), d_2, c='b', label='n4')
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), d_3, c='g', label='green')
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), d_4, c='y', label='n4+clahe')
    plt.ylim(0.065, 0.2)
    plt.title('Mean training loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # scatter_with_median(r'C:\Users\James\Desktop\seg_test\processed_data_testing\clahe\ResUnet\losses.txt')

    # plot_means(r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4_clahe\Shallow_Unet\losses.txt', r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4_clahe\ResUnet\losses.txt', r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4_clahe\Unet\losses.txt',r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4_clahe\Unet\losses.txt')

    '''
    Unet
    Shallow_Unet
    ResUnet
    '''
    plot_means(r'C:\Users\James\Desktop\seg_test\processed_data_testing\clahe\Shallow_Unet\losses.txt', r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4\Shallow_Unet\losses.txt',r'C:\Users\James\Desktop\seg_test\processed_data_testing\green\Shallow_Unet\losses.txt', r'C:\Users\James\Desktop\seg_test\processed_data_testing\n4-clahe\Shallow_Unet\losses.txt')