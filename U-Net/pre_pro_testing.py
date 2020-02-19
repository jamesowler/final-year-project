import matplotlib.pyplot as plt
import numpy as np

def plot_losses(file_1, file_2=None, file_3=None, file_4=None):

    losses_1 = load_data(file_1)
    
    if file_2:
        with open(file_2) as f:
            losses_2 = [float(x.strip()) for x in f]
    if file_3:
        with open(file_3) as f:
            losses_3 = [float(x.strip()) for x in f]
    if file_4:
        with open(file_4) as f:
            losses_4 = [float(x.strip()) for x in f]

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

def find_medians(data, bin_size):
    '''
    Inputs
    data: array object of data
    bin_size: distribution size to calculate median from

    Returns: numpy array of median values
    '''
    medians = []
    for i in range(0, len(data), bin_size):
        medians.append(np.median(data[i:i+bin_size]))
    return np.array(medians)

def scatter_with_median(fname):
    '''
    Overlays average median loss 
    '''
    bin_size = 1000
    data = load_data(fname)
    plt.scatter(range(len(data)), data, label='n4_clahe', s=0.1, alpha=0.5)
    medians = find_medians(data, bin_size)
    # plot median values - points lie in the center of the bin
    plt.plot(np.arange(bin_size/2, len(data) + bin_size/2, bin_size), medians, c='r')
    plt.ylim(0.08, 0.2)
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    scatter_with_median(r'C:\Users\James\Desktop\seg_test\processed_data_testing\green\k1\losses.txt')