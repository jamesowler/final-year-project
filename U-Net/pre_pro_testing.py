import matplotlib.pyplot as plt
import numpy as np

def plot_two_losses(file_1, file_2=None, file_3=None):

    with open(file_1) as f:
        losses_1 = [float(x.strip()) for x in f]
    
    with open(file_2) as f:
        losses_2 = [float(x.strip()) for x in f]

    with open(file_3) as f:
        losses_3 = [float(x.strip()) for x in f]

    plt.figure(1)
    plt.plot(losses_1[0:5000])
    plt.plot(losses_2[0:5000], 'r')
    plt.plot(losses_3[0:5000], 'g')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')

    # plt.figure(2)
    # plt.plot(losses_1[0:5000])
    # plt.plot(losses_2[0:5000], 'r')
    # plt.xlabel('Batch number')
    # plt.ylabel('Loss')

    plt.show()

plot_two_losses(r'C:\Users\James\Desktop\seg_test\norm_vs_no-norm\losses_n4_clahe_k1.txt', file_2=r'C:\Users\James\Projects\final-year-project\losses.txt', file_3=r'C:\Users\James\Projects\final-year-project\losses.txt')

# plot_two_losses(r'C:\Users\James\Desktop\seg_test\norm_vs_no-norm\accuries_n4_clahe_k1.txt', file_2=r'C:\Users\James\Desktop\seg_test\norm_vs_no-norm\accuries_n4_clahe_k2.txt', file_3=r'C:\Users\James\Projects\final-year-project\accuries.txt')