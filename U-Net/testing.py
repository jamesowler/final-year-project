import os
import glob

import scipy
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


def calc_metrics(filename1, filename2, fov_filename):
    
    '''
    Caculates area under curve of a segmentation given that segmentation and the ground truth...

    filename 1: predicted mask

    filename 2: grouth truth mask

    '''
    
    img1 = np.array(Image.open(filename1).convert('LA'))[:, :, 0]
    img1[img1 < 200] = 0
    img1[img1 > 200] = 1
    img1 = cv2.resize(img1, (565, 584))   
    # using PIL here instead as cv2 seems to have an issue opening .gif files
    img2 = np.array(Image.open(filename2).convert('LA'))[:, :, 0]
    img2[img2 < 200] = 0
    img2[img2 > 200] = 1

    fov_img = np.array(Image.open(fov_filename).convert('LA'))[:, :, 0]
    # print('fov max: ', np.max(fov_img))

    ## Calculate stats based on pixels witin FOV

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 1))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 0))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 0))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 1))
    # print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))


    ###################
    # value counts

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP_v = np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 1)  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN_v = np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 0)  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP_v = np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 0)  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN_v = np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 1)
    # print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))

    # ##################

    #

    # auc = 0.5*((TP_v/(TP_v+FN_v)) + (FP_v/(TN_v+FP_v)))

    # fp = FP_v/(TN+FP)
    # tp = TP_v/(TP+FN)

    # fp = fp[::-1]
    # tp = tp[::-1]

    # az = 0
    # for i in range(1,len(fp)):
    #     a = tp[i]
    #     b = tp[i-1]
    #     h = fp[i] - fp[i-1]
        
    #     area = (a + b)*h/2
    #     az += area

    # auc = az

    # ###########

    fov_reshaped = np.reshape(fov_img, (fov_img.shape[0]*fov_img.shape[1]))
    auc = roc_auc_score(np.reshape(img2, (img2.shape[0]*img2.shape[1]))[fov_reshaped == 255], np.reshape(img1, (img1.shape[0]*img1.shape[1]))[fov_reshaped == 255])
    
    # auc = 0.5*(TP/(TP+FN) + TN/(TN+FP))

    # accuray = TP + FN / FOV Pixel count
    fov_img = np.array(Image.open(fov_filename).convert('LA'))[:, :, 0]
    plt.imshow(fov_img)
    pixels_in_fov = np.count_nonzero(fov_img)

    accuray = (TP + TN)/(TP + TN + FP + FN)

    return auc, accuray


def multi_test(results_dir):

    fov_files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\mask\*')
    seg_ground_truths = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\1st_manual\*')

    test_results = []
    [test_results.append(i) for i in glob.glob(results_dir + '\*') if 'test' in i]

    aucs = []
    accs = []

    # auto-seg .png image first
    for i, j, k in zip(test_results, seg_ground_truths, fov_files):
        auc, accuracy = calc_metrics(i, j, k)
        aucs.append(auc)
        accs.append(accuracy)

    print(aucs, accs)
    print()
    print('AUC:', np.mean(aucs), 'Accuracy: ', np.mean(accs))

if __name__ == '__main__':
    # auc, accuracy = auc(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-test-1\01_test-seg.png', r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\1st_manual\01_manual1.gif', r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\mask\01_test_mask.gif')

    multi_test(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-Soares')

    # img = np.asarray(Image.open(r'C:\Users\James\Projects\final-year-project\Wei-Sam\data\drive_images\01_manual1.gif'))
    # print(type(img))