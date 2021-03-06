import os
import glob

import scipy
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


def auc(pred_file_probs, gt_file, fov_filename):
    
    '''
    Caculates area under curve of a segmentation given that segmentation probabilities and the ground truth...

    '''
    # predicted probability map
    img1 = np.array(Image.open(pred_file_probs).convert('LA'))[:, :, 0]

    # Ground truth segmentation
    img2 = np.array(Image.open(gt_file).convert('LA'))[:, :, 0]

    fov_img = np.array(Image.open(fov_filename).convert('LA'))[:, :, 0]
    fov_reshaped = np.reshape(fov_img, (fov_img.shape[0]*fov_img.shape[1]))

    auc = roc_auc_score(np.reshape(img2, (img2.shape[0]*img2.shape[1]))[fov_reshaped == 255], np.reshape(img1, (img1.shape[0]*img1.shape[1]))[fov_reshaped == 255])
    
    return auc


def accuracy(pred_file, gt_file, fov_filename):

    '''
    Calculates accuracy of final segmentation mask
    '''

    # using PIL instead as cv2 seems to have an issue opening .gif files
    img1 = np.array(Image.open(pred_file).convert('LA'))[:, :, 0]

    if np.max(img1) == 215:
        img1[img1 == 30] = 0
        img1[img1 == 215] = 1

    img2 = np.array(Image.open(gt_file).convert('LA'))[:, :, 0]
    img2[img2 == 255] = 1

    fov_img = np.array(Image.open(fov_filename).convert('LA'))[:, :, 0]

    ## Calculate stats based on pixels witin FOV
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 1))  
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 0))  
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(img1[fov_img == 255] == 1, img2[fov_img == 255] == 0))  
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(img1[fov_img == 255] == 0, img2[fov_img == 255] == 1))

    # print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    accuray = (TP + TN)/(TP + TN + FP + FN)

    return accuray

def non_fov_accuracy(pred_file, gt_file):

    '''
    Calculates accuracy of final segmentation mask
    '''

    # using PIL instead as cv2 seems to have an issue opening .gif files
    img1 = np.array(Image.open(pred_file).convert('LA'))[:, :, 0]

    if np.max(img1) == 215:
        img1[img1 == 30] = 0
        img1[img1 == 215] = 1

    img2 = np.array(Image.open(gt_file).convert('LA'))[:, :, 0]
    img2[img2 == 255] = 1

    ## Calculate stats based on pixels witin FOV
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(img1[:] == 1, img2[:] == 1))  
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(img1[:] == 0, img2[:] == 0))  
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(img1[:] == 1, img2[:] == 0))  
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(img1[:] == 0, img2[:] == 1))

    accuray = (TP + TN)/(TP + TN + FP + FN)

    return accuray


def multi_test(results_dir, filter1, filter2, epoch_num=None):

    fov_files = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\mask\*')
    seg_ground_truths = glob.glob(r'C:\Users\James\Projects\final-year-project\data\DRIVE\DRIVE\test\1st_manual\*')

    test_results = []
    test_results_eval = []
    [test_results.append(i) for i in glob.glob(results_dir + '\*') if filter1 in i]
    [test_results_eval.append(i) for i in glob.glob(results_dir + '\*') if filter2 in i]

    accs = []
    non_fov_accs = []
    aucs = []

    # print(test_results, '\n', test_results_eval)

    for i, j, k in zip(test_results, seg_ground_truths, fov_files):
        acc = accuracy(i, j, k)
        accs.append(acc)
    
    for i, j in zip(test_results, seg_ground_truths):
        non_fov_acc = non_fov_accuracy(i,j)
        non_fov_accs.append(non_fov_acc)

    for i, j, k in zip(test_results_eval, seg_ground_truths, fov_files):
        auc_value = auc(i,j,k)
        aucs.append(auc_value)

    print(aucs, '\n \n', accs)
    print()
    print('AUC:', np.mean(aucs), '\nAccuracy: ', np.mean(accs), '\nNon FOV Accuracy', np.mean(non_fov_accs))

    return np.mean(aucs), np.mean(accs)

if __name__ == '__main__':
        
    # multi_test(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-test-1', 'seg.png', 'seg-eval.png')
    multi_test(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-TEST-3', 'seg.png', 'seg-eval.png')

    # multi_test(r'C:\Users\James\Projects\final-year-project\data\U-Net-testing\DRIVE-Soares', 'class.png', 'gray-eval')

