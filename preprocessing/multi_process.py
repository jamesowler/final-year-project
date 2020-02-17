import glob
import os
import shutil

from preprocess import preprocessing, contrast_enhancement, extract_channel


def multi_process(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*')):
        print(i)
        preprocessing(i)

# multi_process(r"C:\Users\James\Projects\final-year-project\data\DRIVE\imgs2")

def rename(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*')):
        i_new = i.replace('-processed', '')
        print(i_new)
        shutil.move(i, i_new)

def multi_process_contrast(input_dir):
    for i in glob.glob(os.path.join(input_dir, '*')):
        print(i)
        # extract_channel(i)
        contrast_enhancement(i)

if __name__ == '__main__':

    # rename(r"C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4")
    multi_process_contrast(r"C:\Users\James\Projects\final-year-project\data\DRIVE\imgs-n4-clahe")
