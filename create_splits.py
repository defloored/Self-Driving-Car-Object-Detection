import argparse
import glob
import os
import random

import numpy as np
import cv2

from utils import get_module_logger
from utils import get_dataset
import shutil


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.

    # We are given 3 tfrecords for testing in the VM workspace and we will keep it as-is. The other files
    # will be split 70/30 into training and evaluation respectively.

    # Before we do the split, we should evenly distribute the night time records.
    # To do this, we find all night time tfrecords by using OpenCV to compute average brightness, and filter by threshold.

    data = glob.glob(data_dir + '/training_and_validation/*.tfrecord')
    os.makedirs(os.path.abspath(data_dir + "/train"), exist_ok=True)
    os.makedirs(os.path.abspath(data_dir + "/val"), exist_ok=True)
    daytime = []
    nighttime = []
    mypath = "data/waymo/training_and_validation/"

    for filename in data:
        daynightdataset = get_dataset(filename)
        daynightdataset = daynightdataset.take(1)
        for batch in daynightdataset:
            if (is_night(batch)):
                nighttime.append(filename)
            else:
                daytime.append(filename)

    random.shuffle(daytime)
    random.shuffle(nighttime)
    # Split to 70% train, 30% val
    dcutoff = (len(daytime)*3) // 10 + 1
    ncutoff = (len(nighttime)*3) // 10 + 1

    for data_file in daytime[:dcutoff]:
        shutil.move(data_file, data_dir + '/val/' + os.path.basename(data_file))
    for data_file in daytime[dcutoff:]:
        shutil.move(data_file, data_dir + '/train/' + os.path.basename(data_file))
    for data_file in nighttime[:ncutoff]:
        shutil.move(data_file, data_dir + '/val/' + os.path.basename(data_file))
    for data_file in nighttime[ncutoff:]:
        shutil.move(data_file, data_dir + '/train/' + os.path.basename(data_file))

def is_night (batch):
    """
    Determine if tf record is night time.

    args:
        - batch [numpy arr]: sample image from tfrecord
    """
    img = batch['image'].numpy()
    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # compute mean of brightness
    brightnessavg = np.sum(hsv[:,:,2]) / (hsv.shape[0] * hsv.shape[1])
    # display only if night time, using a brightness threshold
    if (brightnessavg < 70):
        return True
    return False


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)