import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
from random import shuffle
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

    data = glob.glob(data_dir + '/training_and_validation/*.tfrecord')
<<<<<<< HEAD
    os.makedirs(os.path.abspath(data_dir + "/train"), exist_ok=True)
    os.makedirs(os.path.abspath(data_dir + "/val"), exist_ok=True)

    # Split to 90% train, 10% val
    cutoff = len(data)//10 + 1
=======
    os.makedirs(os.path.abspath(data_dir + "/train", exist_ok=True))
    os.makedirs(os.path.abspath(data_dir + "/val", exist_ok=True))

    # Split to 90% train, 10% val
    cutoff = len(data)/10 + 1
>>>>>>> origin/master
    shuffle(data)
    for data_file in data[:cutoff]:
        shutil.move(data_file, data_dir + '/val/' + os.path.basename(data_file))
    for data_file in data[cutoff:]:
        shutil.move(data_file, data_dir + '/train/' + os.path.basename(data_file))




if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)