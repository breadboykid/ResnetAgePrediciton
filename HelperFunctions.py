import glob
from enum import Enum
import os
import numpy as np

class DataEnum(Enum):
    Image = 0
    Label = 1

channel_dict = {
    0: 'Cortical Thickness',
    1: 'Cortical Myelin',
    2: 'Cortical Curvature'
}

# helper function - list all files within a directory specified by extension
def list_files(path, extension):
    return glob.glob(f'{path}/*.{extension}')

# helper function - extract subject_id from file name
def extract_subject_id(filename, extension='npy'):
    return int(filename.replace('2D_projection_R_sub-', '').replace(f'.{extension}', ''))

def get_batch_image_label(dict_data):
    return dict_data[DataEnum.Image], dict_data[DataEnum.Label]

# Load all numpy files and returns
def load_numpy_files(path='', extension='npy'):
    npy_data = {}
    numpy_paths = list_files(path, extension)

    for numpy_path in numpy_paths:
        filename = os.path.basename(numpy_path)
        subj_id = extract_subject_id(filename, extension='npy')
        npy_data[subj_id] = np.load(numpy_path)

    return npy_data