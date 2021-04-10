from HelperFunctions import DataEnum
import numpy as np
from torch.utils.data import Dataset
import torch


class CustomDataSet(Dataset):
    def __init__(self, x, y, type='train', transform=None):
        self.transform = transform
        mmap_mode = 'r'
        self.x_data = x
        self.y_data = np.reshape(np.array(y), (-1, 1))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        desired_x_data = self.x_data[idx]
        desired_y_data = self.y_data[idx]

        if len(desired_x_data.shape) == 2:
            desired_x_data.unsqueeze_(0)

        # Returns dict format on get item
        data = {
            DataEnum.Image: desired_x_data,
            DataEnum.Label: desired_y_data
        }

        if self.transform:
            data = self.transform(data)

        return data


