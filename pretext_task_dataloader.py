import os
import torch
import pickle
from torch.utils.data import Dataset
import glob
from positions import positions
import matplotlib.pyplot as plt

class PretextTaskDataset(Dataset):

    def __init__(self, base_path='./pretext_task/'):

        self.base_path = base_path

        self.class_names = self.get_files_in_folder(base_path)

        self.file_names = glob.glob(os.path.join(base_path, '*/*/*'))

        
    def __getitem__(self, idx):
        """
        returns a dict with the followign keys:
        img: [640 x 480 x 3] image
        dep: [640 x 480] depth
        pos: [3] cartesian coordinates of the robot end effector with the refernce frame being base of the robot
        """
        file_name = self.file_names[idx]
        data = pickle.load(open(file_name, "rb"))
        pos_id = data["pos"]
        data["pos"] = positions[pos_id]
        return data

    def __len__(self):
        return len(self.file_names)

    def get_files_in_folder(self, path):
        list_of_files = glob.glob(os.path.join(path, '*'))
        class_names = []
        for l in list_of_files:
            class_names.append(l.split("/")[2])
        return class_names

def test():
    dataset = PretextTaskDataset()

    for d in dataset:
        print(d.keys())
        exit()