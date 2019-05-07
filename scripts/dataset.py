
from skimage import io
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class KittiDatasetTrain(Dataset):
    def __init__(self, rootdir = "../input/data_road/data_road", transform=None):
        self.transform = transform
        self.rootdir = rootdir
        self.traindir = rootdir + "/training/image_2/"
        self.labeldir = rootdir + "/training/gt_image_2/"

    def __getitem__(self, index):

        image_path = list(label_paths)[index]
        img = io.imread(os.path.join(self.traindir, image_path))
        label = io.imread(label_paths[image_path])
        background_color = np.array([255, 0, 0])
        img = cv2.resize(img, (256, 256))
        label = cv2.resize(label, (256, 256))
        gt_bg = np.all(label == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        img = normalize(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        img = img.transpose((2, 0, 1))
        gt_image = gt_image.transpose((2, 0, 1))
        gt_image = gt_image.astype("float")

        img = torch.from_numpy(img)
        #img = torch.clamp(img, 0, 1)
        sample = {'image': img,
                'label': torch.from_numpy(gt_image)}

        return sample

    def __len__(self):
        path, dirs, files = next(os.walk(self.traindir))
        n = len(files)
        return n # of how many examples(images?) you have1

class KittiDatasetTest(Dataset):
    def __init__(self, rootdir = "../input/data_road/data_road", transform=None):
        self.transform = transform
        self.rootdir = rootdir
        self.testdir = os.path.join(rootdir, "testing/image_2")

    def __getitem__(self, index):

        image_path = test_paths[index]
        img = io.imread(os.path.join(self.testdir, image_path))

        img = cv2.resize(img, (256, 256))
        img = normalize(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img)
        #img = torch.clamp(img, 0, 1)
        sample = {'image': img}

        return sample

    def __len__(self):
        path, dirs, files = next(os.walk(self.testdir))
        n = len(files)
        return n
