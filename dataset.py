from __future__ import absolute_import, division, print_function

import os
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from collections import OrderedDict as odict
from sklearn.model_selection import StratifiedKFold
from utils import copy_mmap, pickle_load
import transforms
#import transforms_2d
from visdom import Visdom
from summary import Image3D


def fold_split(FG, seed=1234):
    labels = tuple(FG.labels)
    assert labels in [('AD', 'CN'), ('AD', 'MCI', 'CN')]
    subject_ids = pickle_load(FG.subject_ids_path)
    diagnosis = pickle_load(FG.diagnosis_path)

    X = np.array([sid for sid in subject_ids['smri'] if diagnosis[sid] in labels])
    y = np.array([diagnosis[x] for x in X])

    skf = StratifiedKFold(n_splits=FG.fold, shuffle=True, random_state=seed)
    train, test = list(skf.split(X, y))[FG.running_fold]

    print('Train ', tuple(
        zip(labels, [len(X[train][y[train] == label]) for label in labels])))
    print('Test ', tuple(
        zip(labels, [len(X[test][y[test] == label]) for label in labels])))

    ratio = [len(X[train][y[train] == label])/len(X[train]) for label in labels]
    ratio = torch.Tensor(2*(1-np.array(ratio)))

    return X, y, train, test, ratio


def Trainset(FG):
    FG.labels = tuple(FG.labels)
    assert FG.labels in [('AD', 'CN'), ('AD', 'MCI', 'CN')]
    subject_ids = pickle_load(FG.subject_ids_path)
    diagnosis = pickle_load(FG.diagnosis_path)

    X = np.array([sid
                  for sid in subject_ids['smri']
                  if diagnosis[sid] in FG.labels])
    y = np.array([diagnosis[x] for x in X])

    AD, MCI, CN = 0, 0, 0
    for x in X:
        if diagnosis[x] == 'AD':
            AD += 1
        elif diagnosis[x] == 'MCI':
            MCI += 1
        elif diagnosis[x] == 'CN':
            CN += 1
    print('Train : AD(',AD,') MCI(',MCI,') CN(',CN,')')

    return X, y


class ADNIDataset(Dataset):
    def __init__(self, FG, samples, targets, cropping=None, transform=None):
        self.labels = FG.labels
        self.root = FG.data_root
        self.samples = samples #sid
        self.targets = targets
        self.cropping = cropping
        self.transform = transform
        self.axis = FG.axis
        self.gm = FG.gm
        self.img_size = FG.isize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        target = self.labels.index(self.targets[idx])
        #print(sid, target)
        mri = copy_mmap(nib.load(os.path.join(
                self.root, 'mri_'+str(self.img_size), sid+'.nii')).get_data())

        if self.transform is not None:
            mri = self.transform(mri)

        return dict(image=mri, target=target, sid=sid)


class ADNIDataset2D(Dataset):
    def __init__(self, FG, samples, targets, cropping=None, transform=None):
        self.labels = FG.labels
        self.root = FG.data_root
        self.samples = samples #sid
        self.targets = targets
        self.cropping = cropping
        self.transform = transform
        self.axis = FG.axis
        self.gm = FG.gm
        self.img_size = FG.isize


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        target = self.labels.index(self.targets[idx])
        #print(sid, target)
        mri = torch.load(os.path.join(
            self.root, 'mri_'+str(self.img_size)+'_'+str(self.axis), sid+'.pth'))

        # mri = copy_mmap(nib.load(os.path.join(
        #         self.root, 'mri_'+str(self.img_size), sid+'.nii')).get_data())
        #
        # if self.img_size == 157:
        #     if self.axis == 2:
        #         mri = mri[76, 1:, :]
        #         mri = mri[76, 1:, :]
        #     elif self.axis == 1:
        #         mri = mri[1:,94,:]
        #     elif self.axis == 0:
        #         mri = mri[1:,1:,76]
        # else:
        #     if self.axis == 2:
        #         mri = mri[36,:,:]
        #     elif self.axis == 1:
        #         mri = mri[:,47,:]
        #     elif self.axis == 0:
        #         mri = mri[:,:,36]
        #
        # if self.transform is not None:
        #     mri = self.transform(mri)

        return dict(image=mri, target=target, sid=sid)



if __name__ == '__main__':
    labels = ('AD', 'CN')
    sids, targets, train_idx, test_idx, ratio = fold_split(
        5, 0, labels, 'data/subject_ids.pkl', 'data/diagnosis.pkl')
    trainset = ADNIDataset(labels, 'data/', sids[train_idx], targets[train_idx])
    from summary import Image3D
    from visdom import Visdom

    vis = Visdom(port=10001, env='datacheck')
    mri_print = Image3D(vis, 'mri')
    for smri, target in iter(trainset):
        print(smri.max(), smri.min())
        print(smri[:, :, 2:-2].shape)
        mri_print(str(target), smri[:, :, 2:-2], 10)
        exit()
