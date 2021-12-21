import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm

data_path = '../dataset/wonjun_processing/imagesTr'
label_path = '../dataset/wonjun_processing/labelsTr'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

ivh = 0
ich = 0

for d in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[d])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()

    mask_path = os.path.join(label_path,label_file_list[d])
    mask = nib.load(mask_path)
    mask = mask.get_fdata()

    # ct data slice
    ct = ct[:512, :512, :32]
    mask = mask[:512, :512, :32]

    h, w, c = ct.shape

    if c < 32:
        z_padding = 32-c
        ct = np.pad(ct, ((0, 0), (0, 0), (0, z_padding)), 'constant')
        mask = np.pad(mask, ((0, 0), (0, 0), (0, z_padding)), 'constant')

    print(d)
    print(ct.shape)

    c_path = '../dataset/wonjun_processing/img/' + data_file_list[d][:3]
    m_path = '../dataset/wonjun_processing/label/' + label_file_list[d][:4]

    ct_t = nib.Nifti1Image(ct, affine)
    mask_t = nib.Nifti1Image(mask, affine)

    nib.save(ct_t, c_path)
    nib.save(mask_t, m_path)