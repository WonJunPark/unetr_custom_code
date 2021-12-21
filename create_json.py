import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
from batchgenerators.utilities.file_and_folder_operations import save_json

data_path = '../dataset/wonjun_processing/img'
label_path = '../dataset/wonjun_processing/label'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

patient_names = []
train_patient_names = []
test_patient_names = []

file_lis = sorted(os.listdir(data_path))

for file_li in file_lis:
    image_file = os.path.join(data_path, file_li)
    label_file = os.path.join(label_path, "m" + file_li)
    patient_names.append(image_file)

file_split = int(len(file_lis) * 0.8)
train_patient_names = patient_names[:file_split]
test_patient_names = patient_names[file_split:]


json_dict = {}
json_dict['name'] = "brain"
json_dict['description'] = "brain hematoma segmentation"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "brain data for nnunet"
json_dict['licence'] = ""
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "0": "background",
    "1": "IVH",
    "2": "ICH"
}

json_dict['numTraining'] = len(train_patient_names)
json_dict['numTest'] = len(test_patient_names)

json_dict['training'] = [
    {'image': "./img/%s" % i.split("/")[-1], "label": "./label/m%s" % i.split("/")[-1]} for i in
    train_patient_names]

json_dict['validation'] = [
    {'image': "./img/%s" % i.split("/")[-1], "label": "./label/%s" % i.split("/")[-1]} for i in
    test_patient_names]

save_json(json_dict, os.path.join('../dataset/wonjun_processing', "dataset2.json"))
