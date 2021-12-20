import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)
from sklearn.metrics import confusion_matrix
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.utils import first, set_determinism

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
from utils2.colors import get_colors
import cv2
import nibabel as nib
from PIL import Image

print_config()

directory = './dataset/wonjun_processing/cache_dir'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        #Spacingd(
        #    keys=["image", "label"],
        #    pixdim=(1.5, 1.5, 2.0),
        #    mode=("bilinear", "nearest"),
        #),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        #RandCropByPosNegLabeld(
        #            keys=["image", "label"],
        #            label_key="label",
        #            spatial_size=(128, 128, 8),
        #            pos=1,
        #            neg=1,
        #            num_samples=4,
        #            image_key="image",
        #            image_threshold=0,
        #        ),
        ToTensord(keys=["image", "label"]),
    ]
)

data_dir = './dataset/wonjun_processing/'
split_JSON = "dataset.json"
datasets = data_dir + split_JSON

val_files = load_decathlon_datalist(datasets, True, "validation")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


case_num = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(512, 512, 32),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)


model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
dice_vals = list()

epoch_iterator_val = tqdm(
    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 1
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def total_metric(pred, label):
    intersection = (pred*label).sum()
    dice_coef = (2*intersection)/(pred.sum() + label.sum())
    jaccard_coef = intersection/((pred.sum() + label.sum())-intersection)

    pred = pred.flatten()
    label = label.flatten()

    # print(confusion_matrix(pred, label, labels=[1, 0]))
    # https://sites.google.com/site/torajim/articles/performance_measure

    conf = confusion_matrix(label, pred, labels=[1, 0])
    TP = conf[0, 0]
    FN = conf[0, 1]
    FP = conf[1, 0]
    TN = conf[1, 1]

    dice_coef = 2 * TP / (2 * TP + FP + FN + 1.)
    jaccard_coef = TP / (TP + FP + FN + 1.)
    acc = (TP + TN) / (TP + TN + FP + FN + 1.)
    sensitivity = TP / (TP + FN + 1.)
    specificity = TN / (FP + TN + 1.)
    precision = TP / (TP + FP + 1.)

    return round(dice_coef*100, 2), round(acc*100,2), round(sensitivity*100,2), \
           round(specificity*100,2), round(precision*100,2), round(jaccard_coef*100,2)

with torch.no_grad():
    ivh_dice_t, acc_t, sensitivity_t, specificity_t, precision_t, ivh_jaccard_t = 0, 0, 0, 0, 0, 0
    ich_dice_t, acc2_t, sensitivity2_t, specificity2_t, precision2_t, ich_jaccard_t = 0, 0, 0, 0, 0, 0
    ivh_c = 0
    ich_c = 0

    for step, batch in enumerate(epoch_iterator_val):

        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        val_outputs = sliding_window_inference(val_inputs, (512, 512, 32), 4, model)

        val_labels_list = decollate_batch(val_labels)

        val_labels_convert = [
            post_label(val_label_tensor) for val_label_tensor in val_labels_list
        ]

        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [
            post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]

        print(len(val_output_convert))
        out_ = val_output_convert[0].cpu().numpy()
        label_ = val_labels_convert[0].cpu().numpy()

        out_ = np.transpose(out_, (3, 0, 1, 2))
        label_ = np.transpose(label_, (3, 0, 1, 2))

        idx = len(out_)

        for i in range(idx):
            out = out_[i]
            label = label_[i]

            ivh = label[1]
            ich = label[2]

            if ivh.sum() != 0:
                ivh_dice, acc, sensitivity, specificity, precision, ivh_jaccard \
                    = total_metric(out[1], ivh)

                ivh_dice_t += ivh_dice
                acc_t += acc
                sensitivity_t += sensitivity
                specificity_t += specificity
                precision_t += precision
                ivh_jaccard_t += ivh_jaccard
                ivh_c += 1

            if ich.sum() != 0:
                ich_dice, acc2, sensitivity2, specificity2, precision2, ich_jaccard \
                    = total_metric(out[2], ich)

                ich_dice_t += ich_dice
                acc2_t += acc2
                sensitivity2_t += sensitivity2
                specificity2_t += specificity2
                precision2_t += precision2
                ich_jaccard_t += ich_jaccard
                ich_c += 1

            output_path = './output/{:03d}'.format(step+1)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            colors = get_colors(n_classes=3)

            img_path = './dataset/wonjun_processing/imagesTr/'
            data_file_list = sorted(os.listdir(img_path))
            data_file_list2 = data_file_list[69:]

            img_path2 = img_path + data_file_list2[step]
            imgs = nib.load(img_path2)
            imgs2 = imgs.get_fdata()
            imgs3 = np.transpose(imgs2, (2, 0, 1))

            img = imgs3[i]

            _, h, w = label.shape
            img_mask = np.zeros([h, w, 3], np.uint8)

            for idx in range(0, len(out)):
                image_idx = Image.fromarray((out[idx] * 255).astype(np.uint8))
                array_img = np.asarray(image_idx)
                img_mask[np.where(array_img == 255)] = colors[idx]

            img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            img_mask = cv2.flip(img_mask, 0)
            img_mask = cv2.flip(img_mask, 1)
            output = cv2.addWeighted(img, 0.6, img_mask, 0.4, 0)

            cv2.imwrite(output_path + '/{:03d}.png'.format(i + 1), output)

        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        dice = dice_metric.aggregate().item()

        dice_vals.append(dice)
        epoch_iterator_val.set_description(
            "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
        )

    dice_metric.reset()

print("========IVH========")
print("dice : {}".format(round(ivh_dice_t / ivh_c, 2)))
print("jacard : {}".format(round(ivh_jaccard_t / ivh_c, 2)))
print("sensitivity : {}".format(round(sensitivity_t / ivh_c, 2)))
print("specificity : {}".format(round(specificity_t / ivh_c, 2)))
print("precision : {}".format(round(precision_t / ivh_c, 2)))

print("========ICH========")
print("dice : {}".format(round(ich_dice_t / ich_c, 2)))
print("jacard : {}".format(round(ich_jaccard_t / ich_c, 2)))
print("sensitivity : {}".format(round(sensitivity2_t / ich_c, 2)))
print("specificity : {}".format(round(specificity2_t / ich_c, 2)))
print("precision : {}".format(round(precision2_t / ich_c, 2)))

mean_dice_val = np.mean(dice_vals)
print(mean_dice_val)

#
# for step, batch in enumerate(epoch_iterator_val):
#
#     with torch.no_grad():
#         img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
#         img = val_ds[case_num]["image"]
#         label = val_ds[case_num]["label"]
#         val_inputs = torch.unsqueeze(img, 1).cuda()
#         val_labels = torch.unsqueeze(label, 1).cuda()
#         val_outputs = sliding_window_inference(
#             val_inputs, (512, 512, 32), 4, model, overlap=0.8
#         )

# plt.figure("train", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Iteration Average Loss")
# x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
# y = epoch_loss_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# x = [eval_num * (i + 1) for i in range(len(metric_values))]
# y = metric_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.show()
#
# case_num = 4
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
#     img = val_ds[case_num]["image"]
#     label = val_ds[case_num]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(
#         val_inputs, (512, 512, 32), 4, model, overlap=0.8
#     )

#     plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title("image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title("label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
#     plt.subplot(1, 3, 3)
#     plt.title("output")
#     plt.imshow(
#         torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
#     )
#     plt.show()
#
