from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import SegformerForSemanticSegmentation
from torchvision import models
import torch
import dataloader
import os
import torch.nn as nn
import torch.optim as optim
import time
import torch.cuda.amp
import numpy as np
import cv2
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models import UNet16, UNet11
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, confusion_matrix
import argparse
from cmath import nan

def metrics(labels, pred, lbl):
    ind_found = pred == lbl
    ind_not_found = pred != lbl
    ind_present = labels == lbl
    ind_not_present = labels != lbl

    tp = np.sum(ind_found*ind_present)
    fp = np.sum(ind_found*ind_not_present)
    fn = np.sum(ind_not_found*ind_present)
    tn = np.sum(ind_not_found*ind_not_present)

    if tp+fn+fp > 0:
        f1 = tp/(tp + 0.5*(fp + fn))
        jac = tp/(tp+fn+fp)
    else:
        f1 = nan
        jac = nan
    if tp + fp > 0:
        p = tp/(tp+fp)
    else:
        p = nan
    if tp + fn > 0:
        r = tp/(tp+fn)
    else:
        r = nan
    if tn+fp > 0:
        s = tn/(tn+fp)
    else:
        s = nan
    
    return f1, jac, p, r, s

parser = argparse.ArgumentParser()
parser.add_argument("organ_id", help="ID of the organ to train a model for", type=int)
parser.add_argument("data_dir", help="Path to DSAD dataset")
parser.add_argument("model_path", help="Path to model")
parser.add_argument("--segformer", help="Specifity to use SegFormer instead of DeepLabV3", action="store_true")
parser.add_argument("--unet", help="Specifity to use UNet instead of DeepLabV3", action="store_true")
args = parser.parse_args()

if args.unet and args.segformer:
    print("You cannot specify both segformer and unet")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
mixed_precision = True

organs = ["abdominal_wall",
            "colon",
            "inferior_mesenteric_artery",
            "intestinal_veins",
            "liver",
            "pancreas",
            "small_intestine",
            "spleen",
            "stomach",
            "ureter",
            "vesicular_glands",
            "nerves"]

organ_id = args.organ_id
organ = organs[organ_id]

test_ids = ["02", "07", "11", "13", "14", "18", "20", "32"] # Test IDs of DSAD

num_classes = 2

test_batches = 0
f1 = []
iou = []
precision = []
recall = []
specificity = []

for i in range(num_classes -1):
    f1.append([])
    iou.append([])
    precision.append([])
    recall.append([])
    specificity.append([])

batch_size = 1
mini_batch_size = 1
num_mini_batches = batch_size//mini_batch_size
image_size = (640, 512)

files = []

model_path = args.model_path

if args.unet:
    model = UNet11(num_classes=num_classes, pretrained=True)
elif args.segformer:
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",num_labels=num_classes,ignore_mismatched_sizes=True)
else:
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, num_classes)

model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

val_transform = A.Compose(
[
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

data_folder = args.data_dir

val_sets = []

print("Loading data")

for x in os.walk(data_folder):
    val = False

    if not organ in x[0]:
        continue

    if not os.path.isfile(x[0] + "/image00.png"):
        continue
    
    for id in test_ids:
        if id in x[0]:
            val = True
            break
   
    if val:
        print(x[0])
        dataset = dataloader.CobotLoaderBinary(x[0], 1, num_classes, val_transform, image_size=image_size)
        val_sets.append(dataset)
        files += dataset.files
        

val_sets = torch.utils.data.ConcatDataset(val_sets)
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=mini_batch_size, shuffle=False)
run_times = []
preds_collection = {}

for i in test_ids:
    preds_collection[i] = []

count = 0
with torch.inference_mode():
    for img, lbl in tqdm(val_loader):
        preds = []
        labels = []
        t_start = time.time()
        img = img.to(device)
        lbl = lbl.to(device).long()

        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = model(img)
            if args.unet:
                out = outputs
            elif args.segformer:
                out = nn.functional.interpolate(outputs["logits"], size=img.shape[-2:], mode="bilinear", align_corners=False)
            else:
                out = outputs['out']
            pred = torch.argmax(out, 1)
            for j in range(pred.size(0)):
                preds.append(np.ndarray.flatten(pred[j].cpu().numpy()))
                labels.append(np.ndarray.flatten(lbl[j].cpu().numpy()))
        t_stop = time.time()
        run_times.append(t_stop - t_start)
        for i in range(len(preds)):
            for j in range(1, num_classes):
                f, jac, p, r, s = metrics(labels[i], preds[i], j)
                
                f1[j-1].append(f)
                precision[j-1].append(p)
                recall[j-1].append(r)
                iou[j-1].append(jac)
                specificity[j-1].append(s)

                tid = ""

                for t in test_ids:
                    if ("/" + t + "/") in files[count]:
                        tid = t
                assert not tid == ""

                preds_collection[tid].append((jac, files[count], pred[i].cpu().numpy()))

                count+= 1

print("\nF1,IoU,Precision,Recall,Specificity,")
for i in range(num_classes-1):
    f_m = np.nanmean(f1[i])
    f_s = np.nanstd(f1[i])
    r_m = np.nanmean(recall[i])
    r_s = np.nanstd(recall[i])
    p_m = np.nanmean(precision[i])
    p_s = np.nanstd(precision[i])
    i_m = np.nanmean(iou[i])
    i_s = np.nanstd(iou[i])
    s_m = np.nanmean(specificity[i])
    s_s = np.nanstd(specificity[i])

    f = open("Results_%s.csv" % organ, "w")
    f.write("File,f1, recall, precision, iou, specificity\n")
    for j in range(len(files)):
        f.write(files[j] + ",%f,%f,%f,%f,%f\n" % (f1[i][j], recall[i][j], precision[i][j], iou[i][j], specificity[i][j]))
    f.close()

    str = "%.2f pm %.2f," % (f_m, f_s)
    str += "%.2f pm %.2f," % (i_m, i_s)
    str += "%.2f pm %.2f," % (p_m, p_s)
    str += "%.2f pm %.2f," % (r_m, r_s)
    str += "%.2f pm %.2f," % (s_m, s_s)

    print(str)

def tsort(elem):
    return elem[0]

#preds_collection.sort(key=tsort)
for tid in test_ids:
    data = preds_collection[tid]
    if len(data) == 0:
        continue
    data.sort(key=tsort)

    for i in range (1):
        elem_best = data[-1 - i]
        elem_worse = data[i]
        
        img = cv2.imread(elem_best[1]).astype(np.int32)

        orig_mask = cv2.imread(elem_best[1].replace("image", "mask"), cv2.IMREAD_GRAYSCALE)#.astype(np.int32)
        contours, hierarchy = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = np.copy(img)

        cv2.drawContours(img_contours, contours, -1, (152, 90, 19), 4)
        cv2.imwrite(organ + "_" + tid + "_best%d_orig.png" % i, img_contours)

        mask = cv2.resize(elem_best[2], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        img[mask] = (img[mask] + 255)//2
        img = img.astype(np.uint8)

        cv2.imwrite(organ + "_" + tid + "_best%d.png" % i, img)

        img = cv2.imread(elem_worse[1]).astype(np.int32)

        orig_mask = cv2.imread(elem_worse[1].replace("image", "mask"), cv2.IMREAD_GRAYSCALE)#.astype(np.int32)
        contours, hierarchy = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = np.copy(img)

        cv2.drawContours(img_contours, contours, -1, (152, 90, 19), 4)
        cv2.imwrite(organ + "_" + tid + "_worse%d_orig.png" % i, img_contours)

        mask = cv2.resize(elem_worse[2], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        img[mask] = (img[mask] + 255)//2
        img = img.astype(np.uint8)

        cv2.imwrite(organ + "_" + tid + "_worse%d.png" % i, img)
    
print("Runtime", np.mean(run_times[1:]))