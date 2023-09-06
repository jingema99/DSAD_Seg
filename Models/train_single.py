from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import SegformerForSemanticSegmentation
from torchvision import models
import torch
import dataloader
import os
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
import numpy as np
import sys
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from models import UNet11
import argparse

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("organ_id", help="ID of the organ to train a model for", type=int)
parser.add_argument("data_dir", help="Path to DSAD dataset")
parser.add_argument("output_dir", help="Path to output folder")
parser.add_argument("--segformer", help="Specifity to use SegFormer instead of DeepLabV3", action="store_true")
parser.add_argument("--unet", help="Specifity to use UNet instead of DeepLabV3", action="store_true")
args = parser.parse_args()

if args.unet and args.segformer:
    print("You cannot specify both segformer and unet")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mixed_precision = True
epochs = 100

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
            "vesicular_glands"]

organ_id = args.organ_id
organ = organs[organ_id]

val_ids = ["03", "21", "26"] # Validation IDs of DSAD

test_ids = ["02", "07", "11", "13", "14", "18", "20", "32"] # Test IDs of DSAD

# Parameters for training
num_classes = 2
batch_size = 16
mini_batch_size = 16
best_f1 = 0
num_mini_batches = batch_size//mini_batch_size
image_size = (640, 512)
lr = 1e-4

output_folder = "Seg_single_" + organ
if args.unet:
    output_folder += "_unet"

if args.segformer:
    output_folder += "_segformer"
    
output_folder = os.path.join(args.output_dir, output_folder)

os.makedirs(output_folder, exist_ok=True)

train_transform = A.Compose(
[
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose(
[
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

data_folder = args.data_dir

train_sets = []
val_sets = []

print("Loading data")

#Functions for calculating metrics
def init_metrics(list, num_labels):
    for i in range(num_labels):
        list.append([0,0,0,[],[]])

def update_metrics(list, pred, lbl):
    for i in range(pred.size(0)):
        label = torch.max(lbl[i]).item()
        tp = torch.sum((pred[i] == label)*(lbl[i] != 0)).item()
        fp = torch.sum((pred[i] == label)*(lbl[i] == 0)).item()
        fn = torch.sum((pred[i] != label)*(lbl[i] != 0)).item()
        tn = torch.sum((pred[i] == 0)*(lbl[i] == 0)).item()

        list[label][0] += tp
        list[label][1] += fp
        list[label][2] += fn
        if (tp + fp + fn) > 0:
            f1 = tp/(tp + 0.5*(fp + fn))
            jc = tp/(tp + fp + fn)
            list[label][3].append(f1)
            list[label][4].append(jc)

def compute_avg_metrics(list, ignore_zero_label=True):
    f1s = []
    f1s2 = []
    prs = []
    rcs = []
    jcs = []
    jcs2 = []

    for i in range(len(list)):
        if i == 0 and ignore_zero_label:
            continue

        tp = list[i][0]
        fp = list[i][1]
        fn = list[i][2]

        if (tp + fp + fn) > 0:
            f1 = tp/(tp + 0.5*(fp + fn))
            jc = tp/(tp + fp + fn)

            f1s.append(f1)
            jcs.append(jc)
            if (fp + tp) > 0:
                prs.append(tp/(fp + tp))
            if (fn + tp) > 0:
                rcs.append(tp/(tp+fn))
        f1s2.append(np.mean(list[i][3]))
        jcs2.append(np.mean(list[i][4]))
            
    return np.nanmean(f1s), np.nanmean(prs), np.nanmean(rcs), np.nanmean(jcs), np.nanmean(f1s2), np.nanmean(jcs2)

weights = np.zeros(num_classes, dtype=np.float32)

for x in os.walk(data_folder):
    val = False
    test = False

    if debug:
        if not "03" in x[0] and not "04" in x[0] and not "05" in x[0]:
            continue

    if organ in x[0]:
        c_lbl = 1
    else:
        continue

    if not os.path.isfile(x[0] + "/image00.png"):
        continue

    for id in test_ids: #Skip test data
        if id in x[0]:
            test = True
            break
    if test:
        continue #Skip test data

    for id in val_ids:
        if id in x[0]:
            val = True
            break
    print(x[0])
    
    # Create dataloaders
    if val:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, num_classes, val_transform, image_size=image_size)
        val_sets.append(dataset)
    else:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, num_classes, train_transform, image_size=image_size)
        train_sets.append(dataset)
        #Collect frequencies for class weights
        bg_w, p = dataset.get_frequency()
        c_w = p - bg_w 
        weights[0] += bg_w
        weights[c_lbl] += c_w

print(weights)
n_samples = np.sum(weights)

weights = n_samples/(num_classes*weights)
weights[weights == np.inf] = 0.1
print(weights)

train_sets = torch.utils.data.ConcatDataset(train_sets)
val_sets = torch.utils.data.ConcatDataset(val_sets)

train_loader = torch.utils.data.DataLoader(train_sets, batch_size=mini_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=mini_batch_size, shuffle=False)

if args.unet:
    model = UNet11(num_classes=num_classes, pretrained=True)
elif args.segformer:
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",num_labels=num_classes,ignore_mismatched_sizes=True)
else:
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, num_classes)

model.train()
model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-1)#, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=True)

log_file = open(os.path.join(output_folder, "log.txt"), "w")
print("Training model")
for e in range(epochs):
    optimizer.zero_grad()
    train_batches = 0

    model.train()
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_metrics = []
    val_metrics = []

    init_metrics(train_metrics, num_classes)
    init_metrics(val_metrics, num_classes)

    for img, lbl, _ in train_loader:
        if img.size(0) < 2:
            continue
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
            
            loss = criterion(out, lbl)
            
            pred = torch.argmax(out, 1)

            update_metrics(train_metrics, pred, lbl)

            train_loss.append(loss.item())
            train_accuracy.append(torch.sum(pred == lbl).item()/(mini_batch_size*image_size[0]*image_size[1]))

        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        train_batches += 1

        if train_batches % num_mini_batches == 0:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
    if train_batches % num_mini_batches != 0:
        if mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    model.eval()
    preds = []

    with torch.no_grad():
        for img, lbl, _ in val_loader:
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
                loss = criterion(out, lbl)
                pred = torch.argmax(out, 1)
                preds.append(pred.cpu().numpy())

                update_metrics(val_metrics, pred, lbl)

                val_loss.append(loss.item())
                val_accuracy.append(torch.sum(pred == lbl).item()/(mini_batch_size*image_size[0]*image_size[1]))

    train_metrics, train_pr, train_rc, train_jac, train_f1_2, train_jac_2 = compute_avg_metrics(train_metrics)
    val_metrics, val_pr, val_rc, val_jac, val_f1_2, val_jac_2 = compute_avg_metrics(val_metrics)

    str = "Epoch %d: Train (loss %.3f accuracy %.3f f1 %.3f (f1 %.3f) pr %.3f rc %.3f jac %.3f (jac %.3f ) Validation (loss %.3f accuracy %.3f f1 %.3f (f1 %.3f) pr %.3f rc %.3f jac %.3f (jac %.3f )" % (e, np.mean(train_loss), np.mean(train_accuracy), train_metrics, train_f1_2, train_pr, train_rc, train_jac, train_jac_2, np.mean(val_loss), np.mean(val_accuracy), val_metrics, val_f1_2, val_pr, val_rc, val_jac, val_jac_2)
    print(str)
    log_file.write(str + "\n")
    log_file.flush()

    scheduler.step()

    if (e + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(output_folder, "model%04d.th" % e))

    if best_f1 < val_f1_2:
        best_f1 = val_f1_2
        torch.save(model.state_dict(), os.path.join(output_folder, "model_best.th"))

