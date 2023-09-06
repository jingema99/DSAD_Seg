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
import argparse
from models_multi import MultiHeadSegFormer, MultiHeadDeepLab

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Path to DSAD dataset")
parser.add_argument("output_dir", help="Path to output folder")
parser.add_argument("--segformer", help="Specifity to use SegFormer instead of DeepLabV3", action="store_true")
args = parser.parse_args()

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

val_ids = ["03", "21", "26"] # Validation IDs of DSAD

test_ids = ["02", "07", "11", "13", "14", "18", "20", "32"] # Test IDs of DSAD

# Parameters for training
num_classes = len(organs)
batch_size = 16
mini_batch_size = 16
best_f1 = 0
num_mini_batches = batch_size//mini_batch_size
image_size = (640, 512)
lr = 1e-4

output_folder = "Seg_multi"

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
        tp = torch.sum((pred[i] == label)*(lbl[i] == label)).item()
        fp = torch.sum((pred[i] == label)*(lbl[i] != label)).item()
        fn = torch.sum((pred[i] != label)*(lbl[i] == label)).item()
        tn = torch.sum((pred[i] == 0)*(lbl[i] != label)).item()
        
        list[label-1][0] += tp
        list[label-1][1] += fp
        list[label-1][2] += fn
        if (tp + fp + fn) > 0:
            f1 = tp/(tp + 0.5*(fp + fn))
            jc = tp/(tp + fp + fn)
            list[label-1][3].append(f1)
            list[label-1][4].append(jc)

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

weights = np.zeros((2, num_classes), dtype=np.float32)

for x in os.walk(data_folder):
    val = False

    if debug:
        if not "03" in x[0] and not "04" in x[0] and not "05" in x[0]:
            continue

    if not os.path.isfile(x[0] + "/image00.png"):
        continue
    c_lbl = -1
    for i in range(len(organs)):
        if organs[i] in x[0]:
            c_lbl = i + 1
            break
    
    if c_lbl == -1:
        continue

    for id in val_ids:
        if id in x[0]:
            val = True
            break
    test = False
    for id in test_ids:
        if id in x[0]:
            test = True
            break
    
    if test:
        continue
    print(x[0])
    if val:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, num_classes, val_transform, image_size=image_size)
        val_sets.append(dataset)
    else:
        dataset = dataloader.CobotLoaderBinary(x[0], c_lbl, num_classes, train_transform, image_size=image_size, create_negative_labels=True)
        train_sets.append(dataset)
        bg_w, p = dataset.get_frequency()
        c_w = p - bg_w 
        weights[0, c_lbl-1] += bg_w
        weights[1, c_lbl-1] += c_w

print(weights)
weight = np.zeros(2, dtype=np.float32)
for i in range(len(organs)):
    n_samples = weights[0, i] + weights[1, i]
    weight[0] += n_samples/(2*weights[0, i])
    weight[1] += n_samples/(2*weights[1, i])
    print(i, weight)
weight/=len(organs)

criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device), ignore_index=-1)

train_sets = torch.utils.data.ConcatDataset(train_sets)
val_sets = torch.utils.data.ConcatDataset(val_sets)

train_loader = torch.utils.data.DataLoader(train_sets, batch_size=mini_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=mini_batch_size, shuffle=False)

if args.segformer:
    model = MultiHeadSegFormer(num_classes)
else:
    model = MultiHeadDeepLab(num_classes)

model.train()
model.to(device)

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

    for img, lbl_orig, lbls, mask_orig in train_loader:
        if img.size(0) < mini_batch_size:
            continue

        img = img.to(device)
        lbl = lbl_orig.to(device).long()
        u_lbls = mask_orig.to(device).long()
        lbls = lbls.to(device) - 1

        with torch.cuda.amp.autocast(enabled=mixed_precision):

            outputs = model(img)
            out = torch.stack(outputs, dim=2)
        
            loss = criterion(out, lbl)
            
            pred = torch.argmax(out, 1)

            with torch.no_grad():
                outputs2 = torch.stack(outputs, dim=1)
                ind = lbls
                dims = [1]
                for s in outputs2.shape[2:]:
                    ind = ind.unsqueeze(-1)
                    dims.append(s)

                ind = ind.repeat(dims)

                ind = ind.unsqueeze(1)

                output_filter = torch.gather(outputs2, 1, ind)
                output_filter = output_filter.squeeze(1)
                u_preds = torch.argmax(output_filter.detach(), dim=1)
                for b in range(u_preds.shape[0]):
                    u_preds[b] = u_preds[b]*(lbls[b] + 1)
                    u_lbls[b] = u_lbls[b]*(lbls[b] + 1)

                update_metrics(train_metrics, u_preds, u_lbls)

                train_loss.append(loss.item())

                train_accuracy.append(torch.sum(u_preds == u_lbls).item()/(mini_batch_size*image_size[0]*image_size[1]))

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
        for img, lbl, lbls in val_loader:
            img = img.to(device)
            lbl = lbl.to(device).long()
            lbls = lbls.to(device) - 1

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(img)

                loss = 0
                u_preds = []
                u_lbl = []
                pr = []

                outputs = torch.stack(outputs, dim=1)

                ind = lbls
                dims = [1]
                for s in outputs.shape[2:]:
                    ind = ind.unsqueeze(-1)
                    dims.append(s)

                ind = ind.repeat(dims)

                ind = ind.unsqueeze(1)

                output_filter = torch.gather(outputs, 1, ind)
                output_filter = output_filter.squeeze(1)
                
                loss = criterion(output_filter, lbl)

                u_preds = torch.argmax(output_filter, dim=1).detach()
                for b in range(u_preds.shape[0]):
                    u_preds[b]*=lbls[b]+1
                    lbl[b]*=(lbls[b]+1)

                update_metrics(val_metrics, u_preds, lbl)

                val_loss.append(loss.item())
                val_accuracy.append(torch.sum(u_preds == lbl).item()/(mini_batch_size*image_size[0]*image_size[1]))

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

