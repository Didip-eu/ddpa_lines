#!/usr/bin/env python3

import fargv
from clamm_dataset import ICDAR2019Script, ClammDataset
import torch
from tormentoring import resume, save, iterate_epoch, evaluate_classifier_epoch, last
from tormentor import *

p = {
    "device": "cpu",
    "use_pretrained": 1,
    "feature_extracting":True,
    "model": "resnet",
    "num_classes": 12,
    "resume_fname": "{model}.pt",
    "batch_sz": 10,
    "lr": 0.001,
    "momentum": 0.9,
    "epochs": 100,
    "num_workers": 4,
    "val_root": "data/scripts_test",
    "val_labels": "{val_root}/gt.csv",
    "train_root": "data/ICDAR2017_CLaMM_task1_task3",
    "train_labels": "{train_root}/@ICDAR2017_CLaMM_task1_task3.csv",
    "validate_freq": 1,
    "save_freq": 1,
    "augmentation_str": "RandomPlasmaShadow ^ RandomPlasmaBrightness ^ RandomWrap ^ Identity"
}


args, _ = fargv.fargv(p)

train_ds = ClammDataset(img_root=args.train_root,script_not_date=True, gt_fname=args.train_labels)
val_ds = ICDAR2019Script(img_root=args.val_root, gt_fname=args.val_labels)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_sz, shuffle=True, num_workers=args.num_workers)
train_loader = AugmentedDataloader(train_loader, eval(args.augmentation_str), device=args.device)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_sz, shuffle=True, num_workers=args.num_workers)

model = resume(args)
criterion = torch.nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

while model.last_epoch < args.epochs:
    if model.last_epoch == 0:
        val_aggregations = iterate_epoch(model, val_loader, criterion, None)
        model.val_history[0] = evaluate_classifier_epoch(**val_aggregations)
        train_aggregations = iterate_epoch(model, train_loader, criterion, None) # evaluating training before optimizing
        model.train_history[0] = evaluate_classifier_epoch(**train_aggregations)
    current_performance = f"T:{last(model.train_history)['Accuracy']:.4} V:{last(model.val_history)['Accuracy']:.4} |"
    train_aggregations = iterate_epoch(model, train_loader, criterion, optimizer, desc=current_performance)
    model.train_history[0] = evaluate_classifier_epoch(**train_aggregations)

    if model.last_epoch % args.validate_freq == 0:
        val_aggregations = iterate_epoch(model, val_loader, criterion, None)
        model.val_history[0] = evaluate_classifier_epoch(**val_aggregations)
    if model.last_epoch % args.save_freq == 0:
        save(model, args.resume_fname)
