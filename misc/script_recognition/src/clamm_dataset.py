from PIL import Image
import numpy as np
import torch
import torchvision
import tormentor
import sys

class ClammDataset(object):
    def __init__(self, img_root, gt_fname, script_not_date, crop_to_size=512):
        gt_tbl = [line.split(";") for line in open(gt_fname, "r").read().strip().split("\n")[1:]]
        if script_not_date:
            self.gt = [(f"{img_root}/{row[1]}", int(row[2])-1) for row in gt_tbl]
            #self.gt = [(row[1], int(row[2])) for row in gt_tbl]
        else:
            self.gt = [(f"{img_root}/{row[1]}", int(row[3])-1) for row in gt_tbl]
            #self.gt = [(row[1], int(row[3])) for row in gt_tbl]
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(crop_to_size, pad_if_needed=True)
        ])

    def __getitem__(self, n):
        img = Image.open(self.gt[n][0]).convert("RGB")
        return self.input_transform(img), torch.tensor(self.gt[n][1])

    def __len__(self):
        return len(self.gt)


if __name__ == "__main__":
    ds = ClammDataset(img_root="./data/ICDAR2017_CLaMM_task1_task3/", script_not_date = True, gt_fname="data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True)
    print(ds[0])
    for batch in dataloader:
        print(batch)
        sys.exit()