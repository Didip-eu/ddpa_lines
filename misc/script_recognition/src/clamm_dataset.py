from PIL import Image
import numpy as np
import torch
import torchvision
import tormentor
import sys

class ClammDataset(object):
    def __init__(self, img_root, gt_fname, script_not_date, crop_to_size=512, ):
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

    class2scripts = {0:"caroline",1:"cursiva",2:"half-uncial", 3:"humanistic",4:"humanistic cursive",
                     5:"hybrida", 6:"praegothica", 7:"semihybrida", 8:"semitextualis", 9:"southern_textualis",
                     10:"textualis", 11: "uncial"}
    class2year_ranges = {0:(0,1000),
    1:(1001,1101),
    2:(1101,1201),
    3:(1201,1251),
    4:(1251,1301),
    5:(1301,1351),
    6:(1351,1401),
    7:(1401,1426),
    8:(1426, 1451),
    9:(1451,1476),
    10:(1476,1501),
    11:(1501,1526),
    12:(1526, 1551),
    13:(1551,1576),
    14:(1576,1601)}


class ICDAR2019Script(object):
    def __init__(self, img_root, gt_fname, script_not_date=True, crop_to_size=1024):
        gt_tbl = [line.split(",") for line in open(gt_fname, "r").read().strip().split("\n")[1:]]
        if script_not_date:
            script2class = {v: k for k, v in ClammDataset.class2scripts.items()}
            self.gt = [(f"{img_root}/{row[0]}", script2class[row[1]]) for row in gt_tbl]
            # self.gt = [(row[1], int(row[2])) for row in gt_tbl]
        else:
            raise NotImplementedError
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
    for batch in dataloader:
        print(batch)
        sys.exit()