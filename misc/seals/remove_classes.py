#!/usr/bin/env python3
"""
Makes multiple 
"""


import fargv
import shutil
import os
import glob
from pathlib import Path
import tqdm


p = {
    "filter_validation":False,
    "src_folder":"./seal_ds",
    "dst_folder":"./cloned_sealds",
    "mode":("l1out", "l1in","custom"),
    "remove_classes": set([]),
    "keep_classes": set([]),
}


def clone_yolods_dataset(input_path, output_path, hardlink_images=True, remove_classes=[], filter_validation=False):
    parent = lambda x:"/".join(x.split("/")[:-1])

    input_dataset_yaml = glob.glob(f"{input_path}/*.yaml")
    assert len(input_dataset_yaml) == 1
    input_dataset_yaml = input_dataset_yaml[0]
    output_dataset_yaml = input_dataset_yaml.replace(input_path, output_path)
    assert output_dataset_yaml != input_dataset_yaml

    files = glob.glob(f"{input_path}/*/*/*")    
    dst_dirs = set([parent(file).replace(input_path, output_path) for file in files])
    for dst_dir in dst_dirs:
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
    if hardlink_images:
        link_files = set([f for f in files if f.split(".")[-1].lower() in ["jpg","jpeg","png","gif","tif","tiff"]])
    else:
        link_files = set([])
    copy_files = set(files) - link_files
    for src_file in tqdm.tqdm(link_files,desc="Linking files"):
        dst_file = src_file.replace(input_path, output_path)
        os.link(src_file, dst_file)
    for src_file in tqdm.tqdm(copy_files,desc="Copying files"):
        dst_file = src_file.replace(input_path, output_path)
        if "validate" in dst_file.split("/") and not filter_validation:
            lines=[l.strip() for l in open(src_file,"r").read().strip().split("\n")]
        else:
            lines=[l.strip() for l in open(src_file,"r").read().strip().split("\n") if l.split(" ")[0] not in remove_classes]
        try:
            open(dst_file, "w").write("".join([f"{line}\n" for line in lines]))
        except FileNotFoundError:
            print("FileNotFoundError ",src_file, dst_file)
    open(output_dataset_yaml,"w").write(open(input_dataset_yaml,"r").read().replace(input_path, output_path).replace("//", "/"))



if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    # either remove or keep is defined but not both
    all_classes = set([str(n)for n in range(11)])
    if args.mode == "l1out":
        for remove in all_classes:
            remove_classes = set([remove])
            dst_dir = f"{args.dst_folder}/l1out/cl_{remove}/"
            print(f"{args.src_folder} -> {dst_dir}")
            clone_yolods_dataset(args.src_folder, dst_dir, remove_classes=remove_classes, filter_validation=args.filter_validation)
    elif args.mode == "l1in":
        for keep in all_classes:
            remove_classes = all_classes - set([keep])
            dst_dir = f"{args.dst_folder}/l1in/cl_{keep}/"
            clone_yolods_dataset(args.src_folder, dst_dir, remove_classes=remove_classes, filter_validation=args.filter_validation)    
    elif args.mode == "custom":
        assert bool(args.remove_classes) != bool(args.keep_classes)
        if len(args.remove_classes) == 0:
            remove_classes = set(all_classes) - set(args.keep_classes)
        else:
            remove_classes = args.remove_classes
        clone_yolods_dataset(args.src_folder, args.dst_folder, remove_classes=remove_classes, filter_validation=args.filter_validation)
    else:
        raise ValueError