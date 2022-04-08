#!/usr/bin/env python3

import fargv
from pathlib import Path
from lxml import etree
import tqdm
import glob

p = {
    "mode": ["search_images", "atomid_url_filename_csv"],
    "icarus_root": "./data/icarus_mirror/",
    "md5_root": "{icarus_root}/md5imgdb/",
    "img_root": "{icarus_root}/img/",
    "icarus_map":"{md5_root}/map.json",
    "root": "../db",
    "directory_path": "./data/images_xml/",
    "file_extension": '*.xml',
    "namespaces": 'dict([["atom": "http://www.w3.org/2005/Atom"], ["cei": "http://www.monasterium.net/NS/cei"]])'
}

def get_all_atomid_url_filename(args):
    atomids_urls =[]
    for file in tqdm.tqdm(sorted(Path(args.directory_path).rglob(args.file_extension))):
        tree = etree.parse(str(file))  # requres conversion to str since lxml does not vibe with windowspath
        root = tree.getroot()
        for img in root.findall('.//img', eval(args.namespaces)):
            filename = img.attrib['src'].split("/")[-1]
            atomids_urls.append((img.getparent().attrib['id'], img.attrib['src'], filename))
    return atomids_urls

if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    if args.mode == "search_images":
        icarus_img_path = Path(args.icarus_map)
        triplets = sorted(get_all_atomid_url_filename(args))
        open("/tmp/allrows.txt", "w").write("\n".join([f"r{n:07}:{r}" for n, r in enumerate(triplets)]))
        for atom, url, filename in tqdm.tqdm(triplets):
            found = icarus_img_path.rglob(filename)
            print(f"{atom}@@{'@@'.join(found)}@@")
    elif args.mode == "atomid_url_filename_csv":
        print("\n".join([",".join(item) for item in get_all_atomid_url_filename(args)]))
    else:
        raise ValueError