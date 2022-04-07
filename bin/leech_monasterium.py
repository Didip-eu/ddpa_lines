#!/usr/bin/env python3

import fargv
from pathlib import Path
from lxml import etree
import tqdm

#namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'cei': 'http://www.monasterium.net/NS/cei'}

p = {
    "root":"../db",
    "directory_path": "./data/images_xml/",
    "file_extension": '*.xml',
    "namespaces": '{"atom": "http://www.w3.org/2005/Atom", "cei": "http://www.monasterium.net/NS/cei"}'
}

def atomid_to_url(args):
    atomids_urls =[]
    for file in tqdm.tqdm(sorted(Path(args.directory_path).rglob(args.file_extension))):
        tree = etree.parse(str(file))  # requres conversion to str since lxml does not vibe with windowspath
        root = tree.getroot()
        for img in root.findall('.//img', eval(args.namespaces)):
            atomids_urls.append((img.getparent().attrib['id'], img.attrib['src']))
    return atomids_urls

if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    print("\n".join([",".join(item) for item in atomid_to_url(args)]))
