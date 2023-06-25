""" TODO (optional): Map attributes to elements with second mapping instead of long xpaths; test speeds - expectation: should be faster
"""

import json
import tqdm
import fargv
from lxml import etree
from pathlib import Path

from ddp_util import get_path_list

p = {
    "root_dir": ".",
    "fsdb_dir": "{root_dir}/data/leech_db",
    "cei_filename": "cei.xml",
    "output_filename":"charter.cei2json.json"
    }

namespaces = {"atom": "http://www.w3.org/2005/Atom", "cei": "http://www.monasterium.net/NS/cei"}
xpath_expressions = {
    "cei_date": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/text()",
    "cei_date_ATTRIBUTE_value": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@value",
    "cei_date_ATTRIBUTE_notBefore": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@notBefore",
    "cei_date_ATTRIBUTE_notAfter": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@notAfter",
    "cei_dateRange": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/text()",
    "cei_dateRange_ATTRIBUTE_from": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/@from",
    "cei_dateRange_ATTRIBUTE_to": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/@to"
    }

def parse_cei(cei_path):
    with open(cei_path, "rb") as f:
        root = etree.parse(f).getroot()
        data = {}
        for key, xpath_expr in xpath_expressions.items():
            result = root.xpath(xpath_expr, namespaces=namespaces, smart_strings=False)
            if len(result) == 1:
                data[key] = result[0] or None
            else:
                data[key] = result or None
        return data


if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    charter_paths = get_path_list(args.fsdb_dir, args.cei_filename, amount=100)
    for charter_path in tqdm.tqdm(charter_paths):
        data = parse_cei(charter_path)
        json.dump(data, open(f"{Path(charter_path).parent}/{args.output_filename}","w"))