"""
work in progress, together with compute_improved

TODO: Do reality check on scope of actual elements; later vs. schema

This joins some functionalities as used in
https://github.com/flamminger/2tei-validation/blob/main/src/describe_xml.py
https://github.com/anguelos/didipcv/blob/main/notebooks/XML2PD/XML2PD.ipynb
https://github.com/anguelos/didipcv/blob/main/notebooks/playground/playground.ipynb heading #Mapping

The code should later be divided better, and parts of it should go into an actual cei.py in the ddp_util module.

"""

import fargv
import bs4
import re
import json
import tqdm
from lxml import etree
from ddp_util import get_path_list
import re
import os
import pprint



def filter_xpath(xpath):
    """Filters an xpath by occurrence of an item order specification.
    """
    return re.sub(r"\[\d+\]", "", xpath)


def extract_xpaths_from_file(file_path, truncation):
    try:
        tree = etree.parse(file_path)
        elements = tree.xpath("//*")
        xpaths = set()

        for element in elements:
            xpath = tree.getpath(element)
            if "*" not in xpath:
                xpaths.add(filter_xpath(xpath)) if truncation == True else xpaths.add(xpath)

            attributes = element.attrib
            for attr, value in attributes.items():
                attr_xpath = f"{xpath}/@{attr}"
                if "*" not in attr_xpath:
                    xpaths.add(filter_xpath(attr_xpath)) if truncation == True else xpaths.add(xpath)

        return list(xpaths)

    except etree.XMLSyntaxError:
        #print(f"Error parsing XML file: {file_path}")
        return []

    
def extract_xpaths_from_file_paths(file_paths, truncation=True):
    xpaths = set()
    for file_path in file_paths:
        xpaths.update(extract_xpaths_from_file(file_path, truncation=truncation))

    return list(xpaths)


def create_xpath_dictionary(xpath_list, output_file):
    xpath_dict = {}
    sorted_xpaths = sorted(xpath_list, key=len)

    for xpath in sorted_xpaths:
        split = xpath.lstrip("/").split("/")
        key = split[-1]
        if key in xpath_dict:
            xpath_dict[f"{split[-2]}/{key}"] = xpath
        else:
            xpath_dict[key] = xpath
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    with open(output_file, "w") as json_file:
        json.dump(xpath_dict, json_file, indent=4)
    
    return xpath_dict


p = {
    "root_dir": ".",
    "helpers_dir": "{root_dir}/data/helpers",
    "fsdb_dir": "{root_dir}/data/leech_db",
    "cei_filename": "cei.xml",
    "output_filename":"charter.cei2json.json"
    }


def parse_cei(cei_path):
    dates = []

    return {"dates":sorted(dates)}


if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    paths = get_path_list(args.fsdb_dir, args.cei_filename)
    xpaths = extract_xpaths_from_file_paths(paths)
    xpath_dict = create_xpath_dictionary(xpaths, f"{args.helpers_dir}/mapping.json")    
    
"""     for charter_path in tqdm.tqdm(paths):
        data = parse_cei(f"{charter_path}/{args.cei_filename}")
        json.dump(data, open(f"{charter_path}/{args.output_filename}","w")) """