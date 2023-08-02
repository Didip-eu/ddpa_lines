#!/usr/bin/env python3
<<<<<<< HEAD
"""Used to extract dates from CEI files.
Ideally this script is used to create a json file with all the dates from the CEI files and their desired outputs for dates. 
TODO (optional): Map attributes to elements with second mapping instead of long xpaths; test speeds - expectation: should be faster
This program is used to create a dataset of dates from the CEI files and the cropped images of the charters.
=======
""" TODO (optional): Map attributes to elements with second mapping instead of long xpaths; test speeds - expectation: should be faster
    TODO: cei.xml vs CH.cei.xml?
    TODO: realize additional selectors, possibly with gui or other flexible mapping generator
>>>>>>> 75ede8cc98f614097e6bf9ca20a9172ee2b9c174
"""

import json
import tqdm
import fargv
from lxml import etree
from pathlib import Path
import glob
import time
import re
import os
import anyascii
import json
from ddp_util import infer_date

p = {
    "fsdb_root": "./misc/1000_CVCharters/",
    "cei_filename": "cei.xml",
    "image_subpath": "*.seals.crops/*.Img_WritableArea.*",
}

namespaces = {"atom": "http://www.w3.org/2005/Atom",
              "cei": "http://www.monasterium.net/NS/cei"}
xpath_expressions = {
    "cei_date": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/text()",
    "cei_date_ATTRIBUTE_value": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@value",
    "cei_date_ATTRIBUTE_notBefore": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@notBefore",
    "cei_date_ATTRIBUTE_notAfter": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:date/@notAfter",
    "cei_dateRange": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/text()",
    "cei_dateRange_ATTRIBUTE_from": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/@from",
    "cei_dateRange_ATTRIBUTE_to": "/atom:entry/atom:content/cei:text/cei:body/cei:chDesc/cei:issued/cei:dateRange/@to"
<<<<<<< HEAD
}

=======
    #"all_cei_date": "//cei:date/text()",
    #"all_cei_date_full": "//cei:date/descendant-or-self::node()/text()"
    }
>>>>>>> 75ede8cc98f614097e6bf9ca20a9172ee2b9c174

def parse_cei_dates(cei_path):
    with open(cei_path, "rb") as f:
        root = etree.parse(f).getroot()
        data = {}
        for key, xpath_expr in xpath_expressions.items():
            result = root.xpath(
                xpath_expr, namespaces=namespaces, smart_strings=False)
            if len(result) == 1:
                data[key] = result[0] or None
            else:
                data[key] = result or None
        valid_dates = sorted(list([str(d).strip()
                             for d in data.values() if d is not None]))
        return valid_dates


def create_date_testdata(cei_path):
    with open(cei_path, "rb") as f:
        root = etree.parse(f).getroot()
        data = {}
        for key, xpath_expr in xpath_expressions.items():
            result = root.xpath(
                xpath_expr, namespaces=namespaces, smart_strings=False)
            if len(result) == 1:
                data[key] = result[0] or None
            else:
                data[key] = result or None
        valid_dates = sorted(list([str(d).strip()
                             for d in data.values() if d is not None]))
        return valid_dates


month2num = {
    "janner": 1,
    "januar": 1,
    "leden": 1,  # Czeck?
    "gennaio": 1,  # Italian?
    "i": 1,
    "feb.": 2,
    "februar": 2,
    "unor": 2,  # Czeck?
    "ii": 2,
    "marz": 3,
    "marzo": 3,  # Italian?
    "brezen": 3,  # Czeck?
    "iii": 3,
    "april": 4,
    "iv": 4,
    "duben": 4,  # Czeck? https://en.wikipedia.org/wiki/Slavic_calendar
    "mai": 5,
    "maggio": 5,  # Italian?
    "v": 5,
    "kveten": 5,  # Hungarian?
    "juni": 6,
    "vi": 6,
    "juli": 7,
    "luglio": 7,  # Italian?
    "cervenec": 7,  # Czeck?
    "vii": 7,
    "cerven": 6,  # Czeck?
    "august": 8,
    "srpen": 8,  # Czeck?
    "viii": 8,
    "agosto": 8,  # Italian?
    "september": 9,
    "settembre": 9,  # Italian?
    "septiembre": 9,  # Spanish?
    "zari": 9,  # Czeck?
    "ix": 9,
    "oktober": 10,
    "ottobre": 10,  # Italian?
    "okt": 10,
    "x": 10,
    "rijen": 10,  # Czeck?
    "november": 11,
    "xi": 11,
    "listopad": 11,  # Czeck?
    "dezember": 12,
    "dicembre": 12,  # Italian?
    "xii": 12,
    "prosinec": 12,  # Czeck?
}


def remove_ambiguous_9(*date_tuple):
    assert len(date_tuple) == 3
    if date_tuple[0] == 9999:
        date_tuple = (0, date_tuple[1], date_tuple[2])
    if date_tuple[1] == 99:
        date_tuple = (date_tuple[0], 0, date_tuple[2])
    if date_tuple[2] == 99:
        date_tuple = (date_tuple[0], date_tuple[1], 0)
    return date_tuple


def is_plausible_date(date_tuple):
    if date_tuple[0] < 0 or date_tuple[0] > 2100:
        return False
    if date_tuple[1] < 0 or date_tuple[1] > 12:
        return False
    if date_tuple[2] < 0 or date_tuple[2] > 31:
        return False
    return True


def parse_date(date_str):
    # TODO (anguelos) can we remove anyascii dependency?
    date_str = anyascii.anyascii(date_str).lower()
    date_str = date_str.replace("wohl", "")  # We assume all is aproximate.
    date_str = " ".join(date_str.split())  # Remove extra spaces.
    # YYYMMDD we cant really know what is what but on 1000 Charters, that makes sence eg:25c52625b0576a7eec1a573cda314327/cei.xml
    if re.match("^[0-9]{7}$", date_str):
        date = remove_ambiguous_9(int(date_str[:3]), int(
            date_str[3:5]), int(date_str[5:7]))
        if is_plausible_date(date):
            return date

    if re.match("^1[0-9]{7}$", date_str):  # 1YYYMMDD assuming 1000-1999
        date = remove_ambiguous_9(int(date_str[:4]), int(
            date_str[4:6]), int(date_str[6:8]))
        if is_plausible_date(date):
            return date

    if re.match("^[0-9]{4}1[0-9]{3}$", date_str):
        date = remove_ambiguous_9(int(date_str[4:]), int(
            date_str[4:6]), int(date_str[6:8]))
        if is_plausible_date(date):
            return date

    if re.match("^[0-9\-,\.\s]{10}$", date_str):
        date = re.split("\-|\.|,|\s", date_str)
        if len(date) == 3 and len(date[2]) in (3, 4):
            date = date[::-1]
        if len(date[0]) == 4 and date[0][0] == "1":
            date = remove_ambiguous_9(int(date[0]), int(date[1]), int(date[2]))
            if is_plausible_date(date):
                return date
        else:
            return f"Unparsed_V1: '{date_str}', {repr(date)}"

    if re.match("^[0-9]+\.[0-9]+\.[0-9]+$", date_str):
        date = re.split("\.", date_str)
        if len(date) == 3 and len(date[2]) in (3, 4):
            date = date[::-1]
        if len(date[0]) in (3, 4) and len(date[1]) in (1, 2) and len(date[2]) in (1, 2):
            date = remove_ambiguous_9(int(date[0]), int(date[1]), int(date[2]))
            if is_plausible_date(date):
                return date
        else:
            return f"Unparsed_V2: '{date_str}', {repr(date)}"

    # The year is unknow, the date is broken nomater what.
    if re.match("^[0-9]*9{4}[0-9]*$", date_str):
        # TODO: (do we really care about months or days without years?)
        date = remove_ambiguous_9(0, 0, 0)
        if is_plausible_date(date):
            return date

    # Czeck dates.
    if re.match("^[0-9][0-9]?\.\s+[a-z]+\.?\s+[0-9]{3}[0-9]?$", date_str):
        date_list = date_str.split()
        date_list[1] = month2num[date_list[1]]
        date_list[0], date_list[2] = int(date_list[2]), int(
            date_list[0].replace(".", ""))
        date = remove_ambiguous_9(*tuple(date_list))
        if is_plausible_date(date):
            return date
    # EG '1288 dezember 22.'
    if re.match("^[0-9]{3}[0-9]?\s+[a-z]+\.?\s+[0-9][0-9]?\.?$", date_str):
        date_list = date_str.split()
        date_list[1] = month2num[date_list[1]]
        date_list[0], date_list[2] = int(date_list[0]), int(
            date_list[2].replace(".", ""))
        date = remove_ambiguous_9(*tuple(date_list))
        if is_plausible_date(date):
            return date
    if re.match("^[0-9]{4}$", date_str):  # Only year.
        date = remove_ambiguous_9(int(date_str), 0, 0)
        if is_plausible_date(date):
            return date

    else:
        return f"Unparsed_VALL"


if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    t = time.time()
    charter_paths = glob.glob(f"{args.fsdb_root}/*/*/*/{args.cei_filename}")
    all_dates = []
    wd = os.getcwd()
    for charter_path in tqdm.tqdm(charter_paths):
        dates = create_date_testdata(charter_path)
        dates = [[d, infer_date(d, fail_quietly=True)] for d in dates]
        dates = [d for d in dates if isinstance(d[1], tuple)]
        for image in glob.glob(f"{Path(charter_path).parent}/{args.image_subpath}"):
            dates_and_images = [[image] + d for d in dates]
            print(dates_and_images)
            all_dates += (dates_and_images)
        # json.dump(data, open(f"{Path(charter_path).parent}/{args.output_filename}","w"))
    # print(f"Computer {len(charter_paths)} charters, {time.time()-t:.5} msec.")
    # print(json.dumps(all_dates, indent=2))
    print(
        "\n".join([f"{d[0]:6}, '{d[1][0]}': {list(d[1][1])}, {d[1][2]}" for d in enumerate(sorted(all_dates))]))
