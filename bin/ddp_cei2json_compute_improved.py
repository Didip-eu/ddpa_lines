#!/usr/bin/env python3
""" TODO (optional): Map attributes to elements with second mapping instead of long xpaths; test speeds - expectation: should be faster
    TODO: cei.xml vs CH.cei.xml?
    TODO: realize additional selectors, possibly with gui or other flexible mapping generator
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

from ddp_util import get_path_list

p = {
    "fsdb_root": "./misc/1000_CVCharters/",
    "cei_filename": "cei.xml",
    "output_filename":"charter.cei2json2.json"
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
    #"all_cei_date": "//cei:date/text()",
    #"all_cei_date_full": "//cei:date/descendant-or-self::node()/text()"
    }

def parse_cei_dates(cei_path):
    with open(cei_path, "rb") as f:
        root = etree.parse(f).getroot()
        data = {}
        for key, xpath_expr in xpath_expressions.items():
            result = root.xpath(xpath_expr, namespaces=namespaces, smart_strings=False)
            if len(result) == 1:
                data[key] = result[0] or None
            else:
                data[key] = result or None
        valid_dates = sorted(list([str(d).strip() for d in data.values() if d is not None]))
        return valid_dates

month2num = {
    "janner":1,
    "januar":1,
    "leden":1, # Czeck?
    "gennaio":1, # Italian?
    "i":1,
    "feb.":2,
    "februar":2,
    "unor":2, # Czeck?
    "ii":2,
    "marz":3,
    "marzo":3, # Italian?
    "brezen":3, # Czeck?
    "iii":3,
    "april":4,
    "iv":4,
    "duben":4, # Czeck? https://en.wikipedia.org/wiki/Slavic_calendar
    "mai":5,
    "maggio":5, # Italian?
    "v":5,
    "kveten": 5, # Hungarian?
    "juni":6,
    "vi":6,
    "juli":7,
    "luglio":7, # Italian?
    "cervenec":7, # Czeck?
    "vii":7,
    "cerven":6, # Czeck?
    "august":8,
    "srpen":8, # Czeck?
    "viii":8,
    "agosto":8, # Italian?
    "september":9,
    "settembre":9, # Italian?
    "septiembre":9, # Spanish?
    "zari":9, # Czeck?
    "ix":9,
    "oktober":10,
    "ottobre":10, # Italian?
    "okt":10,
    "x":10,
    "rijen":10, # Czeck?
    "november":11,
    "xi":11,
    "listopad":11, # Czeck?
    "dezember":12,
    "dicembre":12, # Italian?
    "xii":12,
    "prosinec":12, # Czeck?
}

def remove_ambiguous_9(*date_tuple):
    assert len(date_tuple)==3
    if date_tuple[0]==9999:
        date_tuple = (0, date_tuple[1], date_tuple[2])
    if date_tuple[1]==99:
        date_tuple = (date_tuple[0], 0, date_tuple[2])
    if date_tuple[2]==99:
        date_tuple = (date_tuple[0], date_tuple[1], 0)
    return date_tuple

def parse_date(date_str):
    date_str = anyascii.anyascii(date_str).lower() # TODO (anguelos) can we remove anyascii dependency?
    date_str = date_str.replace("wohl","") # We assume all is aproximate.
    date_str = " ".join(date_str.split()) # Remove extra spaces.
    if re.match("^[0-9]{7}$", date_str): # YYYMMDD we cant really know what is what but on 1000 Charters, that makes sence eg:25c52625b0576a7eec1a573cda314327/cei.xml
        return remove_ambiguous_9(int(date_str[:3]), int(date_str[3:5]), int(date_str[5:7]))
    
    elif re.match("^1[0-9]{7}$", date_str): # 1YYYMMDD assuming 1000-1999
        return remove_ambiguous_9(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    
    elif re.match("^[0-9]{4}1[0-9]{3}$", date_str):
        return remove_ambiguous_9(int(date_str[4:]), int(date_str[4:6]), int(date_str[6:8]))

    elif re.match("^[0-9\-,\.\s]{10}$", date_str):
        date = re.split("\-|\.|,|\s",date_str)
        if len(date)==3 and len(date[2])in (3,4):
            date = date[::-1]
        if len(date[0])==4 and date[0][0]=="1":
            return remove_ambiguous_9(int(date[0]), int(date[1]), int(date[2]))
        else:
            return f"Unparsed_V1: '{date_str}', {repr(date)}"

    elif re.match("^[0-9]+\.[0-9]+\.[0-9]+$", date_str):
        date = re.split("\.",date_str)
        if len(date)==3 and len(date[2]) in (3,4):
            date = date[::-1]
        if len(date[0]) in (3,4) and len(date[1]) in (1,2) and len(date[2]) in (1,2):
            return remove_ambiguous_9(int(date[0]), int(date[1]), int(date[2]))
        else:
            return f"Unparsed_V2: '{date_str}', {repr(date)}"
        
    elif re.match("^[0-9]*9{4}[0-9]*$", date_str):  # The year is unknow, the date is broken nomater what.
        return remove_ambiguous_9(0, 0, 0)  # TODO: (do we really care about months or days without years?)
    
    elif re.match("^[0-9][0-9]?\.\s+[a-z]+\.?\s+[0-9]{3}[0-9]?$", date_str):  # Czeck dates.
        date_list=date_str.split()
        date_list[1]=month2num[date_list[1]]
        date_list[0], date_list[2]=int(date_list[2]), int(date_list[0].replace(".",""))
        return remove_ambiguous_9(*tuple(date_list))
    elif re.match("^[0-9]{3}[0-9]?\s+[a-z]+\.?\s+[0-9][0-9]?\.?$", date_str): # EG '1288 dezember 22.'
        date_list=date_str.split()
        date_list[1]=month2num[date_list[1]]
        date_list[0], date_list[2]=int(date_list[0]), int(date_list[2].replace(".",""))
        return remove_ambiguous_9(*tuple(date_list))

    elif re.match("^[0-9]{4}$", date_str): # Only year.
        return remove_ambiguous_9(int(date_str), 0, 0)

    else:
        return f"Unparsed_VALL: '{date_str}'"


if __name__ == "__main__":
    args, _ = fargv.fargv(p)
    t=time.time()
    charter_paths = glob.glob(f"{args.fsdb_root}/*/*/*/{args.cei_filename}")
    all_dates = []
    wd = os.getcwd()
    for charter_path in tqdm.tqdm(charter_paths):
        dates = parse_cei_dates(charter_path)
        dates = [str(parse_date(d)) for d in dates]
        date_charters = [f"{d}\tfile://{wd}/{charter_path}" for d in dates]
        all_dates+=(date_charters)
        #json.dump(data, open(f"{Path(charter_path).parent}/{args.output_filename}","w"))
    print(f"Computer {len(charter_paths)} charters, {time.time()-t:.5} msec.")
    print("\n".join(sorted(all_dates)))
