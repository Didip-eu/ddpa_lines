# !/usr/bin/env python3

"""
Leeches Monasterium's backend xml-database files to construct target file structure system.
"""

# TODO: use md5 and hashing for substituting screwed strings
# TODO: fix permissions or make choices available in fargv for different file writing permission modes (https://stackoverflow.com/questions/5231901/permission-problems-when-creating-a-dir-with-os-makedirs-in-python)
# TODO: move some functions to ddp_util
# TODO: add "url2path_idx_path":"{root_dir}/url2path_idx.pickle" + code; checking dependencies etc.
# TODO: discuss in group:
# - if we include data types in function documentation, we should do it consistently, right? what is best practice?
# - convention for docstring writing, see https://www.datacamp.com/tutorial/docstrings-python; and include datatypes

import os
import random
from pathlib import Path
from pathlib import PurePosixPath
from pprint import pprint
import shutil
from typing import Dict, Generator, List
from lxml import etree
import fargv
from tqdm import tqdm


def get_path_generator(directory: str, file_extension: str) -> Generator:
    """
    Returns Generator of file paths in (sub)directories.
    @param directory: directory of monasterium xml files as a string
    @param file_extension: specifies file type, monasterium files would be .cei.xml
    @return Generator with file paths
    """
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(file_extension):
            yield Path(entry.path)
        elif entry.is_dir():
            yield from get_path_generator(entry.path, file_extension)
        else:
            continue


def get_path_list(directory: str, file_extension: str) -> List[str]:
    """
    Returns List containing file paths.
    @param directory: directory of monasterium xml files as a string
    @param file_extension: specifies file type, monasterium files would be .cei.xml
    @return List with file paths
    """
    paths = [f"{PurePosixPath(path)}" for path in get_path_generator(directory, file_extension)]
    return random.sample(paths, 1000)
    # return paths


def get_atomid_dict(paths: List[str]) -> Dict[str, str]:
    """
    Returns unique ids from xml-cei files.
    @param paths: a List containing paths to xml files, which contain an atom id
    @return Dict with path as key and atomd id as value
    """
    pprint("Parsing .cei.xml for atomids.")
    file_locations = {}
    for file in tqdm(paths):
        with open(file, 'r', encoding='utf-8') as current:
            tree = etree.parse(current)
            root = tree.getroot()
            file_locations[file] = root[0].text
    return file_locations


def move_charter_files(locations: Dict[str, str], target_directory: str):
    """
    For Archive Material: Copies folder and their xml contents from input folder to target
    Builds (sub)directory structure at target based on (length of) atomids.
    @param locations: Dict with path as key and atomd id as value
    @param target_directory: string specifying the target directory; where to create folders
    """
    pprint("Building target directories; copying files.")
    for path, atomid in tqdm(locations.items()):
        parts = atomid.split("/")
        if len(parts) == 5:
            target_path = f"{target_directory}/{parts[2]}/{parts[3]}/{parts[4]}"
            #print(target_path)
            os.makedirs(target_path, exist_ok=True)
            shutil.copy(path, target_path)
        else:
            target_path = f"{target_directory}/COLLECTIONS/{parts[2]}/{parts[3]}"
            #print(target_path)
            os.makedirs(target_path, exist_ok=True)
            shutil.copy(path, target_path)


def move_collection_files(locations: Dict[str, str], target_directory: str):
    """
    For Collection Material: Copies folder and their xml contents from input folder to target
    Builds (sub)directory structure at target based on (length of) atomids.
    @param locations: Dict with path as key and atomd id as value
    @param target_directory: string specifying the target directory; where to create folders
    """
    pprint("Building target directories; copying files.")
    for path, atomid in tqdm(locations.items()):
        parts = atomid.split("/")
        target_path = f"{target_directory}/COLLECTIONS/{parts[2]}"
        os.makedirs(target_path, exist_ok=True)
        shutil.copy(path, target_path)


if __name__ == "__main__":
    p = {
        "root_dir": ".",
        "charter_dir": "{root_dir}/data/db/mom-data/metadata.charter.public",
        "collection_dir": "{root_dir}/data/db/mom-data/metadata.collection.public",
        "archive_dir": "{root_dir}/data/db/mom-data/metadata.archive.public",
        "target_dir": "{root_dir}/data/tmp/data/leech_db"
        # "file_ext": ".cei.xml"
    }

    # params
    args, _ = fargv.fargv(p)

    # charters
    charter_paths = get_path_list(args.charter_dir, ".cei.xml")
    atom_id_dict = get_atomid_dict(charter_paths)
    move_charter_files(atom_id_dict, args.target_dir)

    # collections
    collection_paths = get_path_list(args.collection_dir, ".cei.xml")
    atom_id_dict = get_atomid_dict(collection_paths)
    move_collection_files(atom_id_dict, args.target_dir)

    # TODO: make same structure for archive, fond, then check how they can fit in one function (move files)

    # archives
    # put ead/preferences info etc. into directory structure (so querying for parent folder is possible
    # archive_paths = get_path_list(args.archive_dir, ".cei.xml")

    # fonds
    # here we find the image base (name of fond.preferences.xml ..)
