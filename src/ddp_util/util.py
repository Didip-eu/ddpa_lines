from typing import Dict, Generator, List
import os
from pathlib import Path, PurePosixPath
from pprint import pprint
import hashlib
import random


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


def get_path_list(directory: str, file_extension: str, sample=False) -> List[str]: #TODO: enable sampling mode; refactor maybe
    """
    Returns List containing file paths.
    @param directory: directory of monasterium xml files as a string
    @param file_extension: specifies file type, monasterium files would be .cei.xml
    @return List with file paths
    """
    pprint(f"Scanning {directory} for files.")
    paths = [f"{PurePosixPath(path)}" for path in get_path_generator(directory, file_extension)]
    if sample:
        return random.sample(paths, int(round(len(paths)/1000))) #5%
    else:
        return paths


def to_md5(string, trunc_threshold=0): 
    md5sum = hashlib.md5(string.encode('utf-8')).hexdigest()[trunc_threshold:]
    return md5sum