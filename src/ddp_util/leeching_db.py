"""goal: derive target file structure from mom-db dump (as it is)"""

import os
import random
from pathlib import Path
from pathlib import PurePosixPath
import xml.etree.ElementTree as ET
import re
import shutil


# list paths of all charters in directory
def get_charter_paths(charter_dir, file_ext):
    """
    returns generator of file paths
    """
    for entry in os.scandir(charter_dir):
        if entry.is_file() and entry.name.endswith(file_ext):
            yield Path(entry.path)
        elif entry.is_dir():
            yield from get_charter_paths(entry.path, file_ext)
        else:
            continue


def get_path_list(directory, file_ext):
    """returns List containing File Paths

    """
    paths = [f"{PurePosixPath(path)}" for path in get_charter_paths(directory, file_ext)]
    #return random.sample(paths, 100)
    return paths



def get_atom_id(paths):
    """
    returns unique ids from cei files
    @rtype Dict
    """
    atom_ids = {}
    for file in paths:
        with open(file, 'r', encoding='utf-8') as current:
            tree = ET.parse(current)
            root = tree.getroot()
            atom_ids[file] = root[0].text
    return atom_ids


def move_files(locations):
    """
    reads directory containg file path -> key and atomid -> value
    missing folders and files will be added to the target archive or collection
    """
    pattern = re.compile("^tag:[a-zA-Z0-9.,:-]+/[a-zA-Z0-9.,:-]+/[a-zA-Z0-9.,:-]+/"
                         "[a-zA-Z0-9.,:-]+$")
    for path, atom in locations.items():
        collection = "../../data/main/collections/"
        archive = "../../data/main/archive/"
        folder = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        if pattern.match(atom):
            if not os.path.exists((collection + "/" + folder)):
                os.mkdir(collection + "/" + folder)
        if pattern.match(atom):
            if not os.path.exists(collection + "/" + folder + "/" + filename):
                shutil.copyfile(path, (collection + "/" + folder + "/" + filename))
        elif not os.path.exists(archive + "/" + folder):
            os.mkdir(archive + "/" + folder)
        elif not os.path.exists(archive + "/" + folder + "/" + filename):
            shutil.copyfile(path, (archive + "/" + folder + "/" + filename))


def create_directories():
    """
    create collections / archives folders
    """
    if not os.path.exists("../../data/main/collections"):
        os.mkdir("../../data/main/collections")
    if not os.path.exists("../../data/main/archive/"):
        os.mkdir("../../data/main/archive/")


if __name__ == "__main__":
    charter_directory = "../../data/db/mom-data/metadata.charter.public"
    file_extension = ".cei.xml"
    create_directories()
    charter_paths = get_path_list(charter_directory, file_extension)
    atom_id_list = get_atom_id(charter_paths)
    move_files(atom_id_list)

# def get_names_from_charter_xml(xml)


# def get_charter_xml_path_elements(


# def leech_charter_xml(charter_xml, root_dir, namespaces, extension):
#     with open(charter_xml, "rb") as f:


#     xml = str(urlopen(charter_url).read(), "utf8")


#     archive_name, fond_name, charter_atomid = get_names_from_charter_xml(xml)
#     archive_name, fond_name, charter_name = get_charter_path_elements
#     (archive_name, fond_name, charter_atomid)

#     charter_full_path=f"{root_dir}/{archive_name}/{fond_name}/{charter_name}"
#     Path(charter_full_path).mkdir(parents=True, exist_ok=True)

#     store_charter(charter_html=charter_html,
#     charter_full_path=charter_full_path, charter_atomid=charter_atomid)
