import os
from pathlib import Path


def get_cei_paths(charter_dir, namespaces, extension):
    for entry in os.scandir(charter_dir):
        if entry.is_file() and entry.name.endswith(extension):
            yield Path(entry.path)
        elif entry.is_dir():
            yield from get_cei_paths(entry.path)
        else:
            continue


# def get_names_from_charter_xml(xml)


# def get_charter_xml_path_elements(


# def leech_charter_xml(charter_xml, root_dir, namespaces, extension):
#     with open(charter_xml, "rb") as f:

        

#     xml = str(urlopen(charter_url).read(), "utf8")



#     archive_name, fond_name, charter_atomid = get_names_from_charter_xml(xml)
#     archive_name, fond_name, charter_name = get_charter_path_elements(archive_name, fond_name, charter_atomid)

#     charter_full_path=f"{root_dir}/{archive_name}/{fond_name}/{charter_name}"
#     Path(charter_full_path).mkdir(parents=True, exist_ok=True)

#     store_charter(charter_html=charter_html, charter_full_path=charter_full_path, charter_atomid=charter_atomid)