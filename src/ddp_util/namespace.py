from pathlib import Path
import hashlib


db_root = "./"
monasterium_url_root = "https://www.monasterium.net/mom/"
collections_archive_name = "COLLECTIONS"
trunc_md5 = 16

def chatomid_to_url(atomid):
    parts = atomid.split("/")                                                                                                                                  
    if len(parts) == 5:
        return f"{monasterium_url_root}{parts[2]}/{parts[3]}/{parts[4]}/charter "
    elif len(parts) == 4:
        return f"{monasterium_url_root}{parts[2]}/{parts[3]}/charter "
    else:
        raise ValueError
        #print(f"Unusual structure found at '{atom_id}'")


def decompose_chatomid(chatomid):
    """Infers the atom ids of the archive and the fond from a charters atomid
    """
    splitted = chatomid.split("/")
    assert splitted[:2] == ("tag:www.monasterium.net,2011", "charter")
    if len(splitted) == 5: # ARCHIVE FOND
        fond_atomid = f"{splitted[0]}/fond/{splitted[3]}/{splitted[4]}"
        archive_atomid = f"{splitted[0]}/archive/{splitted[3]}"
    if len(splitted) == 4: # collection
        fond_atomid = f"{splitted[0]}/fond/{collections_archive_name}/{splitted[3]}"
        archive_atomid = f"{splitted[0]}/archive/{collections_archive_name}"
    else:
        raise ValueError
    return archive_atomid, fond_atomid, chatomid


def chatomid_to_pathtuple(chatomid):
    """Return the filesystem names of the path of a charter
    """
    archive_atomid, fond_atomid, _ = decompose_chatomid(chatomid)
    archive_name = archive_atomid.split("/")[-1]
    #fond_name = fond_atomid.split("/")[-1]
    fond_name = hashlib.md5(fond_atomid.encode('utf-8')).hexdigest()[trunc_md5:]
    charter_name = hashlib.md5(chatomid.encode('utf-8')).hexdigest()[trunc_md5:]
    return archive_name, fond_name, charter_name


def chatomid_to_path(chatomid, root=db_root):
    archive_name, fond_name, charter_name = chatomid_to_pathtuple(chatomid)
    return f"{root}/{archive_name}/{fond_name}/{charter_name}"


def url_to_chatomid(url):
    raise NotImplementedError


def url_to_path(url):
    raise NotImplementedError


def path_to_atomid(path):
    return open(f"{path}/atomid.txt").read()


def path_to_url(path):
    return open(f"{path}/url.txt").read()

