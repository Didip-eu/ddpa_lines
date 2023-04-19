from pathlib import Path
import hashlib


db_root = "./"
monasterium_url_root = "https://www.monasterium.net/mom/"
collections_archive_name = "COLLECTIONS"
trunc_md5 = 16

def chatomid_to_url(atomid, root=monasterium_url_root):
    parts = atomid.split("/")                                                                                                                                  
    if len(parts) == 5:
        return f"{root}{parts[2]}/{parts[3]}/{parts[4]}/charter"
    elif len(parts) == 4:
        return f"{root}{parts[2]}/{parts[3]}/charter"
    else:
        raise ValueError("Invalid atom_id length.")


def decompose_chatomid(chatomid):
    """Infers the atom ids of the supercuration (archive/COLLECTIONS) and curation (fond/collection) 
    from a charters atomid
    """
    parts = chatomid.split("/")
    try:
        assert parts[:2] == (['tag:www.monasterium.net,2011:', 'charter'])
    except AssertionError:
        raise ValueError("atom-id is not well-formed.")
    if len(parts) == 5:
        supercuration_id = f"{parts[0]}/archive/{parts[2]}"
        curation_id = f"{parts[0]}/fond/{parts[2]}/{parts[3]}"
    elif len(parts) == 4:
        supercuration_id = "COLLECTIONS"
        curation_id = f"{parts[0]}/collection/{parts[2]}"
    return parts, supercuration_id, curation_id


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
    parts = url.split("/")
    if len(parts) == 8: #archive
        return f"tag:www.monasterium.net,2011:/charter/{parts[-4]}/{parts[-3]}/{parts[-2]}"
    elif len(parts) == 7: #collection
        return f"tag:www.monasterium.net,2011:/charter/{parts[-3]}/{parts[-2]}"
    else:
        raise ValueError

#chatomid_to_fond_id(chatomid)?
#chatomids_to_ (supercuration) url and vice versa? (requires consistent atomids)

def url_to_path(url):
    raise NotImplementedError

def path_to_atomid(path):
    return open(f"{path}/atomid.txt").read()


def path_to_url(path):
    return open(f"{path}/url.txt").read()

