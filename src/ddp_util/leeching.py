from cmath import atan
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import hashlib
from urllib.parse import unquote
from tqdm import tqdm
import csv
from io import StringIO
import re
import sys
import pickle
import traceback
import magic
import time
import json
from .namespace import chatomid_to_path


def get_extention(img_url):
    ext = img_url.split(".")[-1].lower()
    if ext in ["jpg", "png", "jpeg", "tif", "tiff"]:
        return ext
    else:
        buffer = urlopen(img_url).read(4096)
        magic_str = magic.from_buffer(buffer)
        all_magic_patterns = re.findall("[A-Z]+ image data", magic_str)
        if len(all_magic_patterns)==1:
            ext = all_magic_patterns[0][:-11].lower()
        if ext in ["jpg", "png", "jpeg", "tif", "tiff"]:
            return ext
    raise ValueError

archives_root = "https://www.monasterium.net/mom/fonds"

def get_archive_urls(archive_webpage_url):
    archive_list_html = str(urlopen(archive_webpage_url).read(), "utf8")
    soup = BeautifulSoup(archive_list_html, "html.parser")
    archive_urls = ["https://www.monasterium.net"+tag.attrs.get("href") for tag in soup.find_all("a") if tag.attrs.get("href", "").endswith("/archive")]
    return archive_urls


def get_fonds_from_archive(archive_url):
    assert archive_url.endswith("/archive")
    #print(f"a2f: {archive_url}")
    fond_list_html = str(urlopen(archive_url).read(), "utf8")
    soup = BeautifulSoup(fond_list_html, "html.parser")
    fond_urls = [tag.attrs.get("href") for tag in soup.find_all("a") if tag.attrs.get("href", "").endswith("/fond")]
    fond_urls = [f"{url_base}/{fond_url}" for fond_url in fond_urls]
    return list(set(fond_urls))


def get_charters_from_fond(fond_url):
    assert fond_url.endswith("/fond")
    fond_list_html = str(urlopen(fond_url).read(), "utf8")
    soup = BeautifulSoup(fond_list_html, "html.parser")
    block_urls = [f"{fond_url}{tag.attrs.get('href')}" for tag in soup.find_all("a") if tag.attrs.get("href", "").startswith("?block")]
    if len(block_urls) > 0:
        for block_url in block_urls:
            charter_list_html = str(urlopen(fond_url).read(), "utf8")
            soup = BeautifulSoup(charter_list_html, "html.parser")
            charter_urls = [tag.attrs.get("href") for tag in soup.find_all("a") if tag.attrs.get("href", "").endswith("/charter")]
    else:
        charter_urls = [tag.attrs.get("href") for tag in soup.find_all("a") if tag.attrs.get("href", "").endswith("/charter")]
    charter_urls = [f"{fond_url}{charter_url}" for charter_url in charter_urls]
    return sorted(set(charter_urls))


def get_names_from_charter_html(html:str):
    href_list = list(BeautifulSoup(html).find_all("a"))

    archive_re = re.compile("mom/[0-9A-Za-z\-]+/archive")
    archive_hrefs = [a.attrs["href"] for a in href_list if len(archive_re.findall(a.attrs.get("href", "")))>0]

    fond_re = re.compile("mom/[0-9A-Za-z\-]+/.*/fond")
    fond_hrefs = [a.attrs["href"] for a in href_list if len(fond_re.findall(a.attrs.get("href", ""))) > 0]

    collection_re = re.compile("mom/[0-9A-Za-z\-]+/collection")
    collection_hrefs = [a.attrs["href"] for a in href_list if len(collection_re.findall(a.attrs.get("href", "")))>0]

    if len(set(collection_hrefs)) == 1 and len(set(archive_hrefs)) == 0 and len(set(fond_hrefs)) == 0:
        fond_name = collection_hrefs[0].replace("/mom/", "").replace("/collection", "") # TODO (anguelos) name or whole atomid
        archive_name = "COLLECTIONS"
    elif len(set(collection_hrefs)) == 0 and len(set(archive_hrefs)) == 1 and len(set(fond_hrefs)) == 1:
        fond_name = fond_hrefs[0].split("/fond")[0].split("/")[-1] # TODO (anguelos) name or whole atomid
        archive_name = archive_hrefs[0].replace("/mom/", "").replace("/archive", "")  # TODO (anguelos) name or whole atomid
    else:
        print("<<<<<<")
        #print(html)
        print("HREFS:\n","\n".join([a.attrs["href"] for a in href_list if len(fond_re.findall(a.attrs.get("href", ""))) > 0]))
        print(repr(collection_hrefs))
        print(repr(archive_hrefs))
        print(repr(fond_hrefs))
        print(">>>>>>")
        sys.exit(1)
        raise ValueError # html page not a parsable charter

    pdf_export_href_list = [a for a in href_list if a.attrs.get("target", "") == "blank"]
    assert len(pdf_export_href_list) == 1 # hopefully we isolated a single href
    assert pdf_export_href_list[0].text.replace(" ","").lower() == "pdf-export"
    charter_atomid = pdf_export_href_list[0].attrs["href"]
    charter_atomid = charter_atomid.split("?id=")[1].split("&")[0]
    charter_atomid = unquote(charter_atomid)

    return archive_name, fond_name, charter_atomid


def get_charter_path_elements(archive_name, fond_name, charter_atomid, trunc_md5=0, verbose=0):
    valid_names = re.compile(r'[A-Za-z0-9_\-]+')
    if valid_names.fullmatch(archive_name):
        archive_path = archive_name
    else:
        archive_path = hashlib.md5(archive_name.encode('utf-8')).hexdigest()[trunc_md5:]
        if verbose>2:
            print(f"Replacing archive {archive_name} with {archive_path}", file=sys.stderr)
    if valid_names.fullmatch(fond_name):
        fond_path = fond_name
    else:
        fond_path = hashlib.md5(fond_name.encode('utf-8')).hexdigest()[trunc_md5:]
        if verbose > 2:
            print(f"Replacing fond {fond_name} with {fond_path}", file=sys.stderr)
    if valid_names.fullmatch(charter_atomid):
        charter_path = charter_atomid
    else:
        charter_path = hashlib.md5(charter_atomid.encode('utf-8')).hexdigest()[trunc_md5:]
        if verbose > 2:
            print(f"Replacing charter {charter_atomid} with {charter_path}", file=sys.stderr)
    return archive_path, fond_path, charter_path


def download_charter(charter_full_path, url=""):
    if url == "":
        assert Path(f"{charter_full_path}/url.txt").is_file()
        url = open(f"{charter_full_path}/url.txt").read()
    else:
        # if there is a URL in the folder and they gave us one, thay must agree
        assert not(Path(f"{charter_full_path}/url.txt").is_file()) or url == str(open(f"{charter_full_path}/url.txt").read(),"utf-8")
    charter_html = str(urlopen(url).read(), "utf8")
    store_charter(charter_html, charter_full_path)


def store_charter(charter_html, charter_full_path, charter_atomid=""):
    """Store the crawll outcome into an existing folder
    """
    assert Path(charter_full_path).is_dir()
    soup = BeautifulSoup(charter_html, "html.parser")
    if charter_atomid == "":
        _, _, charter_atomid = get_names_from_charter_html(charter_html)

    image_urls = [tag.attrs.get("title") for tag in soup.find_all("a") if tag.attrs.get("class", "") == ["imageLink"]]

    open(f"{charter_full_path}/url.txt", "w").write(charter_url)
    open(f"{charter_full_path}/original.html", "w").write(charter_html)
    open(f"{charter_full_path}/atom_id.txt", "w").write(charter_atomid)

    relinked_images_html = charter_html
    failed = []

    cei_urls = [tag.attrs.get("href") for tag in soup.find_all("a") if tag.attrs.get("target", "_blank") and tag.attrs.get("href", "").lower().endswith(".cei.xml")]
    try:
        assert len(cei_urls) == 1
        cei_absolute_url = f"http://monasterium.net{cei_urls[0]}"
        xml_str = str(urlopen(cei_absolute_url).read(), "utf8")
        relinked_images_html = relinked_images_html.replace(cei_urls[0], f"cei.xml")
        open(f"{charter_full_path}/cei.xml", "w").write(xml_str)
    except HTTPError:
        print(f"charter {charter_url} Failed to download CEI : {cei_urls[0]}")
        failed.append(cei_urls[0])

    imgname2imgurls = {}
    for n, img_url in enumerate(image_urls):
        img_url = img_url.replace(" ", "%20")
        ext = get_extention(img_url)
        #ext = img_url.split(".")[-1].lower()
        try:
            img_bytes = urlopen(img_url).read()
            md5_str = hashlib.md5(img_bytes).hexdigest()
            open(f"{charter_full_path}/{md5_str}.{ext}", "wb").write(img_bytes)
            relinked_images_html = relinked_images_html.replace(img_url, f"{md5_str}.{ext}")
            imgname2imgurls[f"{md5_str}.{ext}"] = img_url
        except HTTPError:
            print(f"charter {charter_url} Failed to download : {img_url}")
            failed.append(img_url)

    
    json.dump(imgname2imgurls, open(f"{charter_full_path}/image_urls.json", "w"), indent=2)
    open(f"{charter_full_path}/index.html", "w").write(relinked_images_html)

    if len(failed) == 0:
        open(f"{charter_full_path}/download_complete.marker", "w").write("") # same as check at the beginning of the function
    else:
        open(f"{charter_full_path}/failed.txt", "w").write("\n".join([f"{time.time()}, {f} " for f in failed]))


def leech_charter(charter_url, root, url2path_idx={}, url2path_idx_path="", verbose=0):
    if charter_url in url2path_idx:
        if Path(f"{url2path_idx[charter_url]}/download_complete.marker").is_file():
            if verbose > 2:
                print(f"{url2path_idx[charter_url]} found! skipping", file=sys.stderr)
            return
    charter_html = str(urlopen(charter_url).read(), "utf8")
    archive_name, fond_name, charter_atomid = get_names_from_charter_html(charter_html)
    archive_name, fond_name, charter_name = get_charter_path_elements(archive_name, fond_name, charter_atomid)

    charter_full_path=f"{root}/{archive_name}/{fond_name}/{charter_name}"
    Path(charter_full_path).mkdir(parents=True, exist_ok=True)

    store_charter(charter_html=charter_html, charter_full_path=charter_full_path, charter_atomid=charter_atomid)

    url2path_idx[charter_url] = charter_full_path

    if url2path_idx_path != "":
        pickle.dump(url2path_idx, open(url2path_idx_path, "wb"))


def leech_spreadsheet(sheet_key, gid, name, root, url2path_idx={}, url2path_idx_path="", verbose=0):
    """Leeches a google speadsheet
    """
    #if Path(f"{root}/{name}/download_complete.marker").is_file():
    #    print(f"{root} found! skipping", file=sys.stderr)
    #    return

    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid={gid}"
    failed = []
    with urlopen(sheet_url) as conn:
        csv_data = str(conn.read(), "utf8")
        pbar = tqdm(enumerate(list(csv.reader(StringIO(csv_data), delimiter=","))[1:]), desc=f"Leeching {name}")
        for n, row in pbar:
            charter_url = row[2]
            pbar.set_description(f"Leeching {charter_url} ")
            pbar.refresh()
            try:
                leech_charter(charter_url, root=f"{root}/{name}",
                        url2path_idx=url2path_idx, url2path_idx_path=url2path_idx_path, verbose=verbose)
            except Exception as e:
                stack_trace = traceback.format_exc()
                if verbose > 0:
                    print(f"\n\nCharter {charter_url} FAILED! Continuing\nException:{repr(e)}\n\n{stack_trace}\n\n",
                        file=sys.stderr)
                failed.append((charter_url, stack_trace))

        #open(f"{root}/{name}/download_complete.marker", "w").write("")




def leech_csv(csv_path, name, root, url2path_idx={}, url2path_idx_path="", verbose=0):
    """Leeches a speadsheet
    """
    #if Path(f"{root}/{name}/download_complete.marker").is_file():
    #    print(f"{root} found! skipping", file=sys.stderr)
    #    return
    assert csv_path.lower().endswith(".csv") and Path(csv_path).is_file()
    failed = []
    csv_data = open(csv_path, "r").read()
    pbar = tqdm(enumerate(list(csv.reader(StringIO(csv_data), delimiter=","))[1:]), desc=f"Leeching {name}")
    for n, row in pbar:
        charter_url = row[2]
        pbar.set_description(f"Leeching {charter_url} ")
        pbar.refresh()
        try:
            leech_charter(charter_url, root=f"{root}/{name}",
                    url2path_idx=url2path_idx, url2path_idx_path=url2path_idx_path, verbose=verbose)
        except Exception as e:
            stack_trace = traceback.format_exc()
            if verbose > 0:
                print(f"\n\nCharter {charter_url} FAILED! Continuing\nException:{repr(e)}\n\n{stack_trace}\n\n",
                    file=sys.stderr)
            failed.append((charter_url, stack_trace))
    open(f"{root}/failed.txt","w").write("\n".join(failed))