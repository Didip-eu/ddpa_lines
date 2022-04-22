#!/usr/bin/env python3
import urllib
from urllib.request import urlopen


sheet_ids_gids_names=[
("1gnIb-jlrx_6hUPLe9mrP_sfcA6TbUhUpDyqKE4HY9cA", "0", "StFlorian"),
("1gnIb-jlrx_6hUPLe9mrP_sfcA6TbUhUpDyqKE4HY9cA", "513762880", "Heiligenkreuz"),
("1gnIb-jlrx_6hUPLe9mrP_sfcA6TbUhUpDyqKE4HY9cA", "814937970", "NapoliAdSSAgostinoM"),
        ]
for key, gid, name in sheet_ids_gids_names:
    url = f"https://docs.google.com/spreadsheets/d/{key}/export?format=csv&gid={gid}"
    with urlopen(url) as conn:
        charter_urls = [line.split(',')[2] for line in  str(conn.read(),"utf8").strip().split("\n")[1:] if len(line.split(',')[2].strip())]
        print("\n".join(charter_urls))


