#!/usr/bin/env python3

"""

Segmentation line meta-data conversions.


"""
# stdlib
import sys
from pathlib import Path
import json
import re
import logging
import glob

# Didip
import fargv

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))

from seglib import seglib


#logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "line_converter",
    "seg_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.xml")]),
    "img_paths": set([]),
    "charter_dirs": set(["./"]),
    "output_format": [('json', 'xml'),"Output format: JSON ('json', the default) or PageXML ('xml'). Given an ouput format, the other option is the implicit input format."],
}

if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug(args)

    input_format = 'xml' if args.output_format == 'json' else 'json'

    # Assumption: any segmentation meta-data are tied to an existing image
    all_img_paths = list(sorted(args.img_paths))

    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir ) 
        if charter_dir_path.is_dir() and charter_dir_path.joinpath('CH.cei.xml').exists():
            charter_imgs = [str(f) for f in charter_dir_path.glob(f"*.img.*") ]
            all_img_paths += charter_imgs

        args.img_paths = list(all_img_paths)

    logger.debug( args.img_paths)

    for path in list( args.img_paths ):
       
        path = Path(path)

        input_file_basename = re.sub(r'\..+', f'.{input_format}', path.name )
        input_file_path = path.parent.joinpath(f'{input_file_basename}')

        if input_format == 'xml':

            segdict = seglib.segmentation_dict_from_xml( str( input_file_path ))

            output_file_path = input_file_path.with_suffix('.lines.gt.json')
            with open( output_file_path, 'w') as of:

                print( json.dumps( segdict, indent=4 ), file=of)
                logger.info("{} → {}".format(input_file_path, output_file_path ))
        else:
            logger.info("JSON → XML conversion not implemented yet.")
            sys.exit()


