#!/usr/bin/env python3

"""

Segmentation line viewer.


"""
# stdlib
import sys
from pathlib import Path
import dataclasses
import json
import re
import logging
import glob

# 3rd party
from PIL import Image

# Didip
import fargv

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))

from seglib import seg_io

#logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
    "appname": "lines",
    #"img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
    "img_paths": set([]),
    "charter_dirs": set(["./"]),
    "mask_classes": [set(['Wr:OldText']), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
    "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>."],
    "colors": [2, "Number of colors in the palette."],
    "style": [("outline", "map"), "Display style: 'map' for pixel map (slow); 'outline' for polygon lines."],
}

if __name__ == "__main__":

    args, _ = fargv.fargv( p )
    logger.debug(args)

    all_img_paths = list(sorted(args.img_paths))
    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir ) 
        if charter_dir_path.is_dir() and charter_dir_path.joinpath('CH.cei.xml').exists():
            charter_images = [str(f) for f in charter_dir_path.glob("*.img.*") ]
            all_img_paths += charter_images

        args.img_paths = list(all_img_paths)

    logger.debug( args.img_paths)

    for path in list( args.img_paths ):
       
        path = Path(path)
        logger.info( path )

        stem = re.sub(r'\..+', '', path.name )

        # only for segmentation on Seals-detected regions
        region_segfile = re.sub(r'.img.jpg', args.region_segmentation_suffix, str(path) )
        
        with Image.open( path, 'r' ) as img:

            # segmentation metadata file
            output_file_path_wo_suffix = path.parent.joinpath( f'{stem}.{args.appname}.pred' )

            json_file_path = Path(f'{output_file_path_wo_suffix}.json')
            xml_file_path = Path(f'{output_file_path_wo_suffix}.xml')
            pt_file_path = Path(f'{output_file_path_wo_suffix}.pt')


            # look at existing files, choose the most recent one
            candidates = sorted([ (f, f.stat().st_mtime) for f in (json_file_path, xml_file_path, pt_file_path ) if f.exists() ], key=lambda x: x[1], reverse=True)

            if candidates == []:
                logger.info("Could not find a segmentation file.")
                continue

            if candidates[0][0].suffix == '.json':
                logger.info("Reading from newest segmentation file {}...".format( json_file_path ))
                if args.style=='map':
                    Image.fromarray( seg_io.display_polygon_map_from_img_and_json_files( path, json_file_path , color_count=args.colors )).show()
                else:
                    Image.fromarray( seg_io.display_polygon_lines_from_img_and_json_files( path, json_file_path, color_count=args.colors )).show()
             
            elif candidates[0][0].suffix == '.xml':
                logger.info("Reading from newest segmentation file {}...".format( xml_file_path ))
                if args.style=='map':
                    Image.fromarray( seg_io.display_polygon_map_from_img_and_xml_files(path, xml_file_path, color_count=args.colors)).show()
                Image.fromarray( seg_io.display_polygon_lines_from_img_and_xml_files( path, xml_file_path, color_count=args.colors )).show()
            
            elif candidates[0][0].suffix == '.pt':
                logger.info("Reading from newest segmentation map {}...".format( pt_file_path ))
                polygon_map_chw = torch.load( pt_file_path ) 
                logger.debug( f"polygon_map_chw.shape={polygon_map_chw.shape}" )
                Image.fromarray( seg_io.display_polygon_set( img, polygon_map_chw, color_count=args.colors ) ).show()


