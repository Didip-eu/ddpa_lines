#!/usr/bin/env python3

"""
Read cropped writeable areas produced by the 'seals' app and 
segment it into lines (use the Kraken engine, for now) 

Note: f.i.;, the 'seals' apps has been called as follow:

    PYTHONPATH="${HOME}/htr/didipcv/src/:${DIDIP_ROOT}/apps/ddpa_seals" "${DIDIP_ROOT}/apps/ddpa_seals/bin/ddp_seals_detect" -img_paths "${FSDB_ROOT}"/*/*/*/*.img.* -weights ~/tmp/ddp_yolov5.pt -save_crops 1 -preview 0 -crop_classes='["Wr:OldText"]' 
i
Input:
    "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg

Output:
    - PageXML file
    - multiplexed image, with polygons as masks
    
Example call:


    curl -o ~/tmp/blla.mlmodel https://github.com/mittagessen/kraken/blob/main/kraken/blla.mlmodel
    export DIDIP_ROOT=. FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH="${HOME}/htr/didipcv/src/:${DIDIP_ROOT}/apps/ddpa_lines" ${DIDIP_ROOT}/apps/ddpa_lines/bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg


Output formats: The segmentation output is a Python dictionary object, that can be converted to PageXML with Kraken's serializer, or dumpde into a JSON file. Additionally, the associated, local seglib library allows for saving the resulting polygons into a 3D tensor, that can then be used for extracting metrics as well as the line crops (bounding boxes) and their corresponding polygon mask. Note that Kraken's CNN model only predicts the baselines; the polygons are extrapolated from them in a second step.

"""


p = {
        "appname": "lines",
        "model_path": "/home/nicolas/tmp/blla.mlmodel",
        "img_paths": set(["/home/nicolas/tmp/1000_CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.seals.crops/OldText.jpg"]),
        "preview": False,
        "preview_delay": 0,
        "dry_run": False,
        "just_show": False,
        "output_format": [("xml", "json", "pt"), "Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons)"],
}


import sys
import fargv
import kraken
from kraken import blla,serialization
from kraken.lib import vgsl
from PIL import Image, ImageDraw
from pathlib import Path
import torch

sys.path.append( str( Path(__file__).parents[1] ) )


import seglib
import vizlib
import re



if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    for path in list( args.img_paths ):
        print( path )

        stem = Path( path ).stem
        # output folder name stem is the Id part of the input image folder
        new_img_dir_stem = re.sub(r'\..+', '',  Path( path ).parent.name )
        output_dir = Path( path ).parents[1].joinpath( f'{new_img_dir_stem}.{args.appname}.lines' )

        with Image.open( path, 'r' ) as img:

            output_file_path = output_dir.joinpath( f'{stem}.{args.output_format}' )

            if args.just_show:
                # only look for existing tensor map
                map_file_path = output_dir.joinpath(f'{stem}.pt')
                if map_file_path.exists():
                    polygon_map_chw = torch.load( map_file_path ) 
                    Image.fromarray( vizlib.display_polygon_set( img, polygon_map_chw ) ).show()
                continue

            output_dir.mkdir( exist_ok=True )

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Cound not find model file", args.model_path)
            model = vgsl.TorchVGSLModel.load_model( args.model_path )

            segmentation_dict = blla.segment( img, model=model )

            # PageXML output
            if args.output_format == 'xml':
                output_file_path = output_dir.joinpath( f'{stem}.xml' )
                page = serialization.serialize_segmentation(
                    segmentation_dict, image_name=img.filename,
                    image_size=img.size, template='pagexml')

                with open( output_file_path, 'w' ) as fp:
                    fp.write( page )

            # JSON file
            elif args.output_format == 'json':
                output_file_path = output_dir.joinpath( f'{stem}.json' )
                json.dumps( segmentation_dict, output_file_path )

            # store the segmentation into a 3D polygon-map
            elif args.output_format == 'pt':
                output_file_path = output_dir.joinpath( f'{stem}.pt' )
                polygon_map = seglib.polygon_map_from_img_segmentation_dict( img, segmentation_dict )
                torch.save( polygon_map, output_file_path )
                print("Segmentation output saved in {}".format( output_file_path ))


