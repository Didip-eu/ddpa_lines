#!/usr/bin/env python3

"""
Read cropped writeable areas produced by the 'seals' app and 
segment it into lines (use the Kraken engine, for now) 

Note: f.i.;, the 'seals' apps has been called as follow::

    PYTHONPATH="${HOME}/htr/didipcv/src/:${DIDIP_ROOT}/apps/ddpa_seals" "${DIDIP_ROOT}/apps/ddpa_seals/bin/ddp_seals_detect" -img_paths "${FSDB_ROOT}"/*/*/*/*.img.* -weights ~/tmp/ddp_yolov5.pt -save_crops 1 -preview 0 -crop_classes='["Wr:OldText"]' 
i
Input:
    "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg

Output:
    - PageXML file
    - multiplexed image, with polygons as masks
    
Example call::

    curl -o ~/tmp/blla.mlmodel https://github.com/mittagessen/kraken/blob/main/kraken/blla.mlmodel
    export DIDIP_ROOT=. FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH="${HOME}/htr/didipcv/src/:${DIDIP_ROOT}/apps/ddpa_lines" ${DIDIP_ROOT}/apps/ddpa_lines/bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg


Output formats: 
    Kraken's segmentation output is a custom Segmentation object (kraken/containers.py) with line records that are either 

    + bounding boxes (BBoxLine) if using the legacy segmentation functions (kraken/pageseg.py)
    + polygons (BaselineLine) if using the CNN-based model (kraken/blla.py)

    This Segmentation object that then can be converted into different formats:

    + PageXML -- with Kraken's own serializer
    + JSON
    + 3D tensor, through the associated, local seglib library; it can then be used for extracting metrics as well as the line crops (bounding boxes) and their corresponding polygon mask. Note that Kraken's CNN model only predicts the baselines; the polygons are extrapolated from them in a second step.

    Note that the last two formats require first converting the Segmentation object into a plain, old Python dictionary (function segmentation_record_to_line_dict())

"""
# stdlib
import sys
from pathlib import Path
import dataclasses
import json
import re
import sys
import logging

# 3rd party
import torch
from PIL import Image, ImageDraw

# Didip
import fargv
from PIL import Image, ImageDraw
import json
import numpy as np


root = Path(__file__).parents[1]
sys.path.append( str( root ))

# local
from kraken import blla
from kraken import pageseg, serialization
from kraken.lib import vgsl, layers
from kraken.containers import Segmentation
from seglib import seglib
from seglib import seg_io

logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
        "appname": "lines",
        "model_path": str(root.joinpath("models/blla.mlmodel")),
        "img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "mask_classes": [set(['Wr:OldText']), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>."],
        "preview": False,
        "preview_delay": 0,
        "dry_run": False,
        "just_show": False,
        "line_type": [("polygon","legacy_bbox"), "Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; legacy_bbox: legacy Kraken segmentation)"],
        "output_format": [("xml", "json", "pt"), "Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons)"],
}


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    logger.debug( args )

    for path in list( args.img_paths ):
        logger.debug( path )
        path = Path(path)

        #stem = Path( path ).stem
        stem = re.sub(r'\..+', '', path.name )

        # only for segmentation on Seals-detected regions
        region_segfile = re.sub(r'.img.jpg', args.region_segmentation_suffix, str(path) )

        # an extra output subfolder is only useful for storing file that may derive from the segmentation (s.a. crops)
        # + location: under the chart's folder, at same level of the chart images
        # + name: stem is the Id part of the input image
        # extra_output_dir = Path( path ).parent.joinpath( f'{stem}.{args.appname}.lines' )
        # extra_output_dir.mkdir( exist_ok=True )
        
        with Image.open( path, 'r' ) as img:

            ############ 1. Look first for existing segmentation data ##########

            # segmentation metadata file
            output_file_path_wo_suffix = path.parent.joinpath( f'{stem}.{args.appname}.pred' )


            json_file_path = Path(f'{output_file_path_wo_suffix}.json')
            xml_file_path = Path(f'{output_file_path_wo_suffix}.xml')
            pt_file_path = Path(f'{output_file_path_wo_suffix}.pt')


            if args.just_show:

            # look at existing files, choose the most recent one
                candidates = sorted([ (f, f.stat().st_mtime) for f in (json_file_path, xml_file_path, pt_file_path ) if f.exists() ], key=lambda x: x[1], reverse=True)

                if candidates == []:
                    logger.info("Could not find a segmentation file.")
                    continue

                if candidates[0][0].suffix == '.json':
                    logger.info("Reading from newest segmentation file {}...".format( json_file_path ))
                    Image.fromarray( seg_io.display_polygon_lines_from_img_and_json_files( path, json_file_path )).show()
                 
                elif candidates[0][0].suffix == '.xml':
                    logger.info("Reading from newest segmentation file {}...".format( xml_file_path ))
                    Image.fromarray( seg_io.display_polygon_lines_from_img_and_xml_files( path, xml_file_path )).show()
                
                elif candidates[0][0].suffix == '.pt':
                    logger.info("Reading from newest segmentation map {}...".format( pt_file_path ))
                    polygon_map_chw = torch.load( pt_file_path ) 
                    logger.debug( f"polygon_map_chw.shape={polygon_map_chw.shape}" )
                    Image.fromarray( seg_io.display_polygon_set( img, polygon_map_chw ) ).show()
                continue

            ############## 2. Segment the file ################

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = vgsl.TorchVGSLModel.load_model( args.model_path )


            # Option 1: go JSON all the way and use this format to segment the region crops (from seals),
            # before merging them into a single, page-wide file

            if args.mask_classes != []:
                logger.debug(f"Run segmentation on masked regions '{args.mask_classes}', instead of whole page.")
                # parse segmentation file, and extract and concatenate the WritableArea crops
                with open(region_segfile) as regseg_if:
                    regseg = json.load( regseg_if )
                   
                    # iterate over seals crops and segment
                    crops_hw, classes = seglib.seals_regseg_to_crops( img, regseg, args.mask_classes )
                    line_seg_dicts = [ dataclasses.asdict( blla.segment( crop, model=model )) for crop in crops_hw ]
                    for (i, lsd, clsname) in zip(range(len(classes)), line_seg_dicts, classes):
                        lsd['imagename']=str(Path(img.filename).with_suffix('.seals.crops').joinpath('{}.{}.jpg'.format(i, clsname.replace(':','_'))))
                    # combine
                    seg_dict = seglib.merge_seals_regseg_lineseg( regseg, args.mask_classes, *line_seg_dicts )

                    output_file_path = Path(f'{output_file_path_wo_suffix}.json')

                    with open(output_file_path, 'w') as of:
                        json.dump( seg_dict, of )
                        logger.info("Segmentation output saved in {}".format( output_file_path ))


            #" Option 2: single-file segmentation, with a choice of output formats.
            else:
                
                segmentation_record = None

                # Legacy segmentation
                if (args.line_type=='legacy_bbox'):
                    # Image needs to be binarized first
                    from kraken import binarization
                    img_bw = binarization.nlbin( img )
                    segmentation_record = pageseg.segment( img_bw )
                else:
                    # CNN-based segmentation
                    logger.info("Starting segmentation")
                    segmentation_record = blla.segment( img, model=model )
                    logger.info("Successful segmentation.")

                
                ############ 3. Handing the output #################
                output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
                
                logger.debug(f"Serializing segmentation for img.shape={img.size}")
                # PageXML output
                if args.output_format == 'xml':
                    page = serialization.serialize(
                        segmentation_record, #, image_name=img.filename,
                        image_size=img.size,
                        template=str(root.joinpath('kraken', 'templates', 'pagexml')), template_source='custom')

                    logger.debug(f"Serializing XML with shape={img.size}")
                    with open( output_file_path, 'w' ) as fp:
                        fp.write( page )

                # JSON file (work from dict)
                elif args.output_format == 'json':
                    with open(output_file_path, 'w') as of:
                        json.dump( dataclasses.asdict( segmentation_record ), of )
                        logger.info("Segmentation output saved in {}".format( output_file_path ))

                # store the segmentation into a 3D polygon-map
                elif args.output_format == 'pt':
                    # create a segmentation dictionary from Segmentation
                    polygon_map = seglib.polygon_map_from_img_segmentation_dict( 
                                        img, 
                                        dataclasses.asdict( segmentation_record ))
                    
                    logger.debug(f"Serializing map with shape={polygon_map.size}")
                    torch.save( polygon_map, output_file_path )
                    logger.info("Segmentation output saved in {}".format( output_file_path ))





