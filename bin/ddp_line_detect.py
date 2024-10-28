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
import seglib
import seg_io

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(funcName)s: %(message)s", force=True )
logger = logging.getLogger(__name__)


p = {
        "appname": "lines",
        "model_path": str(root.joinpath("models/blla.mlmodel")),
        "img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "seal_segmentation_class": ['Img:WritableArea', "Name of the writable area class in the Seals app."],
        "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>; if empty, entire img file is passed to the line segmenter."],
        "preview": False,
        "preview_delay": 0,
        "dry_run": False,
        "just_show": False,
        "mapify": False,
        "line_type": [("polygon","bbox","legacy_bbox"), "Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; bbox = bounding boxes, derived from the former; legacy_bbox: legacy Kraken segmentation)"],
        "output_format": [("xml", "json", "pt"), "Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons)"],
}


def json_segmentation_to_writeable_area( img: Image, json_segfile: Path ) -> Image:
    """
    Concatenate several writable regions into one image. 
    """
    with open( json_segfile, 'r') as infile:
        seg_dict = json.load( infile )
        clsid_2_clsname = { i:n for (i,n) in enumerate( seg_dict['class_names'] )}
        to_keep = [ i for (i,v) in enumerate( seg_dict['rect_classes'] ) if clsid_2_clsname[v]=='Img:WritableArea' ]
        rectangles = []
        for coords in [ c for (index, c) in enumerate( seg_dict['rect_LTRB'] ) if index in to_keep ]:
            rectangles.append( np.asarray( img.crop( coords )))
        channels, dtype = rectangles[-1].shape[-1], rectangles[-1].dtype
        w_concat = max( r.shape[1] for r in rectangles ) 
        h_concat = sum( r.shape[0] for r in rectangles )
        
        concatenation = np.zeros( (h_concat, w_concat, channels), dtype=dtype )
        offset=0
        for r in rectangles:
            concatenation[offset:offset+r.shape[0], :r.shape[1]] = r
            offset += r.shape[0]

        return Image.fromarray( concatenation )



if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    for path in list( args.img_paths ):
        logger.debug( path )

        #stem = Path( path ).stem
        stem = re.sub(r'\..+', '',  Path( path ).name )

        # an extra output subfolder is only useful for storing file that may derive from the segmentation (s.a. crops)
        # + location: under the chart's folder, at same level of the chart images
        # + name: stem is the Id part of the input image
        # extra_output_dir = Path( path ).parent.joinpath( f'{stem}.{args.appname}.lines' )
        # extra_output_dir.mkdir( exist_ok=True )
        
        with Image.open( path, 'r' ) as img:

            ############ 1. Look first for existing segmentation data ##########

            # segmentation metadata file
            output_file_path_wo_suffix = Path(path).parent.joinpath( f'{stem}.{args.appname}.pred' )


            xml_file_path = Path(f'{output_file_path_wo_suffix}.xml')
            pt_file_path = Path(f'{output_file_path_wo_suffix}.pt')

            def mapify_xml():
                if not xml_file_path.exists():
                    raise FileNotFoundError(f"No existing Page XML file {xml_file_path} for image {repr(path)}."
                                             "Check that segmentation was run on this file.")
                smp = seglib.polygon_map_from_img_xml_files(path, xml_file_path )
                logger.info(xml_file_path, '→', pt_file_path )
                torch.save( smp, pt_file_path )

            if args.just_show:
                # only look for existing tensor map
                if not pt_file_path.exists():
                    logger.info(f"No existing segmentation map {pt_file_path} for image {repr(path)}."
                           "Looking for XML page file instead.")
                    mapify_xml()
                    
                polygon_map_chw = torch.load( pt_file_path ) 
                Image.fromarray( seg_io.display_polygon_set( img, polygon_map_chw ) ).show()
                continue

            elif args.mapify:
                mapify_xml()
                continue

            ############## 2. Segment the file ################

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = vgsl.TorchVGSLModel.load_model( args.model_path )

            if args.region_segmentation_suffix != '':
                region_segfile = path.with_suffix( args.region_segmentation_suffix )
                # 1. Parse segmentation file, and extract and concatenate the WritableArea crops
                # 2. Pass this image to the segmetner

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

                # BBox conversion (use a custom method on Kraken_didip
                if args.line_type == 'bbox':
                    segmentation_record = segmentation_record.to_bbox_segmentation()

            
            ############ 3. Handing the output #################
            output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
            
            # PageXML output
            if args.output_format == 'xml':
                page = serialization.serialize(
                    segmentation_record, #, image_name=img.filename,
                    image_size=img.size,
                    template=str(root.joinpath('kraken', 'templates', 'pagexml')), template_source='custom')

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
                torch.save( polygon_map, output_file_path )
                logger.info("Segmentation output saved in {}".format( output_file_path ))





