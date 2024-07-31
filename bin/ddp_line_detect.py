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

from pathlib import Path

p = {
        "appname": "lines",
        "model_path": Path.home().joinpath("tmp/models/segmentation/blla.mlmodel"),
        "img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "preview": False,
        "preview_delay": 0,
        "dry_run": False,
        "just_show": False,
        "mapify": False,
        "line_type": [("polygon","bbox","legacy_bbox"), "Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; bbox = bounding boxes, derived from the former; legacy_bbox: legacy Kraken segmentation)"],
        "output_format": [("xml", "json", "pt"), "Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons)"],
}


import sys
import fargv
import kraken
from kraken import blla, pageseg, serialization
from kraken.containers import Segmentation
from kraken.lib import vgsl
from PIL import Image, ImageDraw
from pathlib import Path
import torch
import dataclasses
import json

sys.path.append( str( Path(__file__).parents[1] ) )


import seglib
import seg_io
import re



#def segmentation_record_to_line_dict( sr: Segmentation) -> dict:
#    """
#    Transforms a Kraken custom Segmentation record into a plain dictionary.
#    TEMPORARY: should be method in kraken/containers.py
#
#    Args:
#        segmentation_record (``Segmentation``): a structure as below::
#
#            Segmentation(type='baselines', imagename='/home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/147c32f12ef7b285bd19e44ab47e253a.img.jpg', text_direction='horizontal-lr', script_detection=False, lines=[BaselineLine(id='b219e3c1-019e-45a6-b3fa-108baeb37ae9', baseline=[[384, 748], [2172, 700]], boundary=[[2167, 652], [1879, 668], ..., [2167, 652]], text=None, base_dir=None, type='baselines', imagename=None, tags={'type': 'default'}, split=None, regions=['4750600c-e32e-4835-a9f4-00a05f9e1c92']), BaselineLine(id='ab2a0883-75a8-43e4-ad59-119ea1c75449', ... )])
#
#    Output:
#        dict: a dictionary of regions (optional) and lines::
#
#             {type="baselines", imagename="...", ..., "lines"=[{id="...", baseline="", boundary=""], ...]}
#    """
#
#    bbox_to_path = lambda bbox: [ [bbox[0],bbox[1]], [bbox[2],bbox[1]], [bbox[2],bbox[3]], [bbox[0],bbox[3]] ]
#
#    segmentation_dict = {
#            'type': sr.type,
#            'imagename': sr.imagename,
#            'text_direction': sr.text_direction,
#            'lines': []
#            }
#    for line in segmentation_record.lines:
#        line_dict = {
#                'id': line.id,
#                'baseline': line.baseline if sr.type=='baselines' else None,
#                'boundary': line.boundary if sr.type=='baselines' else bbox_to_path( line.bbox )
#                } 
#        segmentation_dict['lines'].append( line_dict )
#
#    return segmentation_dict



if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    for path in list( args.img_paths ):
        print( path )

        #stem = Path( path ).stem
        stem = re.sub(r'\..+', '',  Path( path ).name )

        # an extra output subfolder is only useful for storing file that may derive from the segmentation (s.a. crops)
        # + location: under the chart's folder, at same level of the chart images
        # + name: stem is the Id part of the input image
        # extra_output_dir = Path( path ).parent.joinpath( f'{stem}.{args.appname}.lines' )
        # extra_output_dir.mkdir( exist_ok=True )

        with Image.open( path, 'r' ) as img:

            # segmentation metadata file
            output_file_path_wo_suffix = Path(path).parent.joinpath( f'{stem}.{args.appname}.pred' )


            xml_file_path = Path(f'{output_file_path_wo_suffix}.xml')
            pt_file_path = Path(f'{output_file_path_wo_suffix}.pt')

            if args.just_show:
                # only look for existing tensor map
                if pt_file_path.exists():
                    polygon_map_chw = torch.load( pt_file_path ) 
                    Image.fromarray( seg_io.display_polygon_set( img, polygon_map_chw ) ).show()
                else:
                    print(f"No existing segmentation map for image {repr(path)}.")
            elif args.mapify:
                if xml_file_path.exists():
                    segmap = seglib.polygon_map_from_img_xml_files(path, xml_file_path )
                    print(xml_file_path,'→', pt_file_path )
                    torch.save( segmap, pt_file_path )
                continue

            #extra_output_dir.mkdir( exist_ok=True )

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = vgsl.TorchVGSLModel.load_model( args.model_path )

            segmentation_record = None

            # Legacy segemntation
            if (args.line_type=='legacy_bbox'):
                # Image needs to be binarized first
                from kraken import binarization
                img_bw = binarization.nlbin( img )
                segmentation_record = pageseg.segment( img_bw )
            else:
                # CNN-based segmentation
                segmentation_record = blla.segment( img, model=model )

                # BBox conversion (use a custom method on Kraken_didip
                if args.line_type == 'bbox':
                    segmentation_record = segmentation_record.to_bbox_segmentation()

            output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
            
            # PageXML output
            if args.output_format == 'xml':
                page = serialization.serialize(
                    segmentation_record, #, image_name=img.filename,
                    image_size=img.size, template='pagexml')

                with open( output_file_path, 'w' ) as fp:
                    fp.write( page )

            # JSON file (work from dict)
            elif args.output_format == 'json':
                json.dump( dataclasses.asdict( segmentation_record ), output_file_path )

            # store the segmentation into a 3D polygon-map
            elif args.output_format == 'pt':
                # create a segmentation dictionary from Segmentation
                polygon_map = seglib.polygon_map_from_img_segmentation_dict( 
                                    img, 
                                    dataclasses.asdict( segmentation_record ))
                torch.save( polygon_map, output_file_path )
                print("Segmentation output saved in {}".format( output_file_path ))





