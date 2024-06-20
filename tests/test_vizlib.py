
import sys

import pytest
from pathlib import Path
from PIL import Image
import torch
import numpy as np


# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )



import vizlib

@pytest.fixture(scope="module")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


def test_get_n_color_palette_default():
    palette = vizlib.get_n_color_palette( 10 )

    assert palette == [[242, 36, 188],
                       [36, 242, 236],
                       [242, 176, 36],
                       [115, 36, 242],
                       [36, 242, 55],
                       [242, 36, 76],
                       [36, 136, 242],
                       [197, 242, 36],
                       [227, 36, 242],
                       [36, 242, 167]]

def test_get_n_color_palette_saturation_value():

    palette = vizlib.get_n_color_palette( 10, s=.99, v=.99 )

    assert palette == [[252, 2, 186],
                       [2, 252, 245],
                       [252, 172, 2],
                       [99, 2, 252],
                       [2, 252, 26],
                       [252, 2, 51],
                       [2, 124, 252],
                       [197, 252, 2],
                       [234, 2, 252],
                       [2, 252, 161]]


def test_display_polygon_set_default_color_scheme( data_path, ndarrays_regression ):
    input_img_hw = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygons_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))

    actual = vizlib.display_polygon_set( input_img_hw, polygons_chw )

    ndarrays_regression.check( { 'polygon_overlay': actual } )


def test_display_polygon_set_two_color_scheme( data_path, ndarrays_regression ):
    input_img_hw = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygons_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))

    actual = vizlib.display_polygon_set( input_img_hw, polygons_chw, color_count=2 )

    ndarrays_regression.check( { 'polygon_overlay': actual } )


def test_display_polygon_set_three_color_scheme( data_path, ndarrays_regression ):
    input_img_hw = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygons_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))

    actual = vizlib.display_polygon_set( input_img_hw, polygons_chw, color_count=3 )

    ndarrays_regression.check( { 'polygon_overlay': actual } )

def test_display_two_polygon_sets( data_path, ndarrays_regression ):
    input_img_hw = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygons_1_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))
    polygons_2_chw = torch.load(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced_polygon_map.pt'))

    actual = vizlib.display_two_polygon_sets( input_img_hw, polygons_1_chw, polygons_2_chw )

    ndarrays_regression.check( { 'polygon_overlay': actual } )



def Dtest_segmentation_dict_to_polygons_lines():
        """
        To be removed (visualization code)
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_lines( segdict, input_image )
            self.assertTrue( True )

