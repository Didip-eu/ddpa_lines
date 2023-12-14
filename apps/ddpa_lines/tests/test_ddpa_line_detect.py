#!/usr/env python3

import unittest
from torch import Tensor
from pathlib import Path
#from apps.ddpa_lines import seglib
import logging
import pytest
from PIL import Image
from torch import Tensor
import json
import numpy as np
import torch

import sys

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import seglib

class LineDetectTest( unittest.TestCase ):

    @classmethod
    def setUpClass(self):
        self.data_path = Path( __file__ ).parent.joinpath('data')

    # testing local imports
    def test_dummy_1(self):
        self.assertTrue( seglib.dummy() )

    def Dtest_line_segmentation_model_not_found( self ):
        model = Path('nowhere_to_be_found.mlmodel')
        input_image = self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png')
        with pytest.raises( FileNotFoundError ) as e:
            seglib.line_segment(input_image, model)

    def Dtest_line_segmentation_output( self ):
        model = self.data_path.joinpath('kraken_default_blla.mlmodel')
        with Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png')) as input_image:
            lbl, polygons = seglib.line_segment( input_image, model )
            self.assertTrue( type(lbl) is int and typea(polygons) is np.ndarray )

    def test_segmentation_dict_to_polygons_label_count( self ):
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygons( segdict, input_image )
            self.assertTrue( type(lbl) is int and type(polygons) is torch.Tensor )

    def test_segmentation_dict_to_polygons_polygon_img_type( self ):
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygons( segdict, input_image )
            # output should be a 4-channel tensor of 8-bit unsigned integers
            # self.assertTrue( polygons.dtype is np.dtype('uint8') )
            self.assertEqual( polygons.shape, tuple(reversed(input_image.size)) + (4,))

    def test_binary_mask_from_image(self):
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png'))
        self.assertTrue( type(seglib.get_mask( input_img )), torch.Tensor )
        self.assertEqual( seglib.get_mask( input_img ).shape, tuple(reversed(input_img.size)))

    def test_line_segmentation_confusion_matrix( self ):

        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png'))
        print('input_img.size = {} (PIL.Image - WH)'.format( input_img.size ))

        dict_pred = json.load( open(self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r'))
        # (use a MonasteriumTeklia DS class method to generate a json dictionary)
        dict_gt = json.load( open(self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.json'), 'r'))


        # input is a PIL Image, outputs are tensors
        label_count_gt, polygon_gt = seglib.dict_to_polygons( dict_gt, input_img )
        label_count_pred, polygon_pred = seglib.dict_to_polygons( dict_pred, input_img )
        mask = seglib.get_mask( input_img )

        print('mask.shape = {} (np.ndarray - HW)'.format( mask.shape ))

        confusion_matrix = seglib.get_confusion_matrix_from_arrays(polygon_gt, polygon_pred, mask, (label_count_gt, label_count_pred) )

        self.assertTrue( type(confusion_matrix) is torch.Tensor )



if __name__ == "__main__":
    unittest.main()
