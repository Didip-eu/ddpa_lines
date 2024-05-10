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
import skimage as ski
from matplotlib import pyplot as plt

import sys

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import seglib


# make import conditional for visualization functions


class LineDetectTest( unittest.TestCase ):

    @classmethod
    def setUpClass(self):
        self.data_path = Path( __file__ ).parent.joinpath('data')

    def test_dummy_1(self):
        """
        A dummy test, as a sanity check for the test framework.
        """
        self.assertTrue( seglib.dummy() )

    def test_line_segmentation_model_not_found( self ):
        """
        An exception should be raised when no segmentation model found.
        """
        model = Path('nowhere_to_be_found.mlmodel')
        input_image = self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png')
        with pytest.raises( FileNotFoundError ) as e:
            seglib.line_segment(input_image, model)


    def Dtest_flat_to_cube( self ):
        """
        To be removed (utility function).
        """
        t = torch.randint(120398, (7,5), dtype=torch.int32 )
        self.assertTrue( torch.all( t.eq( seglib.rgba_uint8_to_int32( seglib.array_to_rgba_uint8(t) ))))

    def test_binary_mask_from_image_real_img(self):
        """
        Binarization function should return a Boolean tensor with shape( h,w )
        """
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
        mask = seglib.get_mask( input_img )
        self.assertEqual( type(mask), torch.Tensor ) and self.assertEqual( mask.dtype, torch.int32 )
        self.assertEqual( mask.shape, tuple(reversed(input_img.size)))

    def test_binary_mask_from_image_fg_bg(self):
        """
        Binary map should be 1 for FG pixels, 0 otherwise.
        """
        img_arr = np.full((15,15,3), 200, dtype=np.uint8 ) # background
        img_arr[8,8]=5 # foreground = 1 pixel
        binary_map = seglib.get_mask( Image.fromarray( img_arr) )
        self.assertEqual( binary_map[8,8], 1) # single FG pixel = T
        self.assertEqual( torch.sum(binary_map).item(), 1 ) # remaining pixels = F

    def Dtest_line_segmentation_output( self ):
        """
        seglib.line_segment( image, model ) --a thin wrapper around Kraken's blla.segment-- should return a tuple (labels, polygons).
        """
        model = self.data_path.joinpath('kraken_default_blla.mlmodel')
        with Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1.png')) as input_image:
            lbl, polygons = seglib.line_segment( input_image, model )
            self.assertTrue( type(lbl) is int and type(polygons) is torch.Tensor )

    def Dtest_segmentation_dict_to_polygon_map_label_count( self ):
        """
        seglib.dict_to_polygon_map(dict, image) should return a tuple (labels, polygons) 
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_map( segdict, input_image )
            self.assertTrue( type(lbl) is int and type(polygons) is torch.Tensor )


    def Dtest_segmentation_dict_to_polygon_map_polygon_img_type( self ):
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_map( segdict, input_image )
            # output should be a 4-channel tensor of 8-bit unsigned integers
            # self.assertTrue( polygons.dtype is np.dtype('uint8') )
            self.assertEqual( polygons.shape, (4,) + tuple(reversed(input_image.size)) )

    def Dtest_segmentation_dict_to_polygons_lines( self ):
        """ 
        To be removed (visualization code)
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_lines( segdict, input_image )
            self.assertTrue( True )


    def test_line_segmentation_confusion_matrix( self ):
        """
        This test takes way too long to complete and should ultimately be removed.
        """

        print('Dtest_line_segmentation_confusion_matrix()')
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
        #print('input_img.size = {} (PIL.Image - WH)'.format( input_img.size ))

        dict_pred = json.load( open(self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))
        # (note: used a MonasteriumTeklia DS class method to generate a json dictionary)
        dict_gt = json.load( open(self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))


        # input is a PIL Image, outputs are tensors
        label_count_gt, polygon_gt = seglib.dict_to_polygon_map( dict_gt, input_img )
        label_count_pred, polygon_pred = seglib.dict_to_polygon_map( dict_pred, input_img )
        mask = seglib.get_mask( input_img )

        print('mask.shape = {} (np.ndarray - HW)'.format( mask.shape ), "label_max=", max(label_count_gt, label_count_pred))

        confusion_matrix = seglib.get_confusion_matrix_from_arrays(polygon_gt, polygon_pred, mask, (label_count_gt, label_count_pred) )
        torch.save( confusion_matrix, self.data_path.joinpath('confusion_matrix.pt') )

        print(confusion_matrix)

        self.assertTrue( type(confusion_matrix) is torch.Tensor )


    def Dtest_evaluate( self ):
        pass




if __name__ == "__main__":
    unittest.main()
