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

    def test_array_to_rgba_uint8( self ):
        """
        Conversion of a 1-channel 32-bit unsigned int array yields a 4-channel tensor.
        """
        arr = np.array( [[2,2,2,0,0,0],
                         [2,2,2,0,0,0],
                         [2,2,0x203,3,0,0],
                         [0,0,3,3,3,0],
                         [0,0,0,0,0,0],
                         [0,0x40102,0,0,0,0]], dtype='int32')
        tensor = seglib.array_to_rgba_uint8( arr )
        self.assertTrue( torch.equal( tensor,
            torch.tensor([[[2, 2, 2, 0, 0, 0],
                           [2, 2, 2, 0, 0, 0],
                           [2, 2, 3, 3, 0, 0],
                           [0, 0, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0, 0]],

                          [[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]],

                          [[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 4, 0, 0, 0, 0]],

                          [[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)))

    def test_rgba_uint8_to_hw_tensor( self ):
        """
        Conversion of a 1-channel 32-bit unsigned int array yields a 4-channel tensor.
        """
        tensor = torch.tensor([[[2, 2, 2, 0, 0, 0],
                                [2, 2, 2, 0, 0, 0],
                                [2, 2, 3, 3, 0, 0],
                                [0, 0, 3, 3, 3, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 2, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 4, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        self.assertTrue( np.array_equal( seglib.rgba_uint8_to_hw_tensor( tensor ),
                        torch.tensor( [[2,2,2,0,0,0],
                                       [2,2,2,0,0,0],
                                       [2,2,0x203,3,0,0],
                                       [0,0,3,3,3,0],
                                       [0,0,0,0,0,0],
                                       [0,0x40102,0,0,0,0]], dtype=torch.int32)))

    def test_flat_to_cube_and_other_way_around( self ):
        """
        A flat map that is stored into a cube and then retrieved back as a map should contain
        the same values as the original.
        """
        a = np.random.randint(120398, size=(7,5), dtype=np.int32 )
        self.assertTrue( torch.equal( torch.from_numpy( a ), seglib.rgba_uint8_to_hw_tensor( seglib.array_to_rgba_uint8(a) )))

    def test_polygon_mask_to_polygon_map_32b_store_single_polygon(self):
        """
        Storing polygon (as binary mask + label) on empty tensor yields correct map
        """
        label_map = np.zeros((6,6), dtype='int32')
        polygon_mask = np.array( [[1,1,1,0,0,0],
                                  [1,1,1,0,0,0],
                                  [1,1,1,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 2)
        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')))

    def test_polygon_mask_to_polygon_map_32b_store_two_intersecting_polygons(self):
        """
        Storing extra polygon (as binary mask + labels) on labeled tensor yields correct map
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='int32')

        polygon_mask = np.array( [[0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,1,1,0,0],
                                  [0,0,1,1,1,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')

        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)

        # intersecting pixel has value (l1<<8) + l2
        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,0x203,3,0,0],
                                  [0,0,3,3,3,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')))


    def test_polygon_mask_to_polygon_map_32b_store_two_polygons_no_intersection(self):
        """
        Storing extra polygon (as binary mask + labels) on labeled tensor yields
        2-label map.
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='int32')

        polygon_mask = np.array( [[0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,1,1,0],
                                  [0,0,1,1,1,1],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')

        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,2,3,3,0],
                                  [0,0,3,3,3,3],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='int32')))

    def test_polygon_mask_to_polygon_map_32b_store_three_intersecting_polygons(self):
        """
        Storing two extra polygons (as binary mask + labels) on labeled tensor with overlap yields
        map with intersection labels l = l1, l' = (l1<<8) + l2, l''=(l1<<16)+(l2<<8)+l3, ...
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='int32')

        polygon_mask_1 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]], dtype='int32')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_1, 3)

        polygon_mask_2 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,1,1],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,0,0]], dtype='int32')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_2, 4)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,0x20304,0x304,4,0],
                                  [0,0,0x304,0x304,0x304,4],
                                  [0,0,4,4,4,0],
                                  [0,0,4,4,0,0]], dtype='int32')))


    def test_polygon_mask_to_polygon_map_32b_store_large_values(self):
        """
        Storing 3 polygons with large labels (ex. 255) yields correct map (no overflow)
        """
        label_map = np.array( [[255,255,255,0,0,0],
                               [255,255,255,0,0,0],
                               [255,255,255,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='int32')

        polygon_mask_1 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]], dtype='int32')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_1, 255)

        polygon_mask_2 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,1,1],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,0,0]], dtype='int32')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_2, 255)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[0xff,0xff,0xff,0,0,0],
                                  [0xff,0xff,0xff,0,0,0],
                                  [0xff,0xff,0xffffff,0xffff,0xff,0],
                                  [0,0,0xffff,0xffff,0xffff,0xff],
                                  [0,0,0xff,0xff,0xff,0],
                                  [0,0,0xff,0xff,0,0]], dtype='int32')))


    def test_retrieve_polygon_mask_from_map_no_binary_mask_1( self ):
        label_map = torch.tensor( [[2,2,2,0,0,0],
                                   [2,2,2,0,0,0],
                                   [2,2,0x20304,0x304,4,0],
                                   [0,0,0x304,0x304,0x304,4],
                                   [0,0,4,4,4,0],
                                   [0,0,4,4,0,0]], dtype=torch.int)
        expected = torch.tensor( [[False, False, False, False, False, False],
                                   [False, False, False, False, False, False],
                                   [False, False,  True,  True, False, False],
                                   [False, False,  True,  True,  True, False],
                                   [False, False, False, False, False, False],
                                   [False, False, False, False, False, False]])

        self.assertTrue( torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3), expected))
        # second call ensures that the map is not modified by the retrieval operation
        self.assertTrue( torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3), expected))

    def test_retrieve_polygon_mask_from_map_no_binary_mask_2( self ):
        label_map = torch.tensor( [[2,2,2,0,0,0],
                                   [2,2,2,0,0,0],
                                   [2,2,0x20304,0x304,4,0],
                                   [0,3,0x304,0x304,0x304,4],
                                   [0,0,3,4,4,0],
                                   [0,0,4,4,0,0]], dtype=torch.int)

        self.assertTrue( torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3),
                    torch.tensor( [[False, False, False, False, False, False],
                                   [False, False, False, False, False, False],
                                   [False, False,  True,  True, False, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False, False,  True, False, False, False],
                                   [False, False, False, False, False, False]])))


    def test_segmentation_dict_to_polygon_map_label_count( self ):
        """
        seglib.dict_to_polygon_map(dict, image) should return a tuple (labels, polygons)
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            polygons = seglib.dict_to_polygon_map( segdict, input_image )
            self.assertTrue( type(polygons) is torch.Tensor and torch.max(polygons)==4)


    def test_segmentation_dict_to_polygon_map_polygon_img_type( self ):
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            polygons = seglib.dict_to_polygon_map( segdict, input_image )
            self.assertEqual( polygons.shape, (4,)+input_image.size[::-1])

    def test_union_intersection_count_two_maps( self ):
        """
        Provided two label maps that each encode (potentially overlapping) polygons, yield 
        intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
        and j ∈ map2.
        Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
        """
        map1 = torch.tensor([[2,2,2,0,0,0],
                          [2,2,2,0,0,0],
                          [2,2,0x20304,0x304,4,0],
                          [0,0,0x304,0x304,0x304,4],
                          [0,0,4,4,4,0],
                          [0,0,4,0x402,0,0]], dtype=torch.int)

        map2 = torch.tensor( [[0,2,2,0,0,0],
                          [2,2,4,2,2,0],
                          [2,2,0x20304,0x304,4,0],
                          [0,3,0x304,0x304,0x304,4],
                          [0,0,3,4,4,0],
                          [0,0,0x204,4,0,0]], dtype=torch.int)

        pixel_count = seglib.union_intersection_count_two_maps( map1, map2 )

        c1l,c1r = 0,0
        c2l, c2r = 8+1/3+.5, 8+1/3+.5
        c3l, c3r = 1/3+.5*4, 1/3+2+.5*4
        c4l, c4r = 1/3+5*.5+6, 1/3+5*.5+6
        i11, i12, i13, i14 = 0, 0, 0, 0
        i21, i22, i23, i24 = 0, 6+1/3, 1/3, 1+1/3+.5
        i31, i32, i33, i34 = 0, 1/3, 1/3+.5*4, 1/3+.5*4
        i41, i42, i43, i44 = 0, 1/3+.5, 1/3+.5*4+1, 1/3+.5*6+4
        u11, u12, u13, u14 = c1l+c1r-i11, c1l+c2r-i12, c1l+c3r-i13, c1l+c4r-i14
        u21, u22, u23, u24 = c2l+c1r-i21, c2l+c2r-i22, c2l+c3r-i23, c2l+c4r-i24
        u31, u32, u33, u34 = c3l+c1r-i31, c3l+c2r-i32, c3l+c3r-i33, c3l+c4r-i34
        u41, u42, u43, u44 = c4l+c1r-i41, c4l+c2r-i42, c4l+c3r-i43, c4l+c4r-i44

        expected =    torch.tensor([[[ i11, u11],  # 1,1
                                     [ i12, u12],  # 1,2
                                     [ i13, u13],  # 1,3
                                     [ i14, u14]], # 1,4
                                    [[ i21, u21],  # 2,1
                                     [ i22, u22],  # 2,2
                                     [ i23, u23],  # 2,3
                                     [ i24, u24]], # 2,4
                                    [[ i31, u31],  # 3,1
                                     [ i32, u32],  # ...
                                     [ i33, u33],
                                     [ i34, u34]], # 3,4
                                    [[ i41, u41],  # 4,1
                                     [ i42, u42],  # ...
                                     [ i43, u43],
                                     [ i44, u44]]]) # 4,4

        # Note: we're comparing float value here
        self.assertTrue( torch.all(torch.isclose( pixel_count, expected )))


    def test_line_segmentation_confusion_matrix( self ):
        """
        On an actual, only a few sanity checks for testing
        """
        map1 = torch.tensor([[[2, 2, 2, 0, 0, 0],
                              [2, 2, 2, 0, 0, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 0, 4, 4, 4, 4],
                              [0, 0, 4, 4, 4, 0],
                              [0, 0, 4, 2, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        map2 = torch.tensor([[[0, 2, 2, 0, 0, 0],
                              [2, 2, 4, 2, 2, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 3, 4, 4, 4, 4],
                              [0, 0, 3, 4, 4, 0],
                              [0, 0, 4, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        confusion_matrix = seglib.get_confusion_matrix_from_polygon_maps(map1,map2)

        self.assertEqual( confusion_matrix.dtype, torch.float32 )
        self.assertFalse( torch.all( confusion_matrix == 0 ))

        self.assertTrue( torch.all( confusion_matrix.isclose(
            torch.tensor(
                 [[0., 0., 0., 0. ],
                  [0., 0.5588222318300937, 0.025971496029859816, 0.1157876121844467],
                  [0., 0.03076624851153388, 0.538457988138370, 0.2641481665968551],
                  [0., 0.049503068322907566, 0.33898081010444103, 0.7096764828273641]], dtype=torch.float32), 1e-4)))

    def test_line_segmentation_confusion_matrix_wrong_gt_type( self ):
        """
        On an actual, only a few sanity checks for testing
        """
        map1 = torch.tensor([[2,2,2,0,0,0],
                             [2,2,2,0,0,0],
                             [2,2,0x20304,0x304,4,0],
                             [0,0,0x304,0x304,0x304,4],
                             [0,0,4,4,4,0],
                             [0,0,4,0x402,0,0]], dtype=torch.int)

        map2 = torch.tensor([[[0, 2, 2, 0, 0, 0],
                              [2, 2, 4, 2, 2, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 3, 4, 4, 4, 4],
                              [0, 0, 3, 4, 4, 0],
                              [0, 0, 4, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        with self.assertRaises( TypeError ):
            seglib.get_confusion_matrix_from_polygon_maps(map1,map2)

    def test_line_segmentation_confusion_matrix_wrong_pred_type( self ):
        """
        On an actual, only a few sanity checks for testing
        """

        map1 = torch.tensor([[[0, 2, 2, 0, 0, 0],
                              [2, 2, 4, 2, 2, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 3, 4, 4, 4, 4],
                              [0, 0, 3, 4, 4, 0],
                              [0, 0, 4, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        map2 = torch.tensor([[2,2,2,0,0,0],
                             [2,2,2,0,0,0],
                             [2,2,0x20304,0x304,4,0],
                             [0,0,0x304,0x304,0x304,4],
                             [0,0,4,4,4,0],
                             [0,0,4,0x402,0,0]], dtype=torch.int)

        with self.assertRaises( TypeError ):
            seglib.get_confusion_matrix_from_polygon_maps(map1,map2)


    def test_line_segmentation_confusion_matrix_different_map_shapes( self ):
        """
        On an actual, only a few sanity checks for testing
        """
        map1 = torch.tensor([[[2, 2, 2, 0, 0, 0],
                              [2, 2, 2, 0, 0, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 0, 4, 4, 4, 4],
                              [0, 0, 4, 2, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        map2 = torch.tensor([[[0, 2, 2, 0, 0, 0],
                              [2, 2, 4, 2, 2, 0],
                              [2, 2, 4, 4, 4, 0],
                              [0, 3, 4, 4, 4, 4],
                              [0, 0, 3, 4, 4, 0],
                              [0, 0, 4, 4, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 3, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        with self.assertRaises( TypeError ):
            seglib.get_confusion_matrix_from_polygon_maps(map1,map2)

    def test_line_segmentation_confusion_matrix_realistic( self ):
        """
        On an actual image, only a few sanity checks for testing
        """
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
        dict_pred = json.load( open(self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))
        dict_gt = json.load( open(self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))

        polygon_gt = seglib.dict_to_polygon_map( dict_gt, input_img )
        polygon_pred = seglib.dict_to_polygon_map( dict_pred, input_img )
        binary_mask = seglib.get_mask( input_img )

        confusion_matrix = seglib.get_confusion_matrix_from_polygon_maps(polygon_gt, polygon_pred, binary_mask)
        #torch.save( confusion_matrix, self.data_path.joinpath('confusion_matrix.pt') )

        self.assertEqual( confusion_matrix.dtype, torch.float32 )
        self.assertFalse( torch.all( confusion_matrix == 0 ))
        self.assertTrue(
        confusion_matrix[0,0]>confusion_matrix[0,1] and
        confusion_matrix[1,1]>confusion_matrix[1,0] and confusion_matrix[1,2] and
        confusion_matrix[2,2]>confusion_matrix[2,1] and confusion_matrix[2,3] and
        confusion_matrix[3,3]>confusion_matrix[3,2] )

    def test_line_segmentation_confusion_matrix_from_img_json( self ):
        """
        On an actual image, only a few sanity checks for testing
        """
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
        dict_pred = json.load( open(self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))
        dict_gt = json.load( open(self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))

        confusion_matrix = seglib.get_confusion_matrix_from_img_json(input_img, dict_gt, dict_pred)

        self.assertEqual( confusion_matrix.dtype, torch.float32 )
        self.assertFalse( torch.all( confusion_matrix == 0 ))
        self.assertTrue(
        confusion_matrix[0,0]>confusion_matrix[0,1] and
        confusion_matrix[1,1]>confusion_matrix[1,0] and confusion_matrix[1,2] and
        confusion_matrix[2,2]>confusion_matrix[2,1] and confusion_matrix[2,3] and
        confusion_matrix[3,3]>confusion_matrix[3,2] )


    def test_map_to_depth( self ):
        """
        Provided a polygon map with compound pixels (intersections), the matrix representing
        the depth of each pixel 
        """
        map_hw = torch.tensor([[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,0x20304,0x304,4,0],
                               [0,0,0x304,0x304,0x304,4],
                               [0,0,4,4,4,0],
                               [0,0,4,4,0,0]], dtype=torch.int)


        self.assertTrue( torch.equal(
            seglib.map_to_depth( map_hw ),
            torch.tensor([[1,1,1,1,1,1],
                          [1,1,1,1,1,1],
                          [1,1,3,2,1,1],
                          [1,1,2,2,2,1],
                          [1,1,1,1,1,1],
                          [1,1,1,1,1,1]], dtype=torch.int)))


    def test_recover_labels_from_map_value_single_polygon( self ):
        self.assertEqual( seglib.recover_labels_from_map_value( 0 ), [0])
        self.assertEqual( seglib.recover_labels_from_map_value( 3 ), [3])
        self.assertEqual( seglib.recover_labels_from_map_value( 255 ), [255])

    def test_recover_labels_from_map_value_two_polygons( self ):
        self.assertEqual( seglib.recover_labels_from_map_value( 515 ), [2,3] )

    def test_recover_labels_from_map_value_three_polygons( self ):
        self.assertTrue( seglib.recover_labels_from_map_value( 0x20304 ), [2,3,4])




if __name__ == "__main__":
    unittest.main()
