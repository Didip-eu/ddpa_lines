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

    def test_array_to_rgba_uint8( self ):
        """
        Conversion of a 1-channel 32-bit unsigned int array yields a 4-channel tensor.
        """
        arr = np.array( [[2,2,2,0,0,0],
                         [2,2,2,0,0,0],
                         [2,2,515,3,0,0],
                         [0,0,3,3,3,0],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0]], dtype='uintc')
        tensor = seglib.array_to_rgba_uint8( arr )
        self.assertTrue( torch.equal( tensor,
            torch.tensor([[[2, 2, 2, 0, 0, 0],
                           [2, 2, 2, 0, 0, 0],
                           [2, 2, 3, 3, 0, 0],
                           [0, 0, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]],

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
                           [0, 0, 0, 0, 0, 0]],

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
                                [0, 0, 0, 0, 0, 0]],

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
                                [0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

        self.assertTrue( np.array_equal( seglib.rgba_uint8_to_hw_tensor( tensor ),
                        torch.Tensor( [[2,2,2,0,0,0],
                                   [2,2,2,0,0,0],
                                   [2,2,515,3,0,0],
                                   [0,0,3,3,3,0],
                                   [0,0,0,0,0,0],
                                   [0,0,0,0,0,0]])))


    def test_polygon_mask_to_polygon_map_32b_store_single_polygon(self):
        """
        Storing polygon (as binary mask + label) on empty tensor yields correct map
        """
        label_map = np.zeros((6,6), dtype='uintc')
        polygon_mask = np.array( [[1,1,1,0,0,0],
                                  [1,1,1,0,0,0],
                                  [1,1,1,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 2)
        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')))

    def test_polygon_mask_to_polygon_map_32b_store_two_intersecting_polygons(self):
        """
        Storing extra polygon (as binary mask + labels) on labeled tensor yields correct map
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='uintc')

        polygon_mask = np.array( [[0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,1,1,0,0],
                                  [0,0,1,1,1,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')

        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)

        # intersecting pixel has value (l1<<8) + l2
        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,515,3,0,0],
                                  [0,0,3,3,3,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')))


    def test_polygon_mask_to_polygon_map_32b_store_two_polygons_no_intersection(self):
        """
        Storing extra polygon (as binary mask + labels) on labeled tensor with overlap yields
        map with intersection label = (l1<<8) + l2
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='uintc')

        polygon_mask = np.array( [[0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,1,1,0],
                                  [0,0,1,1,1,1],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')

        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,2,3,3,0],
                                  [0,0,3,3,3,3],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')))

    def test_polygon_mask_to_polygon_map_32b_store_two_polygons_boolean_mask(self):
        """
        Using a boolean mask should be possible.
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='uintc')

        polygon_mask = np.array( [[0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,1,1,0],
                                  [0,0,1,1,1,1],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='bool')

        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,2,3,3,0],
                                  [0,0,3,3,3,3],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]], dtype='uintc')))

    def test_polygon_mask_to_polygon_map_32b_store_three_intersecting_polygons(self):
        """
        Storing two extra polygons (as binary mask + labels) on labeled tensor with overlap yields
        map with intersection labels l = (l1<<8) + l2, l'=(l2<<8)+l3, l''=(l'<<8)+l2)
        """
        label_map = np.array( [[2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [2,2,2,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype='uintc')

        polygon_mask_1 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,0,0,0,0],
                                    [0,0,0,0,0,0]], dtype='uintc')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_1, 3)

        polygon_mask_2 = np.array( [[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,1,1],
                                    [0,0,1,1,1,0],
                                    [0,0,1,1,0,0]], dtype='uintc')
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_2, 4)

        self.assertTrue( np.array_equal( label_map,
                       np.array( [[2,2,2,0,0,0],
                                  [2,2,2,0,0,0],
                                  [2,2,131844,772,4,0],
                                  [0,0,772,772,772,4],
                                  [0,0,4,4,4,0],
                                  [0,0,4,4,0,0]], dtype='uintc')))

    def test_recover_labels_from_map_value_single_polygon( self ):
        self.assertEqual( seglib.recover_labels_from_map_value( 0 ), [0])
        self.assertEqual( seglib.recover_labels_from_map_value( 3 ), [3])
        self.assertEqual( seglib.recover_labels_from_map_value( 255 ), [255])

    def test_recover_labels_from_map_value_two_polygons( self ):
        self.assertEqual( seglib.recover_labels_from_map_value( 515 ), [2,3] )

    def test_recover_labels_from_map_value_three_polygons( self ):
        self.assertTrue( seglib.recover_labels_from_map_value( 131844 ), [2,3,4])


    def test_retrieve_polygon_mask_from_map_no_binary_mask_1( self ):
        label_map = torch.tensor( [[2,2,2,0,0,0],
                                   [2,2,2,0,0,0],
                                   [2,2,131844,772,4,0],
                                   [0,0,772,772,772,4],
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
                                   [2,2,131844,772,4,0],
                                   [0,3,772,772,772,4],
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
            lbl, polygons = seglib.dict_to_polygon_map( segdict, input_image )
            self.assertTrue( lbl==4 and type(polygons) is torch.Tensor )


    def test_segmentation_dict_to_polygon_map_polygon_img_type( self ):
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_map( segdict, input_image )
            self.assertEqual( polygons.shape, (4,)+input_image.size[::-1])

    def Dtest_segmentation_dict_to_polygons_lines( self ):
        """
        To be removed (visualization code)
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_lines( segdict, input_image )
            self.assertTrue( True )


    def Dtest_polygon_counts_from_array( self ):
        """
        Provided a label map and a foreground, should yield correct polygon counts,
        with intersection pixels distributed among respective labels.
        """

        print('test_polygon_counts_from_array()')
        input_img = Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
        #print('input_img.size = {} (PIL.Image - WH)'.format( input_img.size ))

        dict_pred = json.load( open(self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))

        # input is a PIL Image, outputs are tensors
        label_count, polygon_img = seglib.dict_to_polygon_map( dict_pred, input_img )
        mask = seglib.get_mask( input_img )

        counts = seglib.get_polygon_counts_from_array(polygon_img, mask )

        self.assertEqual( counts, {0: 3458981, 1: 16750, 2: 15028, 4: 14978, 3: 17511})


    def test_union_intersection_count_two_map( self ):
        """
        Provided two label maps that each encode (potentially overlapping) polygons, yield 
        intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
        and j ∈ map2.
        Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
        """
        map1 = torch.tensor([[2,2,2,0,0,0],
                          [2,2,2,0,0,0],
                          [2,2,131844,772,4,0],
                          [0,0,772,772,772,4],
                          [0,0,4,4,4,0],
                          [0,0,4,4,0,0]], dtype=torch.int)

        map2 = torch.tensor( [[0,2,2,0,0,0],
                          [2,2,4,2,2,0],
                          [2,2,131844,772,4,0],
                          [0,3,772,772,772,4],
                          [0,0,3,4,4,0],
                          [0,0,4,4,0,0]], dtype=torch.int)

        pixel_count = seglib.union_intersection_count_two_maps( map1, map2 )

        self.assertTrue( torch.equal(
            pixel_count,
            torch.tensor([[[ 0, 0],  # 1,1
                           [ 0, 9],  # 1,2
                           [ 0, 7],  # ...
                           [ 0,12]], # 1,4
                          [[ 0, 9],  # 2,1
                           [ 7,11],  # ...
                           [ 1,15],
                           [ 2,19]], # 2,4
                          [[ 0, 5],  # 3,1
                           [ 1,13],  # ...
                           [ 5, 7],
                           [ 5,12]], # 3,4
                          [[ 0,12],  # 4,1
                           [ 1,20],  # ...
                           [ 6,13],
                           [11,13]]]))) # 4,4


    def test_zline_segmentation_confusion_matrix( self ):
        """
        """

        print('test_line_segmentation_confusion_matrix()')
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

        confusion_matrix = seglib.get_confusion_matrix_from_polygon_maps(polygon_gt, polygon_pred, mask)
        #torch.save( confusion_matrix, self.data_path.joinpath('confusion_matrix.pt') )

        print(confusion_matrix)

        self.assertTrue( type(confusion_matrix) is torch.Tensor )


    def Dtest_evaluate( self ):
        pass




if __name__ == "__main__":
    unittest.main()
