#!/usr/env python3

import pytest
from torch import Tensor
from pathlib import Path
from PIL import Image
import json
from torch import Tensor
import numpy as np
import torch
from functools import partial

import sys

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

import seglib



@pytest.fixture(scope="module")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

def test_dummy( data_path):
    """
    A dummy test, as a sanity check for the test framework.
    """
    print(data_path)
    assert seglib.dummy() == True

def test_line_segmentation_model_not_found(  data_path ):
    """
    An exception should be raised when no segmentation model found.
    """
    model = Path('nowhere_to_be_found.mlmodel')
    input_image = data_path.joinpath('NA-ACK_14201223_01485_r-r1.png')
    with pytest.raises( FileNotFoundError ) as e:
        seglib.line_segment(input_image, model)

def test_binary_mask_from_image_real_img( data_path):
    """
    Binarization function should return a Boolean tensor with shape( h,w )
    """
    input_img = Image.open( data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    mask = seglib.get_mask( input_img )
    assert type(mask) == torch.Tensor 
    assert mask.dtype == torch.bool
    assert mask.shape == tuple(reversed(input_img.size))

def test_binary_mask_from_image_fg_bg():
    """
    Binary map should be 1 for FG pixels, 0 otherwise.
    """
    img_arr = np.full((15,15,3), 200, dtype=np.uint8 ) # background
    img_arr[8,8]=5 # foreground = 1 pixel
    binary_map = seglib.get_mask( Image.fromarray( img_arr) )
    assert binary_map[8,8] == 1 # single FG pixel = T
    assert torch.sum(binary_map).item() == 1 # remaining pixels = F


def test_line_binary_mask_from_img_segmentation_dict( data_path, ndarrays_regression ):
    """
    Provided an image and a segmentation dictionary, should return a boolean mask for all lines.
    """
    input_img = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    segmentation_dict = json.load(open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))
    mask = seglib.line_binary_mask_from_img_segmentation_dict( input_img, segmentation_dict )
    ndarrays_regression.check( { 'mask': mask.numpy() } ) 


def test_line_images_from_img_segmentation_dict_type_checking( data_path, ndarrays_regression ):
    """
    The elements in the pair (image and mask) should both be numpy arrays with shape (H,W,3).
    """
    input_img = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    segmentation_dict = json.load(open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))
    imgs_and_masks = seglib.line_images_from_img_segmentation_dict( input_img, segmentation_dict )
    #ndarrays_regression.check( { 'mask': mask.numpy() } ) 
    assert len(imgs_and_masks) == 4
    assert type(imgs_and_masks[0][0]) is np.ndarray 
    assert type(imgs_and_masks[0][1]) is np.ndarray 
    assert imgs_and_masks[0][0].shape == imgs_and_masks[0][1].shape
    assert imgs_and_masks[0][0].shape[-1] == 3
    assert imgs_and_masks[0][1].shape[-1] == 3


def test_line_images_from_img_segmentation_dict_image_content_checking( data_path, ndarrays_regression ):
    """
    Each line image should match the line polygon's bounding box.
    """
    input_img = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    segmentation_dict = json.load(open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))
    imgs_and_masks = seglib.line_images_from_img_segmentation_dict( input_img, segmentation_dict )
    ndarrays_regression.check( {
        'image1': imgs_and_masks[0][0], 'mask1': imgs_and_masks[0][1],
        'image2': imgs_and_masks[1][0], 'mask2': imgs_and_masks[1][1],
        'image3': imgs_and_masks[2][0], 'mask3': imgs_and_masks[2][1],
        'image4': imgs_and_masks[3][0], 'mask4': imgs_and_masks[3][1],
        })


def test_line_images_from_img_polygon_map_type_checking( data_path, ndarrays_regression ):
    """
    The elements in the pair (image and mask) should both be numpy arrays with shape (H,W,3).
    """
    input_img = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygon_map_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))
    imgs_and_masks = seglib.line_images_from_img_polygon_map( input_img, polygon_map_chw )
    assert len(imgs_and_masks) == 4
    assert type(imgs_and_masks[0][0]) is np.ndarray 
    assert type(imgs_and_masks[0][1]) is np.ndarray 
    assert imgs_and_masks[0][0].shape == imgs_and_masks[0][1].shape
    assert imgs_and_masks[0][0].shape[-1] == 3
    assert imgs_and_masks[0][1].shape[-1] == 3


def test_line_images_from_img_polygon_map_content_checking( data_path, ndarrays_regression ):
    """
    Each line image should match the line polygon's bounding box.
    """
    input_img = Image.open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r')
    polygon_map_chw = torch.load(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt'))
    imgs_and_masks = seglib.line_images_from_img_polygon_map( input_img, polygon_map_chw )
    ndarrays_regression.check( {
        'image1': imgs_and_masks[0][0], 'mask1': imgs_and_masks[0][1],
        'image2': imgs_and_masks[1][0], 'mask2': imgs_and_masks[1][1],
        'image3': imgs_and_masks[2][0], 'mask3': imgs_and_masks[2][1],
        'image4': imgs_and_masks[3][0], 'mask4': imgs_and_masks[3][1],
        })

@pytest.mark.parametrize('input_map,n,expected',
        [ ( torch.tensor([[1]]), 3, np.array([[[1,1,1]]])),
          ( torch.tensor([[1]]), 4, np.array([[[1,1,1,1]]])),
          ( torch.tensor([[ 1, 2, 3],
                          [ 4, 5, 6],
                          [ 7, 8, 9],
                          [10,11,12]]), 3, np.array([[[ 1, 1, 1],[ 2, 2, 2],[ 3, 3, 3]],
                                                     [[ 4, 4, 4],[ 5, 5, 5],[ 6, 6, 6]],
                                                     [[ 7, 7, 7],[ 8, 8, 8],[ 9, 9, 9]],
                                                     [[10,10,10],[11,11,11],[12,12,12]]])),
          ( torch.tensor([[ 1, 2, 3],
                          [ 4, 5, 6],
                          [ 7, 8, 9],
                          [10,11,12]]), 4, np.array([[[ 1, 1, 1, 1],[ 2, 2, 2, 2],[ 3, 3, 3, 3]],
                                                     [[ 4, 4, 4, 4],[ 5, 5, 5, 5],[ 6, 6, 6, 6]],
                                                     [[ 7, 7, 7, 7],[ 8, 8, 8, 8],[ 9, 9, 9, 9]],
                                                     [[10,10,10,10],[11,11,11,11],[12,12,12,12]]]))])
def test_expand_flat_tensor_to_n_channels( input_map, n, expected):
    
    assert np.array_equal( seglib.expand_flat_tensor_to_n_channels( input_map, n ), expected )

    

def test_array_to_rgba_uint8_overflow():
    """
    Conversion of a map of 64-bit integers should raise an exception
    """
    arr = np.random.default_rng().integers(1, 0xffffffff, (5,4), dtype='int')
    arr[2,2]=0xffffffff

    with pytest.raises(TypeError):
        seglib.array_to_rgba_uint8( arr )

def test_array_to_rgba_uint8_incorrect_shape():
    """
    Conversion of a map with more than 2 dimensions should raise an exception.
    """
    arr = np.random.default_rng().integers(1, 0xff, (2,5,4), dtype='int32')

    with pytest.raises(TypeError):
        seglib.array_to_rgba_uint8( arr )

@pytest.mark.parametrize("dtype", ["int64", "uint64", "uint32"])
def test_array_to_rgba_uint8_wrong_dtype(dtype):
    """
    Converting a map that has not 'int32' as dtype raises an exception.
    """
    arr = np.array( [[2,2,2,0,0,0],
                     [2,2,2,0,0,0],
                     [2,2,0x203,3,0,0],
                     [0,0,3,3,3,0],
                     [0,0,0,0,0,0],
                     [0,0x40102,0,0,0,0]], dtype=dtype)

    with pytest.raises(TypeError) as e:
        tensor = seglib.array_to_rgba_uint8( arr )


def test_array_to_rgba_uint8():
    """
    Conversion of a 1-channel array of 32-bit integers yields a 4-channel tensor.
    """
    arr = np.array( [[2,2,2,0,0,0],
                     [2,2,2,0,0,0],
                     [2,2,0x203,3,0,0],
                     [0,0,3,3,3,0],
                     [0,0,0,0,0,0],
                     [0,0x40102,0,0,0,0]], dtype='int32')
    tensor = seglib.array_to_rgba_uint8( arr )
    assert torch.equal( tensor,
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
                       [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8))


def test_polygon_mask_to_polygon_map_32b_store_single_polygon():
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

    assert np.array_equal( label_map,
                   np.array( [[2,2,2,0,0,0],
                              [2,2,2,0,0,0],
                              [2,2,2,0,0,0],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0]], dtype='int32'))

def test_polygon_mask_to_polygon_map_32b_store_too_large_a_label():
    """
    Trying to store a label larger than 255 causes an exception.
    """
    label_map = np.zeros((6,6), dtype='int32')
    polygon_mask = np.array( [[1,1,1,0,0,0],
                              [1,1,1,0,0,0],
                              [1,1,1,0,0,0],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0]], dtype='int32')

    with pytest.raises(OverflowError) as e:
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 256)


def test_polygon_mask_to_polygon_map_32b_store_two_intersecting_polygons():
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
    assert np.array_equal( label_map,
                   np.array( [[2,2,2,0,0,0],
                              [2,2,2,0,0,0],
                              [2,2,0x203,3,0,0],
                              [0,0,3,3,3,0],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0]], dtype='int32'))


def test_polygon_mask_to_polygon_map_32b_store_two_polygons_no_intersection():
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

    assert np.array_equal( label_map,
                   np.array( [[2,2,2,0,0,0],
                              [2,2,2,0,0,0],
                              [2,2,2,3,3,0],
                              [0,0,3,3,3,3],
                              [0,0,0,0,0,0],
                              [0,0,0,0,0,0]], dtype='int32'))

def test_polygon_mask_to_polygon_map_32b_store_three_intersecting_polygons():
    """
    Storing 2 extra polygons (as binary mask + labels) on labeled tensor with overlap yields
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

    assert np.array_equal( label_map,
                   np.array( [[2,2,2,0,0,0],
                              [2,2,2,0,0,0],
                              [2,2,0x20304,0x304,4,0],
                              [0,0,0x304,0x304,0x304,4],
                              [0,0,4,4,4,0],
                              [0,0,4,4,0,0]], dtype='int32'))


@pytest.mark.parametrize('label_values, expected_matrix',[
    ((2,3,4,1), np.array( [[2,2,2,0,0,0],
                           [2,2,2,0,0,0],
                           [2,2,0x2030401,0x304,4,0],
                           [0,0,0x30401,0x304,0x304,4],
                           [0,1,0x401,0x401,4,0],
                           [0,1,0x401,4,0,0]], dtype='int32')),
    ((0xff, 0xfe, 0xfd, 0xfc), np.array( [[0xff,0xff,0xff,0,0,0], 
                                          [0xff,0xff,0xff,0,0,0],
                                          [0xff,0xff,0xfffefdfc,0xfefd,0xfd,0],
                                          [0,0,0xfefdfc,0xfefd,0xfefd,0xfd],
                                          [0,0xfc,0xfdfc,0xfdfc,0xfd,0],
                                          [0,0xfc,0xfdfc,0xfd,0,0]], dtype='int32'))])
def test_polygon_mask_to_polygon_map_32b_store_four_intersecting_polygons(label_values, expected_matrix):
    """
    Storing 4 polygons, both with small and large values (including negative compound labels)
    """
    label_map = np.array( [[label_values[0],label_values[0],label_values[0],0,0,0],
                           [label_values[0],label_values[0],label_values[0],0,0,0],
                           [label_values[0],label_values[0],label_values[0],0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0]], dtype='int32')

    polygon_mask_1 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,1,0,0],
                                [0,0,1,1,1,0],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0]], dtype='int32')
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask_1, label_values[1])

    polygon_mask_2 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,1,1,0],
                                [0,0,1,1,1,1],
                                [0,0,1,1,1,0],
                                [0,0,1,1,0,0]], dtype='int32')
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask_2, label_values[2])

    polygon_mask_3 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,0,0,0],
                                [0,0,1,0,0,0],
                                [0,1,1,1,0,0],
                                [0,1,1,0,0,0]], dtype='int32')
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask_3, label_values[3])

    assert np.array_equal( label_map, expected_matrix)

def test_polygon_mask_to_polygon_map_32b_store_overflow():
    """
    Storing 4 extra polygons (as binary mask + labels) on labeled tensor 
    causes an exception when intersection pixel value overflows.
    """
    label_map = np.array( [[0xff,0xff,0xff,0,0,0],
                           [0xff,0xff,0xff,0,0,0],
                           [0xff,0xff,0xff,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0]], dtype='int32')

    polygon_mask_1 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,1,0,0],
                                [0,0,1,1,1,0],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0]], dtype='int32')
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask_1, 0xfe)

    polygon_mask_2 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,1,1,0],
                                [0,0,1,1,1,1],
                                [0,0,1,1,1,0],
                                [0,0,1,1,0,0]], dtype='int32')
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask_2, 0xfd)

    polygon_mask_3 = np.array( [[0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,1,0,0,0],
                                [0,0,1,0,0,0],
                                [0,1,1,1,0,0],
                                [0,1,1,0,0,0]], dtype='int32')

    with pytest.raises( OverflowError ) as e:
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask_3, 0x102ff+1)


def test_polygon_mask_to_polygon_map_32b_5_polygon_exception():
    """
    Exception raised when trying to store more than 4 polygons on same pixel.
    """
    label_map = np.array( [[1,1,1,0],
                           [1,1,1,0],
                           [1,1,1,0],
                           [0,0,0,0]], dtype='int32')

    polygon_mask = np.array( [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,1],
                              [0,0,0,0]], dtype='int32')

    seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 2)
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 3)
    seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 4)

    with pytest.raises( ValueError ) as e:
        seglib.apply_polygon_mask_to_map(label_map, polygon_mask, 1)

def test_polygon_mask_to_polygon_map_32b_duplicate_label_single_channel():
    """
    Test that a pixel cannot store the same label twice.
    """

    label_map = np.array( [[2,2,2,0],
                           [2,2,2,0],
                           [2,2,2,0],
                           [0,0,0,0]], dtype='int32')

    polygon_mask = np.array( [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,1],
                              [0,0,0,0]], dtype='int32')

    with pytest.raises( ValueError ) as e:
        seglib.apply_polygon_mask_to_map( label_map, polygon_mask, 2)

@pytest.mark.parametrize(('label'),[2, 0x402, 0x40102, 0x204, 0x20401])
def test_polygon_mask_to_polygon_map_32b_duplicate_label_different_channels( label ):
    """
    Test that a pixel cannot store the same label twice.
    """

    label_map = np.array( [[1,1,1,0],
                           [1,1,1,0],
                           [1,label,1,0],
                           [0,0,0,0]], dtype='int32')

    polygon_mask = np.array( [[0,0,0,0],
                              [0,0,0,0],
                              [0,1,1,1],
                              [0,1,0,0]], dtype='int32')

    with pytest.raises( ValueError ) as e:
        seglib.apply_polygon_mask_to_map( label_map, polygon_mask, 2)


def test_retrieve_polygon_mask_from_map_1():
    label_map = torch.tensor([[[2, 2, 2, 0, 0, 0],
                               [2, 2, 2, 0, 0, 0],
                               [2, 2, 4, 4, 4, 0],
                               [0, 0, 4, 4, 4, 4],
                               [0, 0, 4, 4, 4, 0],
                               [0, 0, 4, 4, 0, 0]],
                              [[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 3, 3, 0, 0],
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
                               [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

    expected = torch.tensor( [[False, False, False, False, False, False],
                               [False, False, False, False, False, False],
                               [False, False,  True,  True, False, False],
                               [False, False,  True,  True,  True, False],
                               [False, False, False, False, False, False],
                               [False, False, False, False, False, False]])

    assert torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3), expected)
    # second call ensures that the map is not modified by the retrieval operation
    assert torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3), expected)

def test_retrieve_polygon_mask_from_map_2():
    label_map = torch.tensor([[[2, 2, 2, 0, 0, 0],
                               [2, 2, 2, 0, 0, 0],
                               [2, 2, 4, 4, 4, 0],
                               [0, 3, 4, 4, 4, 4],
                               [0, 0, 3, 4, 4, 0],
                               [0, 0, 4, 4, 0, 0]],
                              [[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 3, 3, 0, 0],
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
                               [0, 0, 0, 0, 0, 3],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)

    assert torch.equal( seglib.retrieve_polygon_mask_from_map(label_map, 3),
                torch.tensor( [[False, False, False, False, False, False],
                               [False, False, False, False, False, False],
                               [False, False,  True,  True, False, False],
                               [False,  True,  True,  True,  True, True],
                               [False, False,  True, False, False, False],
                               [False, False, False, False, False, False]]))


def test_segmentation_polygon_map_from_img_segmentation_dict_label_count(  data_path ):
    """
    seglib.polygon_map_from_img_segmentation_dict(dict, image) should return a tuple (labels, polygons)
    """
    with open( data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r') as segdict_file, Image.open( data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
        segdict = json.load( segdict_file )
        polygons = seglib.polygon_map_from_img_segmentation_dict( input_image, segdict )
        assert type(polygons) is torch.Tensor
        assert torch.max(polygons) == 4
        assert polygons.shape == (4,)+input_image.size[::-1]


def test_segmentation_polygon_map_from_img_json_files(  data_path ):
    img_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    json_file = str(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'))
    polygons = seglib.polygon_map_from_img_json_files( img_file, json_file )
    assert type(polygons) is torch.Tensor
    assert torch.max(polygons) == 4



def test_segmentation_polygon_map_from_img_xml_files(  data_path ):
    img_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    xml_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.xml'))
    polygons = seglib.polygon_map_from_img_xml_files( img_file, xml_file )
    assert type(polygons) is torch.Tensor
    assert torch.max(polygons) == 4



def test_line_binary_mask_from_img_json_files(  data_path ):
    img_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    json_file = str(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'))
    mask = seglib.line_binary_mask_from_img_json_files( img_file, json_file )
    assert type(mask) is torch.Tensor
    assert mask.dtype == torch.bool

def test_line_binary_mask_from_img_xml_files(  data_path ):
    img_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    xml_file = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.xml'))
    mask = seglib.line_binary_mask_from_img_xml_files( img_file, xml_file )
    assert type(mask) is torch.Tensor
    assert mask.dtype == torch.bool



def test_union_intersection_count_two_maps():
    """
    Provided two label maps that each encode (potentially overlapping) polygons, yield 
    intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
    and j ∈ map2.
    Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
    """
    # Pred. labels: 2,3,4
    map1 = seglib.array_to_rgba_uint8(
            np.array([[2,2,2,0,0,0],
                      [2,2,2,0,0,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,0,0x304,0x304,0x304,4],
                      [0,0,4,4,4,0],
                      [0,0,4,0x402,0,0]], dtype='int32'))

    # GT labels: 2,3,4
    map2 = seglib.array_to_rgba_uint8(
            np.array([[0,2,2,0,0,0],
                      [2,2,4,2,2,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,3,0x304,0x304,0x304,4],
                      [0,0,3,4,4,0],
                      [0,0,0x204,4,0,0]], dtype='int32'))

    pixel_count = seglib.polygon_pixel_metrics_two_maps( map1, map2 )[:,:,:2]

    c2l, c2r = 8.5+1/3, 8.5+1/3
    c3l, c3r = 2.0+1/3, 4.0+1/3
    c4l, c4r = 8.5+1/3, 8.5+1/3
    i22, i23, i24 = 6.0+1/3, 1/3, 1.5+1/3
    i32, i33, i34 = 1/3, 2.0+1/3, 2.0+1/3
    i42, i43, i44 = 0.5+1/3, 3.0+1/3, 7.0+1/3
    u22, u23, u24 = c2l+c2r-i22, c2l+c3r-i23, c2l+c4r-i24
    u32, u33, u34 = c3l+c2r-i32, c3l+c3r-i33, c3l+c4r-i34
    u42, u43, u44 = c4l+c2r-i42, c4l+c3r-i43, c4l+c4r-i44

    expected = np.array([[[ i22, u22],   # 2,2
                          [ i23, u23],   # 2,3
                          [ i24, u24]],  # 2,4
                         [[ i32, u32],   # ...
                          [ i33, u33],
                          [ i34, u34]],  # 3,4
                         [[ i42, u42],   # ...
                          [ i43, u43],
                          [ i44, u44]]]) # 4,4

    # Note: we're comparing float value here
    assert np.all(np.isclose( pixel_count, expected ))

def test_union_intersection_count_two_maps_more_labels_in_pred():
    """
    Provided two label maps that each encode (potentially overlapping) polygons, yield 
    intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
    and j ∈ map2.
    Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
    The metrics function should deal correctly with GT labels that are missing in the predicted map.
    """
    # Pred. labels: 2,3,4,5
    map1 = seglib.array_to_rgba_uint8(np.array(
                     [[2,2,2,0,0,0],
                      [2,2,2,0,0,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,0x502,0x304,0x304,0x304,4],
                      [5,0x405,4,4,4,0], 
                      [0,0,4,0x402,0,0]], dtype='int32'))

    # GT labels: 2,3,4
    map2 = seglib.array_to_rgba_uint8(np.array(
                     [[0,2,2,0,0,0],
                      [2,2,4,2,2,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,3,0x304,0x304,0x304,4],
                      [0,0,3,4,4,0],
                      [0,0,0x204,4,0,0]], dtype='int32'))

    pixel_count = seglib.polygon_pixel_metrics_two_maps( map1, map2 )[:,:,:2]

    c2l, c2r = 9.0+1/3, 8.5+1/3
    c3l, c3r = 2.0+1/3, 4.0+1/3
    c4l, c4r = 9.0+1/3, 8.5+1/3
    c5l, c5r = 2.0, 0.0
    i22, i23, i24 = 6.0+1/3, .5+1/3, 1.5+1/3
    i32, i33, i34 = 1/3, 2.0+1/3, 2.0+1/3
    i42, i43, i44 = .5+1/3, 3.0+1/3, 7.0+1/3
    i52, i53, i54 = 0, .5, 0
    u22, u23, u24 = c2l+c2r-i22, c2l+c3r-i23, c2l+c4r-i24
    u32, u33, u34 = c3l+c2r-i32, c3l+c3r-i33, c3l+c4r-i34
    u42, u43, u44 = c4l+c2r-i42, c4l+c3r-i43, c4l+c4r-i44
    u52, u53, u54 = c5l+c2r-i52, c5l+c3r-i53, c5l+c4r-i54

    #print("\n"+repr( pixel_count ))
    expected =    np.array([[[ i22, u22],   # 2,2
                             [ i23, u23],   # 2,3
                             [ i24, u24]],  # 2,4
                            [[ i32, u32],   # ...
                             [ i33, u33],
                             [ i34, u34]],  # 3,4
                            [[ i42, u42],   # ...
                             [ i43, u43],
                             [ i44, u44]],
                            [[ i52, u52],   # ...
                             [ i53, u53],
                             [ i54, u54]]]) # 4,4
    #print("\n"+repr(expected))

    # Note: we're comparing float value here
    assert np.all(np.isclose( pixel_count, expected ))

def test_union_intersection_count_two_maps_more_labels_in_GT():
    """
    Provided two label maps that each encode (potentially overlapping) polygons, yield 
    intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
    and j ∈ map2.
    Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
    The metrics function should deal correctly with GT labels that are missing in the predicted map.
    """
    # Pred. labels: 2,3,4
    map1 = seglib.array_to_rgba_uint8(np.array( 
                     [[0,2,2,0,0,0],
                      [2,2,4,2,2,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,3,0x304,0x304,0x304,4],
                      [0,0,3,4,4,0],
                      [0,0,0x204,4,0,0]], dtype='int32'))

    # GT labels: 2,3,4,5
    map2 = seglib.array_to_rgba_uint8(np.array(
                     [[2,2,2,0,0,0],
                      [2,2,2,0,0,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,0x502,0x304,0x304,0x304,4],
                      [5,0x405,4,4,4,0], 
                      [0,0,4,0x402,0,0]], dtype='int32'))

    pixel_count = seglib.polygon_pixel_metrics_two_maps( map1, map2 )[:,:,:2]

    c2l, c2r = 8+1/3+.5, 8+1/3+.5*2
    c3l, c3r = 1/3+2+.5*4, 1/3+.5*4
    c4l, c4r = 1/3+5*.5+6, 1/3+6*.5+6
    c5l, c5r = 0, 1+2*.5
    i22, i23, i24, i25 = 6+1/3, 1/3, 1/3+.5, 0
    i32, i33, i34, i35 = 1/3+.5, 1/3+.5*4, 1/3+.5*4+1, .5
    i42, i43, i44, i45 = 1+1/3+.5, 1/3+.5*4, 1/3+.5*6+4, 0
    u22, u23, u24, u25 = c2l+c2r-i22, c2l+c3r-i23, c2l+c4r-i24, c2l+c5r-i25
    u32, u33, u34, u35 = c3l+c2r-i32, c3l+c3r-i33, c3l+c4r-i34, c3l+c5r-i35
    u42, u43, u44, u45 = c4l+c2r-i42, c4l+c3r-i43, c4l+c4r-i44, c4l+c5r-i45

    #print("\n"+repr( pixel_count ))
    expected =    np.array([[[ i22, u22],  # 2,2
                             [ i23, u23],  # 2,3
                             [ i24, u24],  # 2,4
                             [ i25, u25]], # 2,5
                            [[ i32, u32],  # ...
                             [ i33, u33],
                             [ i34, u34],
                             [ i35, u35]], # 3,5
                            [[ i42, u42],  # ...
                             [ i43, u43],
                             [ i44, u44],
                             [ i45, u45]]])
    #print("\n"+repr(expected))

    # Note: we're comparing float value here
    assert np.all(np.isclose( pixel_count, expected ))

def test_precision_recall_two_maps():
    """
    Provided two label maps that each encode (potentially overlapping) polygons, yield 
    intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
    and j ∈ map2.
    Shared pixels in each map (i.e. overlapping polygons) are counted independently for each polygon.
    """
    # map1 (_l_)  = Pred, map2 (_r_) = GT
    map1 = seglib.array_to_rgba_uint8(np.array([[2,2,2,0,0,0],
                      [2,2,2,0,0,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,0,0x304,0x304,0x304,4],
                      [0,0,4,4,4,0],
                      [0,0,4,0x402,0,0]], dtype='int32'))

    map2 = seglib.array_to_rgba_uint8(np.array( [[0,2,2,0,0,0],
                      [2,2,4,2,2,0],
                      [2,2,0x20304,0x304,4,0],
                      [0,3,0x304,0x304,0x304,4],
                      [0,0,3,4,4,0],
                      [0,0,0x204,4,0,0]], dtype='int32'))

    metrics = seglib.polygon_pixel_metrics_two_maps( map1, map2 )
    intersection_union, precision_recall = metrics[:,:,:2], metrics[:,:,2:]

    c2l, c2r = 8+1/3+.5, 8+1/3+.5
    c3l, c3r = 1/3+.5*4, 1/3+2+.5*4
    c4l, c4r = 1/3+5*.5+6, 1/3+5*.5+6
    i22, i23, i24 = 6+1/3, 1/3, 1+1/3+.5
    i32, i33, i34 = 1/3, 1/3+.5*4, 1/3+.5*4
    i42, i43, i44 = 1/3+.5, 1/3+.5*4+1, 1/3+.5*6+4
    u22, u23, u24 = c2l+c2r-i22, c2l+c3r-i23, c2l+c4r-i24
    u32, u33, u34 = c3l+c2r-i32, c3l+c3r-i33, c3l+c4r-i34
    u42, u43, u44 = c4l+c2r-i42, c4l+c3r-i43, c4l+c4r-i44

    expected_intersection_union = np.array(
                        [[[ i22, u22],   # 2,2
                          [ i23, u23],   # 2,3
                          [ i24, u24]],  # 2,4
                         [[ i32, u32],   # ...
                          [ i33, u33],
                          [ i34, u34]],  # 3,4
                         [[ i42, u42],   # ...
                          [ i43, u43],
                          [ i44, u44]]]) # 4,4

    expected = np.array([[[ i22/c2l, i22/c2r],   # 2,2
                          [ i23/c2l, i23/c3r],   # 2,3
                          [ i24/c2l, i24/c4r]],  # 2,4
                         [[ i32/c3l, i32/c2r],   # ...
                          [ i33/c3l, i33/c3r],
                          [ i34/c3l, i34/c4r]],  # 3,4
                         [[ i42/c4l, i42/c2r],   # ...
                          [ i43/c4l, i43/c3r],
                          [ i44/c4l, i44/c4r]]]) # 4,4

    # Note: we're comparing float value here
    assert np.all(np.isclose( intersection_union, expected_intersection_union, 1e-4 ))
    assert np.all(np.isclose( precision_recall, expected, 1e-4 ))

def test_get_polygon_pixel_metrics_wrong_pred_type():
    """
    First map should be a 4-channel tensor
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

    with pytest.raises( TypeError ):
        seglib.polygon_pixel_metrics_from_polygon_maps_and_mask(map1,map2)

def test_get_polygon_pixel_metrics_wrong_gt_type():
    """
    Second map should be a 4-channel tensor
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

    with pytest.raises( TypeError ):
        seglib.polygon_pixel_metrics_from_polygon_maps_and_mask(map1,map2)


def test_get_polygon_pixel_metrics_different_map_shapes():
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

    with pytest.raises( TypeError ):
        seglib.polygon_pixel_metrics_from_polygon_maps_and_mask(map1,map2)


@pytest.mark.parametrize("distance", [0, 2, 5])
def test_get_polygon_pixel_metrics_from_maps_and_mask_small_image(  data_path, distance, ndarrays_regression ):
    """
    On an actual image, with polygons loaded from serialized tensors, only a few sanity checks for testing
    """
    polygon_pred = torch.load(str(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced_polygon_map.pt')))
    polygon_gt = torch.load(str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_polygon_map.pt')))
    binary_mask = torch.load(str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced_binarized.pt')))

    metrics = seglib.polygon_pixel_metrics_from_polygon_maps_and_mask(polygon_pred, polygon_gt, binary_mask, label_distance=distance)

    ndarrays_regression.check( {'pixel_metrics': metrics })
    #assert metrics.dtype == np.float32 
    #assert np.all( metrics[:,:,0].diagonal() != 0 ) # intersections
    #assert np.all( metrics[:,:,1] != 0 ) # unions

def test_get_polygon_pixel_metrics_from_img_segmentation_dict_rough_check(  data_path, ndarrays_regression ):
    """
    On an actual image, with polygon loaded from JSON dictionaries, only a few sanity checks for testing
    """
    input_img = Image.open( data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    dict_pred = json.load( open(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))
    dict_gt = json.load( open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))

    metrics = seglib.polygon_pixel_metrics_from_img_segmentation_dict(input_img, dict_pred, dict_gt)

    ndarrays_regression.check( {'pixel_metrics': metrics })
    #assert metrics.dtype == np.float32
    #assert np.all( metrics[:,:,1] != 0 ) # unions

def test_get_polygon_pixel_metrics_from_img_segmentation_dict_content_check(  data_path, ndarrays_regression ):
    """
    On an actual image, with polygon loaded from JSON dictionaries, only a few sanity checks for testing
    """
    input_img = Image.open( data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    dict_pred = json.load( open(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_reduced.json'), 'r'))
    dict_gt = json.load( open(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'), 'r'))

    metrics = seglib.polygon_pixel_metrics_from_img_segmentation_dict(input_img, dict_pred, dict_gt)

    ndarrays_regression.check( { 'pixel_metrics': metrics } )


@pytest.mark.fail_slow('30s')
@pytest.mark.parametrize("distance", [2, 7])
def test_polygon_pixel_metrics_from_full_charter(  data_path, distance ):
    """
    On an full page, with polygons loaded from serialized tensors, only a few sanity checks for testing.
    + Crude check for performance for a couple of label distance.
    """
    polygons_pred = torch.load(str(data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20_polygon_map.pt')))
    polygons_gt = torch.load(str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_polygon_map.pt')))
    binary_mask = torch.load(str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_binarized.pt')))

    # for any polygon label in predicted map, counting intersections with GT map for labels in [l-d .. l+2d]
    metrics = seglib.polygon_pixel_metrics_from_polygon_maps_and_mask(polygons_pred, polygons_gt, binary_mask, label_distance=distance)
    assert metrics.dtype == np.float32 
    # union counts should never be 0
    assert np.all( metrics[:,:,1] != 0 )


@pytest.mark.parametrize("thld,expected_scores", [ 
    (.1, (3.0, 0.0, 0.0 , 1.0, 1.0)),
    (.5, (3.0, 0.0, 0.0, 1.0, 1.0)),
    (.9, (0.0, 3.0, 2.0, 0.0, 0.0))])
def test_polygon_pixel_metrics_to_line_based_scores(thld, expected_scores):
    """
    Made-up matrix, constructed from the tests above (not from an actual image).
    """
    metrics = np.array([[[ 6.3333335 , 11.833334  ,  0.7169811 ,  0.67857146],
                         [ 0.33333334, 10.833334  ,  0.03773585,  0.14285713],
                         [ 0.8333334 , 17.333334  ,  0.09433962,  0.08928571],
                         [ 0.        , 10.833334  ,  0.        ,  0.        ]],
                        [[ 0.8333334 , 12.833333  ,  0.1923077 ,  0.08928572],
                         [ 2.3333335 ,  4.3333335 ,  0.53846157,  1.        ],
                         [ 3.3333335 , 10.333334  ,  0.7692308 ,  0.35714284],
                         [ 0.5       ,  5.8333335 ,  0.11538461,  0.25      ]],
                        [[ 1.8333334 , 16.333334  ,  0.20754716,  0.19642858],
                         [ 2.3333335 ,  8.833334  ,  0.26415095,  1.        ],
                         [ 7.3333335 , 10.833334  ,  0.83018863,  0.78571427],
                         [ 0.        , 10.833334  ,  0.        ,  0.        ]]],
                        dtype=np.float32)

    scores = seglib.polygon_pixel_metrics_to_line_based_scores(metrics, threshold=thld)
    assert scores == expected_scores


def test_polygon_pixel_metrics_to_pixel_based_scores():
    """
    Made-up matrix, constructed from the tests above.
    """
    metrics = np.array([[[ 6.3333335 , 11.833334  ,  0.7169811 ,  0.67857146],
                         [ 0.33333334, 10.833334  ,  0.03773585,  0.14285713],
                         [ 0.8333334 , 17.333334  ,  0.09433962,  0.08928571],
                         [ 0.        , 10.833334  ,  0.        ,  0.        ]],
                        [[ 0.8333334 , 12.833333  ,  0.1923077 ,  0.08928572],
                         [ 2.3333335 ,  4.3333335 ,  0.53846157,  1.        ],
                         [ 3.3333335 , 10.333334  ,  0.7692308 ,  0.35714284],
                         [ 0.5       ,  5.8333335 ,  0.11538461,  0.25      ]],
                        [[ 1.8333334 , 16.333334  ,  0.20754716,  0.19642858],
                         [ 2.3333335 ,  8.833334  ,  0.26415095,  1.        ],
                         [ 7.3333335 , 10.833334  ,  0.83018863,  0.78571427],
                         [ 0.        , 10.833334  ,  0.        ,  0.        ]]],
                        dtype=np.float32)

    scores = seglib.polygon_pixel_metrics_to_pixel_based_scores( metrics )
    assert np.all(np.isclose( scores, (0.23780487, 0.5925925788565435) ))


def test_polygon_pixel_metrics_to_line_based_scores_full_charter(  data_path ):
    """
    On an actual image, a sanity check
    """
    with open(data_path.joinpath('full_charter_pixel_metrics.npy'), 'rb') as f:
        metrics = np.load( f )

        scores = seglib.polygon_pixel_metrics_to_line_based_scores( metrics, .5 )
        assert scores == (32.0, 0.0, 0.0, 1.0, 1.0)

def test_polygon_pixel_metrics_to_pixel_based_scores_full_charter(  data_path ):
    """
    On an actual image, a sanity check
    """
    with open(data_path.joinpath('full_charter_pixel_metrics.npy'), 'rb') as f:
        metrics = np.load( f )

        scores = seglib.polygon_pixel_metrics_to_pixel_based_scores( metrics )
        assert scores[0] > .25
        assert scores[1] > .5 


def test_map_to_depth():
    """
    Provided a polygon map with compound pixels (intersections), the matrix representing
    the depth of each pixel should have 1 for 1-polygon pixels, 2 for 2-polygon pixels, etc.
    """
    map_hw = seglib.array_to_rgba_uint8(np.array([[2,2,2,0,0,0],
                           [2,2,2,0,0,0],
                           [2,2,0x20304,0x304,4,0],
                           [0,0,0x304,0x304,0x304,4],
                           [0,0,4,4,4,0],
                           [0,0,4,4,0,0]], dtype='int32'))

    depth_map = seglib.map_to_depth( map_hw )

    assert torch.equal(
        depth_map,
        torch.tensor([[1,1,1,1,1,1],
                      [1,1,1,1,1,1],
                      [1,1,3,2,1,1],
                      [1,1,2,2,2,1],
                      [1,1,1,1,1,1],
                      [1,1,1,1,1,1]], dtype=torch.int))

def test_segmentation_dict_from_xml(  data_path ):
    """
    Conversion between PageXML segmentation output and JSON/Python dictionary should keep the lines.
    """
    pagexml = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.xml'))

    segdict = seglib.segmentation_dict_from_xml( pagexml )

    assert len(segdict['lines']) == 4
    assert [ l['line_id'] for l in segdict['lines'] ] == ['r1l1', 'r1l2', 'r1l3', 'r1l4'] 
    assert [ len(l['baseline']) for l in segdict['lines'] ] == [21, 21, 21, 21] 


@pytest.mark.parametrize(
        'label,expected',[
            (0, []), 
            (255, [255]),
            (3, [3]), 
            (0x302, [3,2]), 
            (0x203, [2,3]),
            (0x40205, [4,2,5]),
            (0x1fffffff,[31,255,255,255]),
            (0xffffffff,[255,255,255,255]),
            ])
def test_recover_labels_from_map_value_single_polygon(  label, expected ):
    assert seglib.recover_labels_from_map_value( label ) == expected


def test_mask_from_polygon_map_functional():

    t=torch.tensor([[[  1,   1,   1],
                     [  1,   1,   0],
                     [  0,   0,   0],
                     [  0,   0,   0],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  2,   0,   0],
                     [  2,   2,   2],
                     [  0,   0,   0],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  0,   0,   3],
                     [  3,   3,   3],
                     [  0,   3,   3],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  0,   0,   0],
                     [  0,   0,   0],
                     [  4,   4,   0],
                     [  0,   4,   0]]], dtype=torch.uint8 )

    # select every nth positive label
    nth = lambda m, n: np.logical_and( m % n == 0, m != 0)
    expected = np.array([[False, False, False],
                         [False, False, False],
                         [False, False, False],
                         [True,  True, False],
                         [False,  True, False]])
    assert np.array_equal( seglib.mask_from_polygon_map_functional(t, partial(nth, n=4)), expected )

def test_mask_from_polygon_map_functional_wrong_type():
    """
    Passing a Numpy array as an input causes an exception.
    """

    t=np.array([[[  1,   1,   1],
                 [  1,   1,   0],
                 [  0,   0,   0],
                 [  0,   0,   0],
                 [  0,   0,   0]],
                [[  0,   0,   0],
                 [  2,   0,   0],
                 [  2,   2,   2],
                 [  0,   0,   0],
                 [  0,   0,   0]],
                [[  0,   0,   0],
                 [  0,   0,   3],
                 [  3,   3,   3],
                 [  0,   3,   3],
                 [  0,   0,   0]],
                [[  0,   0,   0],
                 [  0,   0,   0],
                 [  0,   0,   0],
                 [  4,   4,   0],
                 [  0,   4,   0]]], dtype='uint8' )

    with pytest.raises(TypeError) as e:
            seglib.mask_from_polygon_map_functional(t, lambda m: m % 2 != 0)


def test_mask_from_polygon_map_functional_wrong_shape():
    """
    Tensor with wrong shape (eg. (H,W,C)) causes an exception."
    """

    t=torch.tensor([[[  1,   1,   1],
                     [  1,   1,   0],
                     [  0,   0,   0],
                     [  0,   0,   0],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  2,   0,   0],
                     [  2,   2,   2],
                     [  0,   0,   0],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  0,   0,   3],
                     [  3,   3,   3],
                     [  0,   3,   3],
                     [  0,   0,   0]],
                    [[  0,   0,   0],
                     [  0,   0,   0],
                     [  0,   0,   0],
                     [  4,   4,   0],
                     [  0,   4,   0]]], dtype=torch.uint8 ).permute(1,2,0)

    with pytest.raises(TypeError) as e:
            seglib.mask_from_polygon_map_functional(t, lambda m: m % 2 != 0)

def test_mask_from_polygon_map_functional_all_line_images( data_path ):
    """
    Selecting all lines in the polygon map through the generic function should give 
    the same result as constructing the mask directly from the segmentation dictionary.
    """

    input_img = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'))
    dict_gt = str(data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.json'))

    polygon_map = seglib.polygon_map_from_img_json_files( input_img, dict_gt )

    assert torch.equal( seglib.mask_from_polygon_map_functional(polygon_map, lambda m: m > 0),
                        seglib.line_binary_mask_from_img_json_files( input_img, dict_gt))
