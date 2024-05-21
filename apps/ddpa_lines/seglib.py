from kraken import blla
from kraken.lib import vgsl
from PIL import Image, ImageDraw
from pathlib import Path
import torchvision

#warning.warn(_BETA_TRANSFORMS_WARNING)
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
from torch import Tensor

import numpy as np
import collections
from typing import Tuple, Callable
import itertools
import random
import torch
import numpy.ma as ma

import skimage as ski
from matplotlib import pyplot as plt


RED = (255,0,0)
GREEN = (0,255,0)
YELLOW = (0,255,255)
BLUE = (0,0,255)

__LABEL_SIZE__=8

def line_segment(img: Image.Image, model_path: str):
    r"""
    Args:
        img (Image.Image): input image
        model_path (str): model location

    Output:
        tuple: a pair with the maximum value of the line labels and the polygons rendered as an image.
    """

    if not Path( model_path ).exists():
        raise FileNotFoundError("Cound not find model file", model_path)
    # you want to load the model once (loaded model as a parameter)
    return dict_to_polygon_map( blla.segment( img, model=vgsl.TorchVGSLModel.load_model( model_path )), img )


def get_mask( img_whc: Image.Image, thresholding_alg: Callable=ski.filters.threshold_otsu ) -> torch.Tensor:
    """
    Compute a binary mask from an image, using the given thresholding algorithm: FG=1s, BG=0s

    Args:
        img (PIL image): input image
    Output:
        torch.Tensor: a binary map with FG pixels=1 and BG=0.
    """
    img_hwc= np.array( img_whc )
    threshold = thresholding_alg( ski.color.rgb2gray( img_hwc) if img_hwc.shape[2]>1 else img_hwc )*255
    img_bin_hw = torch.tensor( (img_hwc < threshold)[:,:,0], dtype=torch.bool )

    return img_bin_hw


def get_confusion_matrix_from_polygon_maps(polygon_img_gt: torch.Tensor, polygon_img_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    # convert to flat 32-bit image
    polygon_img_gt, polygon_img_pred = [ rgba_uint8_to_hw_tensor( polygon_img ) * mask for polygon_img in (polygon_img_gt, polygon_img_pred) ]
    pixel_counts = union_intersection_count_two_maps( polygon_img_gt, polygon_img_pred )

    confusion_matrix = np.ma.MaskedArray( pixel_counts[:,:,0]/pixel_counts[:,:,1], fill_value=0.0 )

    return torch.from_numpy( np.ma.fix_invalid(confusion_matrix) )

def retrieve_polygon_mask_from_map( label_map_hw: torch.Tensor, label: int, binary_mask_hw: torch.Tensor=None) -> torch.Tensor:
    """
    From a label map (that may have compound pixels representing polygon intersections),
    compute a binary mask that covers _all_ pixels for the label, whether they belong to an
    intersection or not.

    Args:
        label_mask_hw (torch.Tensor): a flat map with labeled polygons, with potential overlaps.
        label (int): the label to be selected.

    Output:
        torch.Tensor: a flat, single-label map for the polygon of choice.
    """
    label_map_np = label_map_hw.numpy()
    label_bitmask = 2**__LABEL_SIZE__-1
    polygon_mask_hw = np.zeros( label_map_np.shape, dtype='bool')
    # maximum number of intersecting labels for a single pixel
    # (determine how many times we r-shift the values)
    DEPTH = 3
    for d in range(DEPTH):
        polygon_mask_hw += ((label_map_np >> (__LABEL_SIZE__ * d)) & label_bitmask) == label

    return torch.tensor( polygon_mask_hw )


def dict_to_polygon_map( segmentation_dict: dict, img: Image.Image ) -> Tuple[int, torch.Tensor]:
    """
    Store line polygons into a tensor, as pixel maps.

    Args:
        segmentation_dict (dict): kraken's segmentation output, i.e. a dicitonary of the form::

             {  'text_direction': '$dir',
             'type': 'baseline',
             'lines': [
               {'baseline': [[x0, y0], [x1, y1], ...], 'boundary': [[x0, y0], [x1, y1], ... [x_m, y_m]]},
               ...
             ]
             'regions': [ ... ] }
        img (Image.Image): the input image

    Output:
            tuple: a pair with the maximum value of the line labels and the polygons rendered as a 4-channel image (a tensor).
    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D matrix of 32-bit integers 
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    label_map = np.zeros( img.size[::-1], dtype='int32' )

    # rendering polygons 
    label = 0
    for lbl, polyg in enumerate( polygon_boundaries ):
        label = lbl+1
        polyg_mask = ski.draw.polygon2mask( img.size[::-1], polyg )
        apply_polygon_mask_to_map( label_map, polyg_mask, label )

    #ski.io.imshow( polygon_img.transpose() )

    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( label_map )
    #ski.io.imshow( polygon_img.permute(1,2,0).numpy() )

    # max label + polygons as an image
    return ( label, polygon_img )

def apply_polygon_mask_to_map(label_map: np.ndarray, polygon_mask: np.ndarray, label: int):
    """
    Label pixels matching a given polygon in the segmentation map.

    Args:
        label_map (np.ndarray): the map that stores the polygons
        polygon_mask (np.ndarray): a binary mask representing the polygon to be labeled.
        label (int): the label for this polygon; if the pixel already has a previous
                     label l', the resulting, compound value is (l'<<8)+label. Eg. 
                     applying label 4 on a pixel that already stores label 2 yields
                     2 << 8 + 4 = 0x204 = 8192
    """
    # for every pixel in intersection...
    intersection_boolean_mask = np.logical_and( label_map, polygon_mask )
    # ... shift it
    label_map[ intersection_boolean_mask ] <<= __LABEL_SIZE__

    # only then add label to all pixels matching the polygon
    label_map += polygon_mask.astype('int32') * label


def array_to_rgba_uint8( img_hw: np.ndarray ) -> torch.Tensor:
    """
    Converts a numpy array of 32-bit unsigned ints into a 4-channel tensor.

    Args:
        img_hw (np.ndarray): a flat label map.

    Output:
        torch.Tensor: a 4-channel tensor of unsigned 8-bit integers.
    """
    #img_chw = img_hw.view( torch.uint8 ).reshape( img_hw.shape + (4,) ).permute(2,0,1)
    img_chw = torch.from_numpy( np.moveaxis( img_hw.view(np.uint8).reshape( (img_hw.shape[0], -1, 4)), 2, 0))
    return img_chw


def rgba_uint8_to_hw_tensor( img_chw: torch.Tensor ) -> torch.Tensor:
    """
    Converts a 4-channel tensor of unsigned 8-bit integers into a numpy array of 32-bit ints.

    Args:
        img_chw (torch.Tensor): a 4-channel tensor of 8-bit unsigned integergs.

    Output:
        torch.Tensor: a flat map of 32-bit integers.
    """
    img_hw = img_chw.permute(1,2,0).reshape( img_chw.shape[1], -1 ).view(torch.int32)
    return img_hw


def map_to_depth(map_hw: torch.Tensor) -> torch.Tensor:
    """
    Compute depth of the pixels in the input map, i.e. how many polygons intersect on this pixel.
    Note: 0-valued pixels have depth 1.

    Args:
        map_hw (torch.Tensor): the input flat map (32-bit integers).

    Output:
        torch.Tensor: a tensor of integers, where each value represents the
                      number of intersecting polygons for the same position
                      in the input map.
    """
    layer_1 = map_hw & (map_hw < (1<<__LABEL_SIZE__))
    layer_2 = ((map_hw >= (1<<__LABEL_SIZE__)) & (map_hw < (1<<(__LABEL_SIZE__ * 2))))*2
    layer_3 = (map_hw >= (1<<(__LABEL_SIZE__ * 2)))*3

    depth_map = layer_1 + layer_2 + layer_3
    depth_map[ depth_map==0 ]=1

    return depth_map

def union_intersection_count_two_maps( map_hw_1: torch.Tensor, map_hw_2: torch.Tensor) -> torch.Tensor:
    """
    Provided two label maps that each encode (potentially overlapping) polygons, compute
    intersection and union counts for each possible pair of labels (i,j) with i ∈  map1
    and j ∈ map2.
    Shared pixels in each map (i.e. overlapping polygons) have their weight decreased according
    to the number of polygons they vote for. 

    Args:
        map_hw_1 (torch.Tensor): a flat map with labeled polygons, with potential overlaps.
        map_hw_2 (torch.Tensor): a flat map with labeled polygons, with potential overlaps.

    Output:
        torch.Tensor: a 2 channel tensor, where each cell [i,j] stores respectively intersection and union counts
                      for a pair of labels [i,j].
    """
    label_limit = 2**__LABEL_SIZE__-1
    max_label = torch.max( torch.max( map_hw_1[ map_hw_1<=label_limit ] ), torch.max( map_hw_2[ map_hw_2<=label_limit ]  )).item()
    # 2 channels for the intersection and union counts, respectively
    pixel_count_tensor_hwc = torch.zeros(( max_label, max_label, 2))

    for lbl1, lbl2 in itertools.product( range(1,max_label+1), range(1,max_label+1)):
        # Idea: the intersection pixel of a label 1 of depth m with label 2 of depth n has weight 1/max(m, n)
        # 1. Compute intersection boolean matrix, depth1 and depth2
        label_1_matrix, label_2_matrix = [ retrieve_polygon_mask_from_map( m, l ).type(torch.float) for (m,l) in ((map_hw_1, lbl1), (map_hw_2, lbl2)) ]
        intersection_mask = label_1_matrix * label_2_matrix
        depth_1, depth_2 = map_to_depth( map_hw_1 ), map_to_depth( map_hw_2 )
        # 2. For each pixel, keep the largest depth value of the two maps
        max_depth = torch.max( depth_1, depth_2 )
        # 3. Compute the weighted intersection count of the two maps
        intersection_count = torch.sum( intersection_mask / max_depth )
        # 4. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively
        label_1_count, label_2_count = torch.sum(label_1_matrix / depth_1), torch.sum(label_2_matrix / depth_2)

        # 5. uion = |label 1| + |label 2| - (label 1 ∩ label 2)
        union_count = label_1_count + label_2_count - intersection_count

        pixel_count_tensor_hwc[lbl1-1, lbl2-1]=torch.tensor([ intersection_count, union_count ])

    return pixel_count_tensor_hwc


def evaluate( cm: np.ndarray ) -> float:

    label_count = cm.shape[0]
    # correctly A-predictions over sum of all A predictions (=row sum)
    precision = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )
    # how much of all GT A-labels are correctly predicted (=col sum)
    recall = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )

    f_one = 2 * (precision+recall) / (precision*recall )

    return f_one[ f1>.9 ] / label_count


def test_map_value_for_label( px: int, label: int) -> bool:
    """
    Test whether a given label is encoded into a map pixel (for diagnosis purpose).

    Args:
        vl (int): a map pixel, whose value is a number in base __LABEL_SIZE__ (with digits being 
                  labels).
        label (int): the label to check for.

    Output:
        bool: True if the label is part of the code; False otherwise.
    """
    vl = px
    label_limit = 2**__LABEL_SIZE__-1
    while vl > label_limit:
        if (vl & label_limit) == label:
            return True
        vl >>= __LABEL_SIZE__
    return vl == label


def recover_labels_from_map_value( px: int) -> list:
    """
    Retrieves intersecting polygon labels from a single map pixel value (for
    diagnosis purpose).

    Args:
        vl (int): a map pixel, whose value is a number in base __LABEL_SIZE__ (with digits being 
                  labels).
    Output:
        list: a list of labels
    """
    vl = px
    labels = []
    label_limit = 2**__LABEL_SIZE__-1
    while vl > label_limit:
        labels.append( vl & label_limit )
        vl >>= __LABEL_SIZE__
    labels.append( vl )
    return labels[::-1]


def dummy():
    return True
