
from kraken import blla
from kraken.lib import vgsl
from PIL import Image, ImageDraw
import skimage as ski
from pathlib import Path

import torch
from torch import Tensor

import numpy as np
import numpy.ma as ma
from typing import Tuple, Callable
import itertools


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


def dict_to_polygon_map( segmentation_dict: dict, img: Image.Image ) -> Tensor:
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
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D matrix of 32-bit integers
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    label_map = np.zeros( img.size[::-1], dtype='int32' )

    # rendering polygons
    for lbl, polyg in enumerate( polygon_boundaries ):
        polyg_mask = ski.draw.polygon2mask( img.size[::-1], polyg )
        apply_polygon_mask_to_map( label_map, polyg_mask, lbl+1 )

    #ski.io.imshow( polygon_img.transpose() )

    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( label_map )
    #ski.io.imshow( polygon_img.permute(1,2,0).numpy() )

    # max label + polygons as an image
    return polygon_img

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


def array_to_rgba_uint8( img_hw: np.ndarray ) -> Tensor:
    """
    Converts a numpy array of 32-bit unsigned ints into a 4-channel tensor.

    Args:
        img_hw (np.ndarray): a flat label map.

    Output:
        Tensor: a 4-channel tensor of unsigned 8-bit integers.
    """
    #img_chw = img_hw.view( torch.uint8 ).reshape( img_hw.shape + (4,) ).permute(2,0,1)
    img_chw = torch.from_numpy( np.moveaxis( img_hw.view(np.uint8).reshape( (img_hw.shape[0], -1, 4)), 2, 0))
    return img_chw


def rgba_uint8_to_hw_tensor( img_chw: Tensor ) -> Tensor:
    """
    Converts a 4-channel tensor of unsigned 8-bit integers into a numpy array of 32-bit ints.

    Args:
        img_chw (Tensor): a 4-channel tensor of 8-bit unsigned integergs.

    Output:
        Tensor: a flat map of 32-bit integers.
    """
    img_hw = img_chw.permute(1,2,0).reshape( img_chw.shape[1], -1 ).view(torch.int32)
    return img_hw



def get_metrics_from_img_json(img: Image.Image, segmentation_dict_gt: dict, segmentation_dict_pred: dict, binary_mask: Tensor=None) -> np.ndarray:
    """
    Compute a IoU matrix from an image and two dictionaries describing the segmentation's output (line polygons).

    Args:
        img (Image.Image): the input page, needed for the size information and the binarization mask.
        segmentation_dict_pred (dict): a dictionary, typically constructed from a JSON file.
        segmentation_dict_gt (dict): a dictionary, typically constructed from a JSON file.
    Output:
        np.ndarray: a 2D array, representing IoU values for each possible pair of polygons.
    """
    #polygon_img_gt: Tensor, polygon_img_pred: Tensor, mask: Tensor) -> Tensor:
    polygon_chw_gt, polygon_chw_pred = [ dict_to_polygon_map( d,img ) for d in (segmentation_dict_gt, segmentation_dict_pred) ]

    binary_mask = get_mask( img )

    return get_metrics_from_polygon_maps( polygon_chw_gt, polygon_chw_pred, binary_mask )


def get_metrics_from_polygon_maps(polygon_chw_gt: Tensor, polygon_chw_pred: Tensor, binary_hw_mask: Tensor=None) -> np.ndarray:
    """
    Compute a IoU matrix from two tensors that each encode (potentially overlapping) polygons
    and a binarized map.

    Args:
        polygon_chw_gt (Tensor): a 4-channel image, where each position may store up to 4 overlapping labels
                                 (one for each channel)
        polygon_chw_pred (Tensor): a 4-channel image, where each position may store up to 4 overlapping labels
                                   (one for each channel)
        binary_hw_mask (Tensor): a boolean mask that selects the input image's FG pixel

    Output:
        np.ndarray: metrics (intersection, union, precision, recall, f1) values for each possible pair of labels
                    (i,j) with i ∈  map1 and j ∈ map2. Shared pixels in each map (i.e. overlapping polygons) have
                    their weight decreased according to the number of polygons they vote for.
    """

    if binary_hw_mask is None:
        binary_hw_mask = torch.full( polygon_chw_gt.shape[1:], 1, dtype=torch.bool )
    if binary_hw_mask.shape != polygon_chw_gt.shape[1:]:
        raise TypeError("Wrong type: binary mask should have shape {}".format(polygon_chw_gt.shape[1:]))

    if len(polygon_chw_gt.shape) != 3 or polygon_chw_gt.shape[0] != 4 or polygon_chw_gt.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon GT map should be a 4-channel tensor of unsigned 8-bit integers.")
    if len(polygon_chw_pred.shape) != 3 or polygon_chw_pred.shape[0] != 4 or polygon_chw_pred.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon predicted map should be a 4-channel tensor of unsigned 8-bit integers.")
    if polygon_chw_gt.shape != polygon_chw_pred.shape:
        raise TypeError("Wrong type: both maps should have the same shape (got {} and {}).".format( polygon_chw_gt.shape, polygon_chw_pred.shape ))

    # convert to flat 32-bit image
    polygon_chw_gt, polygon_chw_pred = [ rgba_uint8_to_hw_tensor( polygon_img ) * binary_hw_mask for polygon_img in (polygon_chw_gt, polygon_chw_pred) ]
    # first 2 columns of the metrics matrix
    metrics = get_metrics_two_maps( polygon_chw_gt, polygon_chw_pred )


    #iou_matrix = np.ma.MaskedArray( pixel_counts[:,:,0]/pixel_counts[:,:,1], fill_value=0.0 )
    #return np.array(np.ma.fix_invalid(iou_matrix))

    return metrics

def get_mask( img_whc: Image.Image, thresholding_alg: Callable=ski.filters.threshold_otsu ) -> Tensor:
    """
    Compute a binary mask from an image, using the given thresholding algorithm: FG=1s, BG=0s

    Args:
        img (PIL image): input image
    Output:
        Tensor: a binary map with FG pixels=1 and BG=0.
    """
    img_hwc= np.array( img_whc )
    threshold = thresholding_alg( ski.color.rgb2gray( img_hwc) if img_hwc.shape[2]>1 else img_hwc )*255
    img_bin_hw = torch.tensor( (img_hwc < threshold)[:,:,0], dtype=torch.bool )

    return img_bin_hw

def retrieve_polygon_mask_from_map( label_map_hw: Tensor, label: int, binary_mask_hw: Tensor=None) -> Tensor:
    """
    From a label map (that may have compound pixels representing polygon intersections),
    compute a binary mask that covers _all_ pixels for the label, whether they belong to an
    intersection or not.

    Args:
        label_mask_hw (Tensor): a flat map with labeled polygons, with potential overlaps.
        label (int): the label to be selected.

    Output:
        Tensor: a flat, single-label map for the polygon of choice.
    """
    if len(label_map_hw.shape) > 2:
        raise TypeError("Wrong type: label map should be a flat map (shape={} instead).".format( label_map_hw.shape ))
    label_map_np = label_map_hw.numpy()
    label_bitmask = 2**__LABEL_SIZE__-1
    polygon_mask_hw = np.zeros( label_map_np.shape, dtype='bool')
    # maximum number of intersecting labels for a single pixel
    # (determine how many times we r-shift the values)
    DEPTH = 3
    for d in range(DEPTH):
        polygon_mask_hw += ((label_map_np >> (__LABEL_SIZE__ * d)) & label_bitmask) == label

    return torch.tensor( polygon_mask_hw )

def get_metrics_two_maps( map_hw_1: Tensor, map_hw_2: Tensor) -> np.ndarray:
    """
    Provided two label maps that each encode (potentially overlapping) polygons, compute
    for each possible pair of labels (i,j) with i ∈  map1 and j ∈ map2.
    + intersection and union counts
    + precision and recall

    Shared pixels in each map (i.e. overlapping polygons) have their weight decreased according
    to the number of polygons they vote for. 

    Args:
        map_hw_1 (Tensor): a flat map with labeled polygons, with potential overlaps.
        map_hw_2 (Tensor): a flat map with labeled polygons, with potential overlaps.

    Output:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union counts,
                    as well as precision and recall for a pair of labels [i,j].
    """
    label_limit = 2**__LABEL_SIZE__-1
    min_label = torch.min( torch.min( map_hw_1[ map_hw_1 > 0 ] ), torch.max( map_hw_2[ map_hw_2 > 0 ]  )).item()
    max_label = torch.max( torch.max( map_hw_1[ map_hw_1<=label_limit ] ), torch.max( map_hw_2[ map_hw_2<=label_limit ]  )).item()
    label2index = { l:i for i,l in enumerate( range(min_label, max_label+1)) }
    
    # 4 channels for the intersection and union counts, and the precision and recall scores, respectively
    metrics_hwc = np.zeros(( max_label-min_label+1, max_label-min_label+1, 4))

    for lbl1, lbl2 in itertools.product( range(min_label, max_label+1), range(min_label, max_label+1)):
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
        # 5. uion = |label 1| + |label 2| - |label 1 ∩ label 2|
        union_count = label_1_count + label_2_count - intersection_count
        # 6. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |
        if label_1_count != 0 and label_2_count != 0:
            precision = intersection_count / label_2_count
            recall = intersection_count / label_1_count

        metrics_hwc[label2index[lbl1], label2index[lbl2]]=[ intersection_count, union_count, precision, recall ]

    return metrics_hwc


def map_to_depth(map_hw: Tensor) -> Tensor:
    """
    Compute depth of the pixels in the input map, i.e. how many polygons intersect on this pixel.
    Note: 0-valued pixels have depth 1.

    Args:
        map_hw (Tensor): the input flat map (32-bit integers).

    Output:
        Tensor: a tensor of integers, where each value represents the
                number of intersecting polygons for the same position
                in the input map.
    """
    layer_1 = map_hw & (map_hw < (1<<__LABEL_SIZE__))
    layer_2 = ((map_hw >= (1<<__LABEL_SIZE__)) & (map_hw < (1<<(__LABEL_SIZE__ * 2))))*2
    layer_3 = (map_hw >= (1<<(__LABEL_SIZE__ * 2)))*3

    depth_map = layer_1 + layer_2 + layer_3
    depth_map[ depth_map==0 ]=1

    return depth_map

def metrics_to_aggregate_scores( metrics: np.ndarray, gt_label_count:int, pred_label_count: int, iou_threshold: float=.5 ) -> Tuple[float, float, float]:
    """
    Compute F1 score over the IoU matrix, taking into account the one-to-one matching pairs that meet the IoU threshold.
    TODO: deal with case where 1 polygon is matched to more than one.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for labels 1..m, where m is the maximum label in either
                            GT or predicted maps. In channels: intersection count, union count, precision, recall.
    Out:
        tuple: a triplet with the precision, recall and F1 score for the matching pairs.
    """
    label_count = metrics.shape[0]

    # one-to-one matches
    iou_matrix = metrics[:,:,0]/metrics[:,:,1]
    o2o = iou_matrix > iou_threshold
    precision = np.sum(o2o)/gt_label_count
    recall = np.sum(o2o)/pred_label_count

    f_one = 2 * (precision*recall) / (precision+recall ) if precision+recall else 0

    return (precision, recall, f_one)

def metrics_to_f1_matrix( metrics: np.ndarray, gt_label_count:int, pred_label_count: int, iou_threshold: float = .5) -> np.ndarray:
    """
    Compute F1 score over the IoU matrix.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for
                        labels 1..m, where m is the maximum label in either
                        GT or predicted maps.
    Out:
        np.ndarray: a 2D array containing for each label the precision (row 0),
                    the recall (row 1), and the F1 score (row 3).

    """
    label_count = metrics.shape[0]
    precisions, recalls = metrics[:,:,2], metrics[:,:,3]

    precision_plus_recall_masked = np.ma.MaskedArray(precisions+recalls, mask=((precisions==0) | (recalls==0)), fill_value=0.0)

    f_ones = np.array( np.ma.fix_invalid( 2 * (precisions*recalls) / (precision_plus_recall_masked )))

    return f_ones



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
    """
    Just to check that the module is testable.
    """
    return True