
from kraken import blla
from kraken.lib import vgsl
from PIL import Image, ImageDraw
import skimage as ski
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import Tensor

import numpy as np
import numpy.ma as ma
from typing import Tuple, Callable
import itertools


__LABEL_SIZE__=8

"""
Functions for segmentation output management

+ storing polygons on tensors (from dictionaries or pageXML outputs)
+ computing IoU and F1 scores over GT/predicted label maps

A note about types:

+ PageXML or JSON: initial input (typically: from segmentation framework) 
+ torch.Tensor: map storage and computations (eg. counting intersections)
+ np.ndarray: metrics and scores
"""


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
        segmentation_dict (dict): kraken's segmentation output, i.e. a dictionary of the form::

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



def xml_to_polygon_map( page_xml: str, img: str ) -> Tensor:
    """
    Read line polygons from a PageXML file and store them into a tensor, as pixel maps.
    Channels allow for easy storage of overlapping polygons.

    Args:
        page_xml (str): path of a PageXML file. 

        img (str): the input image

    Output:
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D matrix of 32-bit integers
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    label_map = np.zeros( img.size[::-1], dtype='int32' )

    with Image.open( img, 'r'):
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



def pagexml_to_segmentation_dict(page: str) -> dict:
    """
    Given a pageXML file, return a JSON dictionary describing the lines.

    Args:
        page (str): path of a PageXML file
    Output:
        dict: a dictionary of the form

        {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}

    """
    direction = {'0.0': 'horizontal-lr', '0.1': 'horizontal-rl', '1.0': 'vertical-td', '1.1': 'vertical-bu'}

    with open( page, 'r' ) as page_file:
        page_tree = ET.parse( page_file )
        ns = { 'pc': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        page_root = page_tree.getroot()

        page_dict = { 'type': 'baselines' }

        page_dict['text_direction'] = direction[ page_root.find('.//pc:TextRegion', ns).get( 'orientation' )]

        lines_object = []
        for line in page_root.findall('.//pc:TextLine', ns):
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                continue
            baseline_points = [ [ int(p) for p in pt.split(',') ] for pt in baseline_elt.get('points').split(' ') ]

            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                continue
            polygon_points = [ [ int(p) for p in pt.split(',') ] for pt in coord_elt.get('points').split(' ') ]

            lines_object.append( {'line_id': line_id, 'baseline': baseline_points, 'boundary': polygon_points} )

        page_dict['lines'] = lines_object

    return page_dict 


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
    max_two_polygon_label=0xffff
    # for every pixel in intersection...
    intersection_boolean_mask = np.logical_and( label_map, polygon_mask )
    # if intersection does not already contain a 3-polygon pixel
    if np.any( label_map[ intersection_boolean_mask ] > max_two_polygon_label ):
        maxed_out_pixels = np.transpose(((label_map * intersection_boolean_mask) > max_two_polygon_label).nonzero())
        print("Some pixels already store 3 polygons:", [ (row,col) for (row,col) in maxed_out_pixels ][:5], ' ...' if len(maxed_out_pixels)>5 else '')
        raise ValueError('Cannot store more than 3 polygons on the same pixel!')
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
        Tensor: a 4-channel (c,h,w) tensor of unsigned 8-bit integers.
    """
    #img_chw = img_hw.view( torch.uint8 ).reshape( img_hw.shape + (4,) ).permute(2,0,1)
    img_chw = torch.from_numpy( np.moveaxis( img_hw.view(np.uint8).reshape( (img_hw.shape[0], -1, 4)), 2, 0))
    return img_chw


def rgba_uint8_to_hw_tensor( img_chw: Tensor ) -> Tensor:
    """
    Converts a 4-channel tensor of unsigned 8-bit integers into a numpy array of 32-bit ints.

    Args:
        img_chw (Tensor): a 4-channel tensor of 8-bit unsigned integers.

    Output:
        Tensor: a flat map of 32-bit integers.
    """
    img_hw = img_chw.permute(1,2,0).reshape( img_chw.shape[1], -1 ).view(torch.int32)
    return img_hw



def polygon_pixel_metrics_from_img_json(img: Image.Image, segmentation_dict_pred: dict, segmentation_dict_gt: dict, binary_mask: Tensor=None) -> np.ndarray:
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

    return polygon_pixel_metrics_from_polygon_maps_and_mask( polygon_chw_gt, polygon_chw_pred, binary_mask )


def get_mask( img_whc: Image.Image, thresholding_alg: Callable=ski.filters.threshold_otsu ) -> Tensor:
    """
    Compute a binary mask from an image, using the given thresholding algorithm: FG=1s, BG=0s

    Args:
        img (PIL image): input image
    Output:
        Tensor: a binary map with FG pixels=1 and BG=0.
    """
    img_hwc= np.array( img_whc )
    threshold = thresholding_alg( ski.color.rgb2gray( img_hwc ) if img_hwc.shape[2]>1 else img_hwc )*255
    img_bin_hw = torch.tensor( (img_hwc < threshold)[:,:,0], dtype=torch.bool )

    return img_bin_hw


def polygon_pixel_metrics_from_polygon_maps_and_mask(polygon_chw_pred: Tensor, polygon_chw_gt: Tensor, binary_hw_mask: Tensor=None, label_distance=0) -> np.ndarray:
    """
    Compute pixel-based metrics from two tensors that each encode (potentially overlapping) polygons
    and a FG mask.

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

    polygon_chw_fg_gt, polygon_chw_fg_pred = [ polygon_img * binary_hw_mask for polygon_img in (polygon_chw_gt, polygon_chw_pred) ]
    
    metrics = polygon_pixel_metrics_two_maps( polygon_chw_fg_gt, polygon_chw_fg_pred, label_distance )

    return metrics

def retrieve_polygon_mask_from_map( label_map_chw: Tensor, label: int, binary_mask_hw: Tensor=None) -> Tensor:
    """
    From a label map (that may have compound pixels representing polygon intersections),
    compute a binary mask that covers _all_ pixels for the label, whether they belong to an
    intersection or not.

    Args:
        label_mask_hw (Tensor): a flat map with labeled polygons, with potential overlaps.
        label (int): the label to be selected.

    Output:
        Tensor: a flat, boolean mask for the polygon of choice.
    """
    if len(label_map_chw.shape) != 3 and label_map_chw.shape[0] != 4:
        raise TypeError("Wrong type: label map should be a 4-channel tensor (shape={} instead).".format( label_map_chw.shape ))
    polygon_mask_hw = torch.sum( label_map_chw==label, axis=0)

    return polygon_mask_hw

def polygon_pixel_metrics_two_maps( map_chw_1: Tensor, map_chw_2: Tensor, label_distance=5) -> np.ndarray:
    """
    Provided two label maps that each encode (potentially overlapping) polygons, compute
    for each possible pair of labels (i,j) with i ∈  map1 and j ∈ map2.
    + intersection and union counts
    + precision and recall

    Shared pixels in each map (i.e. overlapping polygons) have their weight decreased according
    to the number of polygons they vote for. 

    Args:
        map_chw_1 (Tensor): a 4-channel map with labeled polygons, with potential overlaps.
        map_hw_2 (Tensor): a 4-channel map with labeled polygons, with potential overlaps.

    Output:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union counts,
                    as well as precision and recall for a pair of labels [i,j].
    """
    min_label_1, max_label_1 = torch.min( map_chw_1[ map_chw_1 > 0 ] ).item(), torch.max( map_chw_1 ).item()
    min_label_2, max_label_2 = torch.min( map_chw_2[ map_chw_2 > 0 ] ).item(), torch.max( map_chw_2 ).item()
    label2index_1 = { l:i for i,l in enumerate( range(min_label_1, max_label_1+1)) }
    label2index_2 = { l:i for i,l in enumerate( range(min_label_2, max_label_2+1)) }

    # 4 channels for the intersection and union counts, and the precision and recall scores, respectively
    metrics_hwc = np.zeros(( max_label_1-min_label_1+1, max_label_2-min_label_2+1, 4), dtype='float32')

    label_range_1, label_range_2 = range(min_label_1, max_label_1+1), range(min_label_2, max_label_2+1)
    
    # Factor out some computations
    label_matrices_1={ l:retrieve_polygon_mask_from_map( map_chw_1, l) for l in label_range_1 }
    label_matrices_2={ l:retrieve_polygon_mask_from_map( map_chw_2, l) for l in label_range_2 }
    depth_1, depth_2 = map_to_depth( map_chw_1 ), map_to_depth( map_chw_2 )
    max_depth = torch.max( depth_1, depth_2 ) # for each pixel, keep the largest depth value of the two maps
    label_counts_1={ l:torch.sum(label_matrices_1[l]/depth_1).item() for l in label_range_1 }
    label_counts_2={ l:torch.sum(label_matrices_2[l]/depth_2).item() for l in label_range_2 }

    for lbl1, lbl2 in itertools.product(label_range_1, label_range_2):
        # assume that labels beyond a given distance do not intersect
        if label_distance > 0 and abs(lbl1-lbl2) > label_distance:
            metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ 0, label_counts_1[lbl1] + label_counts_2[lbl2], 0, 0 ]
            continue
        # Idea: 
        # + the intersection of a label 1 of depth m (where m = # of polygons that intersect on the pixel) with label 2
        # of depth n has weight 1/max(m, n)
        # + the union of a label-1 pixel of depth m with label-2 pixel of depth n has weight (1/m + 1/n)

        # 1. Compute intersection boolean matrix
        #print("1. Compute intersection boolean matrix, depth1 and depth2")
        label_1_matrix, label_2_matrix = label_matrices_1[ lbl1 ], label_matrices_2[ lbl2 ]
        intersection_mask = label_1_matrix * label_2_matrix

        # 2. Compute the weighted intersection count of the two maps
        #print("3. For each pixel, keep the largest depth value of the two maps")
        intersection_count = torch.sum( intersection_mask / max_depth ).item()

        # 3. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively
        #print('4. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively')
        label_1_count, label_2_count = label_counts_1[lbl1], label_counts_2[lbl2]

        # 4. union = |label 1| + |label 2| - |label 1 ∩ label 2|
        #print('5. union = |label 1| + |label 2| - |label 1 ∩ label 2|')
        union_count = label_1_count + label_2_count - intersection_count

        # 5. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |
        #print('6. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |')
        if label_1_count != 0:
            # rows (label_1) assumed to be predictions
            precision = intersection_count / label_1_count
        if label_2_count != 0:
            # cols (label_2) assumed to be GT
            recall = intersection_count / label_2_count

        metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ intersection_count, union_count, precision, recall ]

    return metrics_hwc


def map_to_depth(map_chw: Tensor) -> Tensor:
    """
    Compute depth of each pixel in the input map, i.e. how many polygons intersect on this pixel.
    Note: 0-valued pixels have depth 1.

    Args:
        map_chw (Tensor): the input tensor (4 channels)

    Output:
        Tensor: a tensor of integers, where each value represents the
                number of intersecting polygons for the same position
                in the input map.
    """
    depth_map = torch.sum( map_chw != 0, axis=0)
    depth_map[ depth_map == 0 ]=1

    return depth_map


def polygon_pixel_metrics_to_line_based_scores( metrics: np.ndarray, threshold: float=.5 ) -> Tuple[float, float, float]:
    """
    Implement ICDAR 2017 evaluation metrics, as described in
    https://github.com/DIVA-DIA/DIVA_Line_Segmentation_Evaluator/releases/tag/v1.0.0
    (a Java implementation)

    IoU = TP / (TP+FP+FN)
    F1 = (2*TP) / (2*TP+FP+FN)

    + find all polygon pairs that have a non-empty intersection
    + sort the pairs by IoU
    + traverse the IoU-descending sorted list and select the first available match 
      for each polygon belonging to the prediction set (thus ensuring that no
      polygon can be matched twice)
    + a pred/GT match is TP if both Recall and Precision > .75.
    + a pred/GT match is FP if P < .75; FN if R < .75 (it can be both!)
    + Line-based, per-page IoU [or Jaccard index]= TP/(TP+FP+FN)
    + Document-wide, pixel-based IoU_px  = TP_px/TP_px + FP_px + FN_px}.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for labels 1..m, where m is the maximum label in either
                            GT or predicted maps. In channels: intersection count, union count, precision, recall.
    Out:
        tuple: a 5-tuple with the TP-, FP-, and FN-counts, as well as the Jaccard (aka. IoU) and F1 score at the line level.
    """
    label_count_pred, label_count_gt = metrics.shape[:2] 

    # find all rows with non-empty intersection
    possible_match_indices = metrics[:,:,0].nonzero()
    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([ 
        (match_rows_cols[i,0],
         match_rows_cols[i,1], 
         possible_matches[i,0], 
         possible_matches[i,1], 
         possible_matches[i,2], 
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))], 
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'), 
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'), 
                ('iou', 'float32')])

    TP, FP, FN = 0.0, 0.0, 0.0
    # select one-to-one matches
    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in np.sort( structured_row_col_match_iou, order=['pred_polygon', 'gt_polygon', 'iou'] ):
        # ensure that each predicted label is matched to at most one GT label
        if not pred2match[possible_match['pred_polygon']]: 
            pred2match[possible_match['pred_polygon']]=True
            precision, recall = possible_match[['precision', 'recall']]
            TP += (precision >= threshold and recall >= threshold )
            # a FP is a non-zero (Pred, GT) pair whose P < .75 or: the system detects
            # a polygon that partially capture the GT, but too much of the rest also
            FP += precision < threshold
            # a FN is a non-zero (Pred, GT) pair whose R < .75 or: the system detects
            # a polygon that matches the GT, but not enough of it
            FN += recall < threshold
    
    Jaccard = TP / (TP+FP+FN)
    F1 = 2*TP / (2*TP+FP+FN)

    return (TP, FP, FN, Jaccard, F1)



def polygon_pixel_metrics_to_pixel_based_scores( metrics: np.ndarray, threshold: float=.5 ) -> Tuple[float, float, float]:
    """
    et mplement ICDAR 2017 pixel-based evaluation metrics, as described in
    Simistira et al., ICDAR2017 Competition on Layout Analysis for Challenging Medieval
    Manuscripts, 2017.

    Two versions of the pixel-based IoU metric:
    + Pixel IU takes all pixels of all intersecting pairs into account
    + Matched Pixel IU only takes into account the pixels from the matched lines 

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for labels 1..m, where m is the maximum label in either
                            GT or predicted maps. In channels: intersection count, union count, precision, recall.
    Out:
        tuple: a pair (Pixel IU, Matched Pixel IU)
    """
    label_count_pred, label_count_gt = metrics.shape[:2] 

    # find all rows with non-empty intersection
    possible_match_indices = metrics[:,:,0].nonzero()
    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([ 
        (match_rows_cols[i,0],
         match_rows_cols[i,1], 
         possible_matches[i,0], 
         possible_matches[i,1], 
         possible_matches[i,2], 
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))], 
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'), 
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'), 
                ('iou', 'float32')])

    # pixel-based, page-wide IoU (over all non-empty intersections)
    intersection_count, union_count = [ np.sum(structured_row_col_match_iou[:][field]) for field in ('intersection', 'union') ]
    pixel_iou = intersection_count / union_count

    # pixel-based, page-wide IoU (over all matched pairs)
    matched_intersection_count, matched_union_count = 0, 0
    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in np.sort( structured_row_col_match_iou, order=['pred_polygon', 'gt_polygon', 'iou'] ):
        # ensure that each predicted label is matched to at most one GT label
        if not pred2match[possible_match['pred_polygon']]: 
            pred2match[possible_match['pred_polygon']]=True
            matched_intersection_count += possible_match['intersection']
            matched_union_count += possible_match['union']
    matched_pixel_iou = matched_intersection_count / matched_union_count
    
    return (pixel_iou, matched_pixel_iou)

    
def metrics_to_precision_recall_curve( metrics: np.ndarray, threshold_range=np.linspace(0, 1, num=21)) -> np.ndarray:
    """
    Compute precision and recalls over a range of IoU thresholds, for plotting purpuse.

    Args:
        metrics (np.ndarray): a 4-channel table with GT labels in rows and predicted labels in columns, where
                              each entry is a [intersection_count, union_count, precision, recall] sequence.
        threshold_range: a series of threshold values, between 0 and 1 (default: [0, 0.05, 0.1, ..., 0.95, 1])

    Output:
        np.ndarray: a 2D array, with precisions in row 0 and recalls in row 1.

    """
    precisions_recalls = np.zeros((len(threshold_range), 2))
    for (i,t) in enumerate(threshold_range):
        precisions_recalls[i] = metrics_to_aggregate_scores(metrics, iou_threshold=t)[:2]
        #print(precisions_recalls[:,i])
    return np.moveaxis( precisions_recalls, 1, 0)


def test_map_value_for_label( px: int, label: int) -> bool:
    """
    Test whether a given label is encoded into a map pixel (for diagnosis purpose only:
    applying this to large array is prohibitively slow).

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
    diagnosis purpose: prohibitively slow on large arrays).

    Args:
        vl (int): a map pixel, whose value is a number in base __LABEL_SIZE__ (with digits being 
                  labels).
    Output:
        list: a list of labels
    """
    vl = px
    labels = []
    label_limit = 2**__LABEL_SIZE__-1
    compound_label_limit = 2**(__LABEL_SIZE__*3)-1
    if px > compound_label_limit:
        return []
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
