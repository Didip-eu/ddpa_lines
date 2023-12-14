from kraken import blla
from kraken.lib import vgsl
from PIL import Image
from pathlib import Path
import torchvision

#warning.warn(_BETA_TRANSFORMS_WARNING)
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
from torch import Tensor

import numpy as np
import cv2
import collections
from typing import Tuple
import itertools
import torch



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
    return dict_to_polygons( blla.segment( img, model=vgsl.TorchVGSLModel.load_model( model_path )), img )


# loading data and calling thisn one


def get_mask( img: Image.Image ) -> torch.Tensor:
    """
    Compute a binary mask from an image.

    Args:
        img (torch.Tensor): input image 
    """
    img = np.array( img )
    if img.shape[2]>1:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

    return torch.from_numpy( cv2.threshold( img, 0, 255, cv2.THRESH_OTSU )[1] / 255 )



def get_confusion_matrix_from_arrays( polygon_img_gt: torch.Tensor, polygon_img_pred: torch.Tensor, mask: torch.Tensor, label_counts: Tuple[int,int] ) -> torch.Tensor:
    """
    Compute a confusion matrix for segmentation output.

    Args:
        img (np.ndarray): input image (4-channel, unsigned 8-bit integers).
        segmentation_dict_gt (np.ndarray): the GT dictionary describing the lines (see above)
        segmentation_dict_pred (np.ndarray): the inferred dictionary describing the lines, as given by the Kraken library call

    Output:
        np.ndarray: a matrix containing I/U scores for all pairs (GT polygon pi, predicted polygon pj)

    TODO
    """

    # 4-channel image to flat 32-bit image
    print('polygon_img_gt.shape = {}, polygon_img_pred.shape = {}'.format( polygon_img_gt.shape, polygon_img_pred.shape ))
    polygon_img_gt=rgba_uint8_to_int32( polygon_img_gt.numpy() )
    polygon_img_pred=rgba_uint8_to_int32( polygon_img_pred.numpy() )
    print('Collating bytes: polygon_img_gt.shape = {}, polygon_img_pred.shape = {}'.format( polygon_img_gt.shape, polygon_img_pred.shape ))

    # introducing tensor type here because it is hashable and can be passed to Counter,
    # but ultimately the rest of the code should use tensors too
    polygon_img_gt = torch.from_numpy( polygon_img_gt )
    polygon_img_pred = torch.from_numpy( polygon_img_pred )

    # How many pixels in each polygon?
    # Technical debt here: Counter cannot work on Numpy arrays, hence the messy writing
    count_gt = collections.Counter( polygon_img_gt * mask )
    count_pred = collections.Counter( polygon_img_pred * mask )

    lbl_max = max( label_counts )
    intersection_matrix = collections.Counter( list((polygon_img_gt*mask + (lbl_max+1)*polygon_img_pred*mask).flatten()) )

    confusion_matrix = torch.zeros((lbl_max, lbl_max))

    for lbl_gt, lbl_pred in itertools.product( range(lbl_max+1), range(lbl_max+1 )):
        intersection = intersection_matrix[ polygon_img_gt*mask + (lbl_max+1)*polygon_img_pred*mask ]
        union = count_gt[ lbl_gt ] + count_pred[ lbl_pred ] - intersection
        confusion_matrix[lbl_gt-1, lbl_pred-1] = intersection/union if union else 0

    return confusion_matrix


def evaluate( cm: np.ndarray ) -> float:

    label_count = cm.shape[0]
    # correctly A-predictions over sum of all A predictions (=row sum)
    precision = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )
    # how much of all GT A-labels are correctly predicted (=col sum)
    recall = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )

    f_one = 2 * (precision+recall) / (precision*recall )

    return f_one[ f1>.9 ] / label_count



def dummy():
    return True

def dict_to_polygons( segmentation_dict: dict, img: Image.Image ) -> Tuple[int, torch.Tensor]:
    """
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

            TODO: add shape into variable names and meaning of points

            let's settle on input/ouput as CHW

    """
    polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

    # create 2D matrix of 32-bit integers 
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    polygon_img = np.zeros( tuple( reversed(img.size)), dtype=np.int32 )

    # rendering polygons ( RGBA - 4 bytes - integer tensor out of it - pack label on the alpha channel - remembering that opencv is GBR)
    for lbl, polyg in enumerate( polygon_boundaries ):
        # Notes: 
        # - fillPoly expects polygs as array of arrays as its second parameter
        cv2.fillPoly( polygon_img, np.array( [polyg] ), lbl+1)

    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( polygon_img )

    # max label + polygons as an image
    return (len(polygon_boundaries)+1, torch.from_numpy( polygon_img ))


def array_to_rgba_uint8( arr: np.ndarray ) -> np.ndarray:
    return arr.astype( np.uint32 ).view( np.uint8 ).reshape( arr.shape + (4,) )


def rgba_uint8_to_int32( arr: np.ndarray ) -> np.ndarray:
    return arr.view( np.int32 ).reshape( arr.shape[:-1] )
