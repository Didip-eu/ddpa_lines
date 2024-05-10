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
import torch

import skimage as ski
from matplotlib import pyplot as plt



RED = (255,0,0)
GREEN = (0,255,0)
YELLOW = (0,255,255)
BLUE = (0,0,255)

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
    img_bin_hw = torch.from_numpy( (img_hwc < threshold)[:,:,0] ) 

    return img_bin_hw.int()


def get_confusion_matrix_from_arrays( polygon_img_gt: torch.Tensor, polygon_img_pred: torch.Tensor, mask: torch.Tensor, label_counts: Tuple[int,int] ) -> torch.Tensor:
    """
    Compute a confusion matrix for segmentation output.

    Args:
        polygon_img_gt (torch.Tensor): ground truth polygon map (4-channel, unsigned 8-bit integers).
        polygon_img_pred (torch.Tensor): predicted polygon map (4-channel, unsigned 8-bit integers).
        mask (torch.Tensor): binary mask representing the foreground pixels
        label_counts (tuple): number of labels in each map

    Output:
        torch.Tensor: a matrix containing I/U scores for all pairs (GT polygon pi, predicted polygon pj)

    """

    # 4-channel image to flat 32-bit image
    print('polygon_img_gt.shape = {}, polygon_img_pred.shape = {}'.format( polygon_img_gt.shape, polygon_img_pred.shape ))
    polygon_img_gt=rgba_uint8_to_int32( polygon_img_gt )
    polygon_gt_fg = polygon_img_gt * mask
    polygon_img_pred=rgba_uint8_to_int32( polygon_img_pred )
    polygon_pred_fg = polygon_img_pred * mask

    print('Collating bytes: polygon_img_gt.shape = {}, polygon_img_pred.shape = {}'.format( polygon_img_gt.shape, polygon_img_pred.shape ))

    # How many pixels in each polygon?
    count_gt = collections.Counter( polygon_gt_fg.flatten().tolist() )
    count_pred = collections.Counter(polygon_pred_fg.flatten().tolist() )

    lbl_max = max( label_counts )

    # For each position w/ to (possibly different) labels m and n,
    # compute a unique integer ( m + lmax*n)
    # Eg. 
    #    l   l_hat
    #    4 + 4*7 →  32
    #    5 + 4*7 →  33
    intersection_matrix = polygon_gt_fg + (lbl_max+1)*polygon_pred_fg
    intersection_counts = collections.Counter( intersection_matrix.flatten().tolist())

    confusion_matrix = torch.zeros((lbl_max, lbl_max))

    for lbl_gt, lbl_pred in itertools.product( range(1,lbl_max+1), range(1,lbl_max+1)):
        intersect_count_key = lbl_gt + (lbl_max+1)*lbl_pred
        intersect_count = intersection_counts[ intersect_count_key ] if intersect_count_key else 0
        print( lbl_gt, '∩', lbl_pred, "=", intersect_count)
        union_count = count_gt[ lbl_gt ] + count_pred[ lbl_pred ] - intersect_count
        confusion_matrix[lbl_gt-1, lbl_pred-1] = intersect_count/union_count if union_count else 0

    return confusion_matrix


def evaluate( cm: np.ndarray ) -> float:

    label_count = cm.shape[0]
    # correctly A-predictions over sum of all A predictions (=row sum)
    precision = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )
    # how much of all GT A-labels are correctly predicted (=col sum)
    recall = np.array( [ cm[l,l] / sum(cm[l,:]) for a in range( label_count ) ] )

    f_one = 2 * (precision+recall) / (precision*recall )

    return f_one[ f1>.9 ] / label_count


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
    polygon_img = np.zeros( tuple( reversed(img.size)), dtype=np.int32 )

    # rendering polygons (RGBA)
    label = 0
    for lbl, polyg in enumerate( polygon_boundaries ):
        #cv2.fillPoly( polygon_img, np.array( [polyg] ), lbl+1)
        # Notes:
        # - with this method, a polygon intersection has its own
        #   (possibly non-unique) label
        label = lbl + 1
        polygon_img += ( ski.draw.polygon2mask( img.size[::-1], polyg ) * label )

        # Another choice is to override an existing label with the new one:
        # - intersections between polygons are lost
        #polygon_img = ( ski.draw.polygon2mask( img.size[::-1], polyg ) * label )

    print("Max polygon = ", label )
    print("Max label = ", polygon_img.max())
     
    print("polygon_boundaries.length =", len(polygon_boundaries))
    #ski.io.imshow( polygon_img.transpose() )
    #plt.show()


    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( torch.from_numpy( polygon_img ))
    #ski.io.imshow( polygon_img.permute(1,2,0).numpy() )
    #plt.show()

    # max label + polygons as an image
    return (label,  polygon_img )

def dict_to_polygon_lines( segmentation_dict: dict, img_file: Path , output_dir, factor=.5 ):
    """
    Visualize polygons on the input image.

    Args:
        segmentation_dict (dict): kraken's segmentation output, i.e. a dicitonary of the form::

             {  'text_direction': '$dir',
             'type': 'baseline',
             'lines': [
               {'baseline': [[x0, y0], [x1, y1], ...], 'boundary': [[x0, y0], [x1, y1], ... [x_m, y_m]]},
               ...
             ]
             'regions': [ ... ] }
        img (Path): input image's path.

    Output:
        None (saves a PNG image) 
    """

    with Image.open( img_file) as img:

        polygon_boundaries = [ line['boundary'] for line in segmentation_dict['lines'] ]

        draw = ImageDraw.Draw( img )

        # rendering polygons (RGBA)
        label = 0
        for lbl, polyg in enumerate( polygon_boundaries ):
            label = lbl + 1
            color = (255,0,0) if lbl%2 else (0,0,255)
            draw.polygon( [ tuple(pt) for pt in polyg ], outline=color, width=4)

        img.resize( tuple( int(elt*factor) for elt in img.size ))
        output_file = Path(output_dir, img_file.with_suffix('.polygons.png') )
        print(output_file)
        img.save(output_file, format="PNG")



def array_to_rgba_uint8( img_hw: torch.Tensor ) -> torch.Tensor:
    img_chw = img_hw.view( torch.uint8 ).reshape( img_hw.shape + (4,) ).permute(2,0,1)
    return img_chw


def rgba_uint8_to_int32( img_chw: torch.Tensor ) -> torch.Tensor:
    img_hw = img_chw.permute(1,2,0).view( torch.int32 ).reshape( img_chw.shape[1:] )
    return img_hw


def polygon_set_display( input_img: Image.Image, polygons: torch.Tensor ) -> np.ndarray:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons (torch.Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: A BGR image (3 channels, 8-bit unsigned integers).
    """

    input_img = np.asarray( input_img )

    # create mask tensor for all pred. polygon: odd-numbered polygons in R, even-numbered ones in G
    polygon_32b_img = rgba_uint8_to_int32( polygons ) # 4-channel -> 1-channel
    odd_polygon_8b1c_mask, even_polygon_8b1c_mask = [ torch.logical_and( (polygon_32b_img > 0 ), pt).numpy().astype( np.uint8 ) for pt in ((polygon_32b_img % 2), (polygon_32b_img % 2) == 0) ]
    odd_polygon_8b3c_mask, even_polygon_8b3c_mask = [ np.stack((pm, pm, pm), axis=2) for pm in (odd_polygon_8b1c_mask, even_polygon_8b1c_mask) ]
    
    # foreground color image (red or green, with RGB), masked
    full_red, full_green = [ np.full( input_img.shape, color, dtype=np.uint8) for color in (RED, GREEN) ]
    #print(foreground_red.dtype, foreground_green.dtype, odd_polygon_8b3c_mask.dtype)
    fg_red_masked, fg_green_masked = [ fg * mask  for (fg, mask) in ((full_red, odd_polygon_8b3c_mask), (full_green, even_polygon_8b3c_mask)) ]

    # combine both FG layers
    foreground = fg_red_masked + fg_green_masked 

    # BG + FG
    alpha = .75
    return (input_img * alpha + foreground * (1-alpha)).astype('uint8')

def polygon_two_set_display( input_img: Image.Image, polygons1: torch.Tensor, polygons2: torch.Tensor ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis. For clarity's sake,
    it returns 2 images:

    + one highlights the even-numbered lines, showing both sets of polygons (1: red, 2: green)
    + one highlights the odd-numbered lines, 

    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons1 (torch.Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons2 (torch.Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
    Output:
        A tuple containing two BGR images (8-bit unsigned integers), rendering the even-numbered lines
        and the odd-numbered lines, respectively.
    """
    input_img = np.asarray( input_img )

    # create mask tensor for polygon set #1
    polygon_32b_img_1, polygon_32b_img_2 = [ rgba_uint8_to_int32( p ) for p in (polygons1, polygons2) ]  # 4-channel -> 1-channel

    # sets 1 and 2, even-numbered lines
    mask1, mask2 = [ torch.logical_and( (p % 2), ( p > 0 )).numpy().astype( np.uint8 ) for p in (polygon_32b_img_1, polygon_32b_img_2) ]
    # sets 1 and 2, odd-numbered lines
    mask3, mask4 = [ torch.logical_and( (p % 2) == 0, ( p > 0 )).numpy().astype( np.uint8 ) for p in (polygon_32b_img_1, polygon_32b_img_2) ]
    pg_8b3c_mask_1, pg_8b3c_mask_2, pg_8b3c_mask_3, pg_8b3c_mask_4 = [ np.stack((p, p, p), axis=2) for p in (mask1, mask2,  mask3, mask4) ]

    full_red, full_green = [ np.full( input_img.shape, color, dtype=np.uint8 ) for color in (RED, GREEN) ]
    fg_red_masked_even, fg_green_masked_even, fg_red_masked_odd, fg_green_masked_odd = [ (fg * mask ) for (fg, mask) in ((full_red, pg_8b3c_mask_1), (full_green, pg_8b3c_mask_2), (full_red, pg_8b3c_mask_3), (full_green, pg_8b3c_mask_4)) ]

    # combine both FG layers
    foreground_even = fg_red_masked_even + fg_green_masked_even
    foreground_odd = fg_red_masked_odd + fg_green_masked_odd

    # BG + FG
    alpha = .75
    oimg1, oimg2 = [ (input_img * alpha + fg * (1-alpha)).astype('uint8') for fg in (foreground_even, foreground_odd) ]
    return ( oimg1, oimg2 ) 


def Dpolygon_two_set_display_alt( input_img: Image.Image, polygons1: torch.Tensor, polygons2: torch.Tensor ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis. 
    Even- and odd-numbered lines use different pairs of colors.

    Note: output is utterly confusing.


    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons1 (torch.Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons2 (torch.Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: A BGR image (3 channels, 8-bit unsigned integers).
    """
    # do everything in openCV
    input_img = cv2.cvtColor( np.array( input_img ), cv2.COLOR_RGB2BGR )

    polygon_32b_img_1, polygon_32b_img_2 = [ rgba_uint8_to_int32( p ) for p in (polygons1, polygons2) ]  # 4-channel -> 1-channel
    # sets 1 and 2, even-numbered lines
    mask1, mask2 = [ torch.logical_and( (p % 2), ( p > 0 )).numpy().astype( np.uint8 ) for p in (polygon_32b_img_1, polygon_32b_img_2) ]
    # sets 1 and 2, odd-numbered lines
    mask3, mask4 = [ torch.logical_and( (p % 2) == 0, ( p > 0 )).numpy().astype( np.uint8 ) for p in (polygon_32b_img_1, polygon_32b_img_2) ]
    pg_8b3c_mask_1, pg_8b3c_mask_2, pg_8b3c_mask_3, pg_8b3c_mask_4 = [ cv2.merge((p, p, p)) for p in (mask1, mask2,  mask3, mask4) ]

    # foreground color image: red/green (set 1), yellow/blue (set 2) 
    full_red, full_green, full_yellow, full_blue = [ np.full( input_img.shape, color, dtype=np.uint8 ) for color in (RED, GREEN, YELLOW, BLUE) ]
    fg_red_masked, fg_green_masked, fg_yellow_masked, fg_blue_masked = [ cv2.multiply( f, m ) for (f, m) in ((full_red, pg_8b3c_mask_1), (full_green, pg_8b3c_mask_3), (full_yellow, pg_8b3c_mask_2), (full_blue, pg_8b3c_mask_4)) ]

    # combine 4 FG layers
    foreground = cv2.add( cv2.add( cv2.add( fg_red_masked, fg_green_masked), fg_yellow_masked), fg_blue_masked)

    # BG + FG
    return cv2.addWeighted( input_img, .75, foreground, .25, 0)


def dummy():
    return True
