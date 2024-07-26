from PIL import Image, ImageDraw
from pathlib import Path

import torch
from torch import Tensor

import numpy as np
import seglib
import random
from typing import Tuple
import skimage as ski

def display_polygon_set_from_img_and_tensor_files( img_file: str, polygon_file: str, color_count=0, alpha=.75) -> np.ndarray:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        img_file (str): path to the original manuscript image.
        polygon_file (str): path to the pickled polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        polygons_chw = torch.load( polygon_file ) 
        return display_polygon_set( input_img_hw, polygons_chw, color_count, alpha )

def display_two_polygon_sets_from_img_and_tensor_files( img_file: str, polygon_file_1: str, polygon_file_2: str, bg_alpha=.75) -> np.ndarray:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis.
   
    Args:
        img_file (str): path to the original manuscript image.
        polygon_file_1 (str): path to the first pickled polygon set, encoded as a 4-channel, 8-bit tensor.
        polygon_file_2 (str): path to the second pickled polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    with Image.open(img_file) as input_img_hw:
        polygons_1_chw = torch.load( polygon_file_1 )
        polygons_2_chw = torch.load( polygon_file_2 )
        return display_two_polygon_sets( input_img_hw, polygons_1_chw, polygons_2_chw, bg_alpha )

def display_polygon_set( input_img_hw: Image.Image, polygons_chw: Tensor, color_count=0, alpha=.75 ) -> np.ndarray:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        input_img_hw (Image.Image): the original manuscript image, as opened with PIL.
        polygons_chw (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """

    input_img_hwc = np.asarray( input_img_hw )
    polygon_count = torch.max( polygons_chw )

    colors = get_n_color_palette( color_count ) if color_count else get_n_color_palette(int( polygon_count ))

    fg_masked_hwc = np.zeros( input_img_hwc.shape ) 

    output_img = input_img_hwc.copy()
    
    for p in range(1, polygon_count+1 ):
        # flat binary mask
        polygon_mask_hw = seglib.mask_from_polygon_map_functional( polygons_chw, lambda m: m == p )
        fg_masked_hwc[ polygon_mask_hw ] += colors[ p % len(colors) ]

    # in original image, transparency applies only to the polygon pixels
    alpha_mask = fg_masked_hwc != 0
    alphas = np.full( alpha_mask.shape, 1.0 )
    alphas[ alpha_mask ] = alpha

    # combine: BG + FG
    # use this statement instead to make the polygons more visible
    #output_img = (input_img_hwc * alpha ) + fg_masked_hwc * (1-alpha)
    output_img = (input_img_hwc * alphas ) + fg_masked_hwc * (1-alpha)

    return output_img.astype('uint8')


def display_two_polygon_sets( input_img_hw: Image.Image, polygons_1_chw: Tensor, polygons_2_chw: Tensor, bg_alpha=.5 ) -> np.ndarray:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis.

    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons_1_chw (Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons_2_chw (Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: a RGB image (H,W,3), 8-bit unsigned integers.
    """
    input_img_hwc = np.asarray( input_img_hw )
    polygon_count_1 = torch.max( polygons_1_chw )
    polygon_count_2 = torch.max( polygons_2_chw )

    #colors = (255,0,0), (0,0,255)
    colors = get_n_color_palette( 2, s=.99, v=.99 )

    fg_masked_hwc = np.zeros( input_img_hwc.shape ) 

    output_img = input_img_hwc.copy()
    
    # create a single mask for each set
    mask_1_hw = seglib.mask_from_polygon_map_functional( polygons_1_chw, lambda m: m != 0 )
    mask_2_hw = seglib.mask_from_polygon_map_functional( polygons_2_chw, lambda m: m != 0 )

    fg_masked_hwc[ mask_1_hw ] = colors[0]
    fg_masked_hwc[ mask_2_hw ] += colors[1]

    # in original image, transparency applies only to the polygon pixels
    alpha_mask = fg_masked_hwc != 0
    alphas = np.full( alpha_mask.shape, 1.0 )
    alphas[ alpha_mask ] = bg_alpha

    # combine: BG + FG
    output_img = (input_img_hwc * alphas ) + fg_masked_hwc * (1-bg_alpha)
    
    return output_img.astype('uint8')

def get_n_color_palette(n: int, s=.85, v=.95) -> list:
    """
    Generate n well-distributed random colors. Use golden ratio to generate colors from the HSV color
    space. 

    Reference: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/

    Args:
        n (int): number of color to generate.

    Output:
        list: a list of (R,G,B) tuples
    """
    golden_ratio_conjugate = 0.618033988749895
    random.seed(13)
    h = random.random() 
    palette = np.zeros((1,n,3))
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        palette[0][i]=(h, s, v)
    return (ski.color.hsv2rgb( palette )*255).astype('uint8')[0].tolist()

 

