from PIL import Image, ImageDraw
from pathlib import Path

import torch
from torch import Tensor

import numpy as np
import seglib
import random
from typing import Tuple
import skimage as ski

RED=(255,0,0)
GREEN=(0,255,0)

def polygon_set_display( input_img_hw: Image.Image, polygons_chw: Tensor, color_scheme=2, alpha=.75 ) -> np.array:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        input_img_hw (Image.Image): the original manuscript image, as opened with PIL.
        polygons_chw (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: A RGB image ((H,W,3), 8-bit unsigned integers).
    """

    input_img_hwc = np.asarray( input_img_hw )

    colors = get_n_color_palette(2)

    # create mask tensor for all pred. polygon: odd-numbered polygons in R, even-numbered ones in G
    odd_polygon_mask_hw = seglib.mask_from_polygon_map_functional( polygons_chw, lambda m: m % 2 )
    even_polygon_mask_hw = seglib.mask_from_polygon_map_functional( polygons_chw, lambda m: torch.logical_and( m % 2 == 0, m != 0)) 
    odd_polygon_mask_hwc, even_polygon_mask_hwc = [ p.reshape( p.shape+(1,)).expand(-1,-1,3).numpy() for p  in (odd_polygon_mask_hw, even_polygon_mask_hw) ]

    # foreground color image (red or green, with RGB), masked
    full_odd_fg_hwc, full_even_fg_hwc = [ np.full( input_img_hwc.shape, color, dtype=np.uint8) for color in colors ]
    #print(foreground_red.dtype, foreground_green.dtype, odd_polygon_8b3c_mask.dtype)
    fg_odd_masked_hwc, fg_even_masked_hwc = (full_odd_fg_hwc * odd_polygon_mask_hwc, full_even_fg_hwc * even_polygon_mask_hwc) 

    # combine both FG layers
    foreground_hwc = fg_odd_masked_hwc + fg_even_masked_hwc 

    # BG + FG
    alpha = .75
    output_img = (input_img_hwc * alpha + foreground_hwc * (1-alpha)).astype('uint8') 
    print(output_img.dtype)
    print(type(output_img))

    return output_img


def polygon_set_display_2( input_img_hw: Image.Image, polygons_chw: Tensor, color_scheme=2, alpha=.75 ) -> np.array:
    """
    Render a single set of polygons using two colors (alternate between odd- and even-numbered lines).

    Args:
        input_img_hw (Image.Image): the original manuscript image, as opened with PIL.
        polygons_chw (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
    Output:
        np.ndarray: A RGB image ((H,W,3), 8-bit unsigned integers).
    """

    input_img_hwc = np.asarray( input_img_hw )

    polygon_count = torch.max( polygons_chw )

    colors = get_n_color_palette(polygon_count)

    alpha = .75

    fg_masked_hwc = np.zeros( input_img_hwc.shape ) 
    # create mask tensor for all pred. polygon: odd-numbered polygons in R, even-numbered ones in G
    for p in range(1, polygon_count ):
        polygon_mask_hw = seglib.mask_from_polygon_map_functional( polygons_chw, lambda m: m == p )
        polygon_mask_hwc = polygon_mask_hw.reshape( polygon_mask_hw.shape+(1,)).expand(-1,-1,3).numpy() 
        color = colors[ p ]
        # foreground color image (red or green, with RGB), masked
        full_fg_hwc = np.full( input_img_hwc.shape, color, dtype=np.uint8 ) 
        fg_masked_hwc += full_fg_hwc * polygon_mask_hwc

        # combine both FG layers
        foreground_hwc = fg_masked_hwc 

    # BG + FG
    output_img = (input_img_hwc * alpha ) + foreground_hwc * (1-alpha))

    return output_img.astype('uint8')



def get_n_color_palette(n: int, s=.85, v=.95) -> list:
    """
    Generate n well-distributed random colors. Use golden ratio to generate colors from the HSV color
    space. 

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

 



def polygon_two_set_display( input_img: Image.Image, polygons1: Tensor, polygons2: Tensor ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis. For clarity's sake,
    it returns 2 images:

    + one highlights the even-numbered lines, showing both sets of polygons (1: red, 2: green)
    + one highlights the odd-numbered lines, 

    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons1 (Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons2 (Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
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


def polygon_two_set_display_alt( input_img: Image.Image, polygons1: Tensor, polygons2: Tensor ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render two sets of polygons (typically: GT and pred.) using two colors, for human diagnosis. 
    Even- and odd-numbered lines use different pairs of colors.

    Note: output is utterly confusing.


    Args:
        input_img (Image.Image): the original manuscript image, as opened with PIL.
        polygons1 (Tensor): polygon set #1, encoded as a 4-channel, 8-bit tensor.
        polygons2 (Tensor): polygon set #2, encoded as a 4-channel, 8-bit tensor.
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


def plot_precision_recall_curve( confusion_matrix: Tensor ):
    pass
