# function for extracting tissue mask from a WSI
import large_image
from histomicstk.saliency.tissue_detection import get_tissue_mask
import numpy as np


def extract_tissue_mask(fp, lr_mag=0.25, sigma=5):
    """Extract a tissue mask from a WSI image file.
    
    INPUTS
    ------
    fp : str
        file path to WSI
    lr_mag : float (default: 0.25)
        low res magnification to apply tissue extraction on
    sigma : int (default: 5)
        sigma of gaussian filter
    
    RETURNS
    -------
    im : array-like
        RGB low res image of the WSI
    tissue_mask : array-like
        2-D tissue mask, 255 = tissue pixels, 0 = background pixels
    : dict
        large image metadata of WSI
        
    """
    # get the tile source
    ts = large_image.getTileSource(fp)
    
    # get the low res image
    im = ts.getRegion(scale={'magnification': lr_mag}, format=large_image.tilesource.TILE_FORMAT_NUMPY)[0]
    
    # estimate the tissue mask
    tissue_mask = (get_tissue_mask(im, sigma=sigma)[0] > 0).astype(np.uint8) * 255
    
    return im, tissue_mask, ts.getMetadata()
