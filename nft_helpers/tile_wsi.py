# Function for tiling a WSI from local file
import large_image
import numpy as np
from .utils import get_filename, imwrite
from tqdm import tqdm
from pandas import DataFrame
import cv2 as cv
import multiprocessing as mp
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()

from os import makedirs
from os.path import join, isfile

import sys


def _tile_wsi(xy, tile_size, save_dir, filename, fp ,rgb_fill, overwrite):
    """Get and save a tile"""
    x1, y1, mask = xy
    x2, y2 = x1 + tile_size, y1 + tile_size
    tile_fp = join(save_dir, f'{filename}_x1-{x1}_y1-{y1}_x2-{x2}_y2-{y2}.png')
    
    # get the image and save it only if it does not exist
    if overwrite or not isfile(tile_fp):
        ts = large_image.getTileSource(fp)

        tile = ts.getRegion(region=dict(left=x1, top=y1, right=x2, bottom=y2), format=large_image.constants.TILE_FORMAT_NUMPY)[0][:, :, :3]

        # pad the edges of tile if it is at the edge and not the right size
        shape = tile.shape[:2]

        if shape != (tile_size, tile_size):
            # edge, pad the tile
            xpad = tile_size - shape[1]
            ypad = tile_size - shape[0]

            tile = cv.copyMakeBorder(tile, 0, ypad, 0, xpad, cv.BORDER_CONSTANT, value=rgb_fill)
        
        # if mask exists then apply it
        if mask is not None:
            tile[mask == 0] = rgb_fill
       
        imwrite(tile_fp, tile)  # save image

    return tile_fp, x1, y1, x2, y2
    

def tile_wsi(fp, save_dir, mask=None, tile_size=1280, stride=None, mask_thr=0.5, rgb_fill=(114, 114, 114), nproc=1, overwrite=False):
    """Tile a WSI from local file.
    
    INPUTS
    ------
    fp : str 
        file path of WSI file
    save_dir : str
        directory to save tiles
    mask : array-like (default: None)
        tissue binary mask, with positive values being areas to tile
    tile_size : int (default: 1280)
        size to use for tiling WSI, tiles are squares
    stride : int (default: None)
        amount to stride in x & y direction when tiling. If None, it is set equal to tile_size for no overlap
    mask_thr : float (default: 0.5)
        fraction of tile that must be in mask to be included
    rgb_fill : list (default: (114, 114, 114))
        RGB fill when padding WSI edges
    nproc : int (default: 10)
        tiling is done in parallel for speed, specify the number of processes to use
    overwrite : bool (default: False)
        if True, then images will be saved even if a file exists already
        
    RETURN
    ------
    tile_df : list
        dataframe containing data about each tile saved
    
    """
    # get the tile source
    ts = large_image.getTileSource(fp)
    
    if stride is None:
        stride = tile_size
    
    # if a mask was passed - calculate the scaling multiplier to go from WSI scan magnification to mask magnification
    wsi_metadata = ts.getMetadata()
    scale_mult = 1 if mask is None else mask.shape[0] / wsi_metadata['sizeY']
    
    # calculate the number of pixels in a tile (low res) that must be positive to include tile
    mask_tile_size = int(tile_size * scale_mult)
    tile_thr = (mask_tile_size * mask_tile_size) * mask_thr  # pixel threshold
    
    # pad the mask
    if mask is not None:
        mask = cv.copyMakeBorder(mask, 0, mask_tile_size, 0, mask_tile_size, cv.BORDER_CONSTANT, value=0)
    
    # create the x, y coordinates to use when tiling, ignoring tiles with not enough mask
    xys = []
    
    for x in range(0, wsi_metadata['sizeX'], stride):
        for y in range(0, wsi_metadata['sizeY'], stride):
            # check if tile is sufficiently in mask
            if mask is None:
                xys.append((x, y, None))
            else:
                tile_x, tile_y = int(x*scale_mult), int(y*scale_mult)
                tile_mask = mask[tile_y:tile_y+mask_tile_size, tile_x:tile_x+mask_tile_size]
                
                if np.count_nonzero(tile_mask) >= tile_thr:
                    # when passing the tile mask - resize it to be at the tisze of the tile
                    xys.append((x, y, cv.resize(tile_mask, (tile_size, tile_size), interpolation=cv.INTER_NEAREST)))
                    
                    
    fn = get_filename(fp)  # name of the wsi without the extension
    makedirs(save_dir, exist_ok=True)  # create dir to save tile images
        
    pool = mp.Pool(nproc)
    jobs = [pool.apply_async(func=_tile_wsi, args=(xy, tile_size, save_dir, fn, fp, rgb_fill, overwrite)) for xy in xys]
    pool.close()
    
    tile_df = [job.get() for job in tqdm(jobs)]
    
                                            # file path to tile image, coordinates of the tile in WSI scan mag space
    return DataFrame(data=tile_df, columns=['fp', 'x1', 'y1', 'x2', 'y2'])
       