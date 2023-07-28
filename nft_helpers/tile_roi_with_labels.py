from typing import Tuple
from pandas import DataFrame
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import numpy as np
import cv2 as cv

from .utils import imread, im_to_txt_path, imwrite, get_filename
from .yolov5.utils import read_yolo_label
from .box_and_contours import corners_to_polygon

from os import makedirs
from os.path import isfile, join


def tile_roi_with_labels(
    fp: str, save_dir: str, tile_size: int = 1280, stride: int = None, 
    boundary_thr: float = 0.2, fill: Tuple[int] = (114, 114, 114), 
    box_thr: float = 0.5
) -> DataFrame:
    """Tile an ROI image with labels.
    
    Args:
        fp: Image filepath, should be in an '/images/ directory'.
        save_dir: Location to save images and labels.
        tile_size: Size of tile, uses square tiles only.
        stride: Stride to use when tiling, if None then it is set equal to 
            tile_size (no overlap between tiles).
        boundary_thr: If ROI has a boundary (for rotated ROIs) then a tile must
            have sufficient area in boundary to be included (0.0 - 1.0).
        fill: RGB when padding image.
        box_thr: Area threshold of box that must be in a tile.
       
    Returns:
        Metadata of tiles saved.
        
    """
    # read the image
    img = imread(fp)
    h, w = img.shape[:2]
    
    # look for labels and boundaries
    label_fp = im_to_txt_path(fp)
    boundary_fp = im_to_txt_path(fp, txt_dir='boundaries')
    
    if isfile(label_fp):
        labels = read_yolo_label(label_fp, im_shape=(w, h), convert=True)
    else:
        labels = []
    
    # Convert the labels into a GeoDataFrame.
    label_df = []
    
    for box in labels:
        label = box[0]
        x1, y1, x2, y2 = box[1:5].astype(int)
        
        label_df.append([label, x1, y1, x2, y2, 
                         corners_to_polygon(x1, y1, x2, y2)])
        
    label_df = GeoDataFrame(
        label_df, 
        columns=['label', 'x1', 'y1', 'x2', 'y2', 'geometry']
    )
    label_areas = label_df.area
    
    # For the boundary, create a polygon object.
    if isfile(boundary_fp):
        # format the boundaries in to a countour shape
        with open(boundary_fp, 'r') as fh:
            boundaries = [float(c) for c in fh.readlines()[0].split(' ')]
            
        # scale to the image size.
        roi = Polygon(
            (np.reshape(boundaries, (-1, 2)) * [w, h]).astype(int)
        )
    else:
        # Default: the whole image is the ROI.
        roi = corners_to_polygon(0, 0, w, h)  
    
    img_dir = join(save_dir, 'images')
    label_dir = join(save_dir, 'labels')
    makedirs(img_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    
    # Default stride is no-overlap.
    if stride is None:
        stride = tile_size  # default behavior - no overlap
    
    # Pad the image to avoid getting tiles not of the right size.
    img = cv.copyMakeBorder(img, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, 
                            value=fill)
    
    # Get the top left corner of each tile.
    xys = list(((x, y) for x in range(0, w, stride) 
                for y in range(0, h, stride)))
        
    tile_df = []  # track tile data.

    # Pre-calculate the number of pixels in tile that must be in ROI to include.
    intersection_thr = tile_size**2 * boundary_thr
    
    # loop through each tile coordinate
    for xy in xys:
        # Check if this tile is sufficiently in the boundary.
        x1, y1 = xy
        x2, y2 = x1 + tile_size, y1 + tile_size
        
        tile_pol = corners_to_polygon(x1, y1, x2, y2)
        intersection = roi.intersection(tile_pol).area
        
        if intersection > intersection_thr:
            # Get the tile image.
            tile = img[y1:y2, x1:x2]
            
            # Create a name for the tile image / label.
            fn = f'{get_filename(fp)}-x{x1}y{y1}.'
            
            img_fp = join(img_dir, fn + 'png')
            
            tile_df.append([img_fp, fp, x1, y1, tile_size])
            
            if not isfile(img_fp):
                imwrite(img_fp, tile)
                
            # Find all boxes that intersect
            label_intersection = label_df.geometry.intersection(tile_pol).area
            
            tile_boxes = label_df[label_intersection / label_areas > box_thr]
            
            # save these as normalized labels, threshold the box edges.
            if len(tile_boxes):
                labels = ''
                
                for _, r in tile_boxes.iterrows():
                    # Shift coordinates to be relative to this tile.
                    xx1 = np.clip(r.x1 - x1, 0, tile_size) / tile_size
                    yy1 = np.clip(r.y1 - y1, 0, tile_size) / tile_size 
                    xx2 = np.clip(r.x2 - x1, 0, tile_size) / tile_size
                    yy2 = np.clip(r.y2 - y1, 0, tile_size) / tile_size

                    xc, yc = (xx1 + xx2) / 2, (yy1 + yy2) / 2
                    bw, bh = xx2 - xx1, yy2 - yy1
                    labels += f'{r.label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n'
                    
                with open(im_to_txt_path(img_fp), 'w') as fh:
                    fh.write(labels.strip())
                    
    return DataFrame(tile_df, columns=['fp', 'roi_fp', 'x', 'y', 'tile_size'])
