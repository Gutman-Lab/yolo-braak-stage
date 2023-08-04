# Update tile label function.
from pandas import DataFrame
from geopandas import GeoDataFrame
from os.path import isfile
import numpy as np

from .utils import im_to_txt_path, delete_file, imread
from .yolov5.utils import read_yolo_label
from .box_and_contours import corners_to_polygon


def update_roi_tile_labels(df: DataFrame, label_fp: str, shape: (int, int), 
                           area_thr: float):
    """Update tile labels from an ROI.
    
    Args:
        df: Tile / image metadata.
        label_fp: ROI label filepath.
        shape: (width, height) of ROI.
        area_thr: Area threshold of object in each tile to include.
        
    """
    w, h = shape
        
    if isfile(label_fp):
        labels = []
        for box in read_yolo_label(label_fp, im_shape=(w, h), convert=True):
            label, x1, y1, x2, y2 = box.astype(int)[:5]
            
            labels.append([label, x1, y1, x2, y2, (x2 - x1) * (y2 - y1),
                           corners_to_polygon(x1, y1, x2, y2)])
    else:
        labels = []
        
    labels = GeoDataFrame(labels, columns=['label', 'x1', 'y1', 'x2', 'y2',
                                           'area', 'geometry'])
    
    for _, r in df.iterrows():
        # Subset to objects that intersect this tile.
        boxes = labels[
            labels.intersection(corners_to_polygon(
                r.x, r.y, r.x + r.tile_size, r.y + r.tile_size
            )).area / labels.area > area_thr
        ]
        
        label_fp = im_to_txt_path(r.fp)
        
        if len(boxes):
            tile_labels = ''
            
            for _, rr in boxes.iterrows():
                x1 = np.clip((rr.x1 - r.x) / r.tile_size, 0, 1)
                y1 = np.clip((rr.y1 - r.y) / r.tile_size, 0, 1)
                x2 = np.clip((rr.x2 - r.x) / r.tile_size, 0, 1)
                y2 = np.clip((rr.y2 - r.y) / r.tile_size, 0, 1)
                
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                bw, bh = x2 - x1, y2 - y1
                
                tile_labels += f'{rr.label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n'
                
            with open(label_fp, 'w') as fh:
                fh.write(tile_labels.strip())
        else:
            # Remove this label file.
            delete_file(label_fp)
