from pandas import DataFrame
from geopandas import GeoDataFrame

from os import makedirs
from os.path import join, isfile

from .utils import read_yolo_label, non_max_suppression, remove_contained_boxes
from ..utils import get_filename
from ..box_and_contours import corners_to_polygon


def create_roi_preds(
    tiles: DataFrame, pred_dir: str, save_dir: str, iou_thr: int = 0.4, 
    contained_thr: int = 0.7, conf: float = 0.25, tile_size: int = 1280
):
    """Create ROI predictions from corresponding tile predictions. This function
    works on a set of ROIs not a single one by passing ROI info as a dataframe 
    and corresponding tile info as another dataframe.
    
    Args:
        tiles: Tile metadata, key columns: 'roi_fp', 'fp', 'w', 'h', 'x', 'y',
            'roi_h', 'roi_w'.
        pred_dir: Directory with tile prediction label files.
        save_dir: Directory to save ROI prediction labels.
        iou_thr: NMS IoU threshold when merging predictions.
        contained_thr: Contained area area threshold.
        conf: Confidence threshold to keep prediction boxes.
        tile_size: Size of tile image.
        
    """
    makedirs(save_dir, exist_ok=True)
    
    for roi_fp in tiles.roi_fp.unique():
        roi_tiles = tiles[tiles.roi_fp == roi_fp]
        
        # Read the tiles into a geodatafarme.
        preds = []
        
        for _, r in roi_tiles.iterrows():
            # Check for predictions.
            label_fp = join(pred_dir, get_filename(r.fp) + '.txt')
            
            if isfile(label_fp):
                for box in read_yolo_label(
                    label_fp, im_shape=(tile_size, tile_size), convert=True
                ):
                    label, x1, y1, x2, y2 = box[:5].astype(int)
                    
                    if box[5] > conf:
                        x1, y1 = x1 + r.x, y1 + r.y
                        x2, y2 = x2 + r.x, y2 + r.y

                        preds.append([
                            label, x1, y1, x2, y2, box[5], 
                            corners_to_polygon(x1, y1, x2, y2)
                        ])
                    
        if len(preds):
            # Use Geopandas to filter out overlapping tiles.
            preds = GeoDataFrame(
                preds, 
                columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry']
            )
            preds = remove_contained_boxes(
                non_max_suppression(preds, iou_thr), contained_thr
            )
            
            # Save the predictions to a file.
            with open(join(save_dir, get_filename(roi_fp) + '.txt'), 'w') as fh:
                labels = ''
                
                for _, p in preds.iterrows():
                    xc = ((p.x2 + p.x1) / 2) / r.roi_w
                    yc = ((p.y2 + p.y1) / 2) / r.roi_h
                    bw, bh = (p.x2 - p.x1) / r.roi_w, (p.y2 - p.y1) / r.roi_h
                    
                    labels += f'{p.label:.0f} {xc:.6f} {yc:.6f} {bw:.6f} ' + \
                              f'{bh:.6f} {p.conf:.6f}\n'
                    
                fh.write(labels.strip())
