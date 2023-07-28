from typing import Union, List
from geopandas import GeoDataFrame
import numpy as np
from os.path import isfile
from pandas import DataFrame

from .yolov5.utils import read_yolo_label
from .box_and_contours import corners_to_polygon


def match_boxes(
    true: Union[str, np.array], pred: Union[str, np.array], 
    iou_thr: float = 0.5, labels: List[int] = None
) -> GeoDataFrame:
    """Match predictions between true and pred for boxes.
    
    Format of boxes: label, x1, y1, x2, y2 and it does not matter if they are 
    int or normalized to image height and width so long as they are both in the 
    same format.
    
    Args:
        true: Box array of true labels.
        pred: Box array of pred labels.
        iou_thr: IoU threshold when matching true and predicted boxes.
        labels: Labels of interest, -1 automatically used for background.
    
    Returns:
        Info of each match.
    
    """
    # Read true and pred metrics in format: [label, x1, y1, x2, y2]
    if isinstance(true, str):
        true = read_yolo_label(true, convert=True) if isfile(true) else []
        
    if isinstance(pred, str):
        pred = read_yolo_label(pred, convert=True) if isfile(pred) else []
        
    # Convert true and predictions to geo dataframes.
    true_df = []
    
    for box in true:
        label, x1, y1, x2, y2 = box[:5]
        
        true_df.append([int(label), x1, y1, x2, y2, 
                        corners_to_polygon(x1, y1, x2, y2)])
    
    true_df = GeoDataFrame(
        true_df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'geometry']
    )
    
    pred_df = []
    
    for box in pred:
        label, x1, y1, x2, y2 = box[:5]
        
        pred_df.append([int(label), x1, y1, x2, y2, box[5],
                        corners_to_polygon(x1, y1, x2, y2)])
    
    pred_df = GeoDataFrame(
        pred_df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry']
    )
    
    # Match predictions and true boxes
    matches = []
    
    for i, r in true_df.iterrows():
        # Find matches for this true box.
        ious = pred_df.intersection(r.geometry).area / + \
               pred_df.union(r.geometry).area
        ious = ious[ious > iou_thr]  # filter by IoU
        
        if len(ious):
            # Match was found, take the match with highest IoU.
            ious = ious.sort_values(ascending=False)
            j, iou = ious.index[0], ious.values[0]
            p = pred_df.loc[j]
            
            # Add the match and remove this index.
            matches.append([
                int(r.label), int(p.label), r.x1, r.y1, r.x2, r.y2, p.conf, iou
            ])
            
            pred_df = pred_df.drop(index=j)
        else:
            # No match.
            matches.append([
                int(r.label), -1, r.x1, r.y1, r.x2, r.y2, 0, 0
            ])
            
    # Add negative posistives.
    for _, r in pred_df.iterrows():
        matches.append([
            -1, int(r.label), r.x1, r.y1, r.x2, r.y2, r.conf, 0
        ])
            
    return DataFrame(matches, columns=['true', 'pred', 'x1', 'y1', 'x2', 'y2',
                                       'conf', 'iou'])
