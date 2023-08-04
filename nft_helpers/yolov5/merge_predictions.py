from ..utils import im_to_txt_path
from .utils import read_yolo_label, non_max_suppression
from os.path import isfile
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
from pandas import DataFrame


def merge_predictions(df, iou_thr=0.75, rm_contained=True):
    """Merge the label text files for images taken from the same large image (i.e. WSI). Handles merging for boxes that overlap for images that overlap.
    Approach is agnostic NMS  + remove small contained predictions.
    
    INPUTS
    ------
    df : dataframe
        each row contains info about an image, with columns "fp" (file path, prediction will be looked for to match the filename with the /images/ folder replaced with /labels/),
        "x1", "y1", "x2", "y2" (coordinates).
    iou_thr : float (default: 0.75)
        NMS IoU threshold
    rm_contained : bool (default: True)
        run function after NMS that removes prediction boxes contained in others, adds considerable time to completion
    
    RETURN
    ------
    predictions : DataFrame
        each row a prediction box, with columns: label, x1, y1, x2, y2, conf
    
    """
    # read the image predictions, format into GeoDataFrame with coordinates in WSI space (i.e. point 0, 0 is the top left of WSI)
    predictions = []
    
    for _, r in df.iterrows():
        shape = (int(r.x2-r.x1), int(r.y2-r.y1))
        
        pred_fp = im_to_txt_path(r.fp)
        
        if isfile(pred_fp):
            # return the predictions for each prediction file - and also add the 
            preds = read_yolo_label(pred_fp, im_shape=shape, shift=(-r.x1, -r.y1), convert=True)
            
            for pred in preds:
                label, x1, y1, x2, y2, conf = pred
                
                predictions.append([label, x1, y1, x2, y2, conf, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])])
            
    predictions = GeoDataFrame(data=predictions, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry'])
    
    if len(predictions):
        # clean up the predictions by removing overlapping predictions using NMS & removes contained functions
        predictions = non_max_suppression(predictions, iou_thr, rm_contained=rm_contained)
    
    del predictions['geometry']
    
    return DataFrame(predictions)
