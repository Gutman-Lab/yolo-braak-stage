from pandas import DataFrame
from geopandas import GeoDataFrame
from typing import Tuple
from os.path import join, abspath, isfile
from shutil import rmtree

from . import predict
from .utils import read_yolo_label, remove_contained_boxes, non_max_suppression
from .. import tile_roi_with_labels
from ..utils import imread, get_filename
from ..box_and_contours import corners_to_polygon


def roi_inference(
    fp: str, weights: str, tile_size: int = 1280, stride: int = 960, 
    temp_dir: str = '.temp', save_fp: str = None, yolodir: str = '/workspace/yolo',
    iou_thr: float = 0.4, contained_thr: float = 0.7, boundary_thr: float = 0.2,
    fill: Tuple[int] = (114, 114, 114), device: str = None,
    conf_thr: float = 0.4
) -> GeoDataFrame:
    """
    Run inference on an ROI with a YOLO model for prediction of NFTs.
    
    Args:
        fp: Filepath to ROI image. Should be in an /images/ directory and 
            have /labels/ and /boundaries/ text files of similar name.
        weights: Weights to use when predicting.
        tile_size: Size of image when tiling.
        stride: Stride when tiling, should be less than tile_size to overlap
            adjacent tiles.
        temp_dir: Tile images are saved temporarily and then deleted.
        save_fp: Save resutls to a text file.
        yolodir: Location of the YOLO repo, to run detection script.
        iou_thr: IoU threshold used when merging predictions.
        contained_thr: IoU threshold used when removing small boxes in others.
        boundary_thr: When tiling ROI with boundaries this excludes tiles mostly
            not in ROI region (handles rotated ROIs).
        fill: When tiling an ROI that is rotated / has boundary file, this RGB color is
            used to mask out regions outside of ROI.
        device: Specify the GPUs to use when predicting, either pass IDs ("0,1,2"), "cpu",
            or None to use all available.
        conf_thr: Remove predictions of low confidence.
            
    Returns:
        Predictions.
    
    """
    # Make sure this is an abosulte path.
    temp_dir = abspath(temp_dir)
        
    # Tile the ROI.    
    tile_df = tile_roi_with_labels.tile_roi_with_labels(
        fp, temp_dir, tile_size=tile_size, stride=stride, boundary_thr=boundary_thr,
        fill=fill
    )
        
    # Predict on tiles.
    predict(
        join(temp_dir, 'images'), join(temp_dir, 'predictions'), weights, 
        device=device, conf_thr=conf_thr, iou_thr=iou_thr, im_size=tile_size,
        yolodir=yolodir
    )

    # Read the tile predictions into geodataframe to merge.
    pred_df = []

    for _, r in tile_df.iterrows():
        fn = get_filename(r.fp)

        label_fp = join(temp_dir, 'predictions', 'labels', fn + '.txt')

        if isfile(label_fp):
            for box in read_yolo_label(label_fp, im_shape=tile_size, 
                                       convert=True):
                label, conf = int(box[0]), box[5]
                x1, y1, x2, y2 = box[1:5].astype(int)
                x1, y1, x2, y2 = x1 + r.x, y1 + r.y, x2 + r.x, y2 + r.y

                pred_df.append([
                    label, x1, y1, x2, y2, conf, 
                    corners_to_polygon(x1, y1, x2, y2)
                ])

    # Compile into GeoDataframe
    pred_df = GeoDataFrame(
        pred_df, 
        columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry']
    )
        
    if len(pred_df):        
        # Merge boxes that overlap.
        pred_df = non_max_suppression(pred_df, iou_thr)
        pred_df = remove_contained_boxes(pred_df, contained_thr)
        
        if save_fp is not None:
            # Format the boxes to YOLO format and save.
            h, w = imread(fp).shape[:2]  # need image size to normalize coordinates
            
            labels = ''

            for _, r in pred_df.iterrows():
                xc, yc = (r.x1 + r.x2) / 2 / w, (r.y1 + r.y2) / 2 / h
                bw, bh = (r.x2 - r.x1) / w, (r.y2 - r.y1) / h
                labels += f'{r.label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f} ' + \
                          f'{r.conf:.4f}\n'

            with open(save_fp, 'w') as fh:
                fh.write(labels.strip())
            
    # Delete the temorary tile directory
    rmtree(temp_dir)
    
    return pred_df
