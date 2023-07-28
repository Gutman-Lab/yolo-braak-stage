# WSI inference workflow for YOLOv5 and NFT project
from colorama import Fore, Style
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import re
import large_image
from shutil import rmtree
from pandas import DataFrame
from time import perf_counter

from os import makedirs
from os.path import join, isfile, abspath

from ..utils import imread, get_filename, im_to_txt_path
from ..girder_dsa import get_annotations_documents
from ..box_and_contours import get_contours, contours_to_points
from .. import tile_wsi
from . import predict
from .utils import read_yolo_label, non_max_suppression, remove_contained_boxes

COLORS = ['rgb(0,0,255)', 'rgb(255,0,0)', 'rgb(0,255,0)']
LABELS = ['Pre-NFT', 'iNFT']


def wsi_inference(fp, gc=None, weights='yolov5m6.pt', mask=None, doc_name='default', wsi_id=None, exist_ok=True, stride=960, tile_size=1280, nproc=10, rgb_fill=(114, 114, 114),
                  device=None, conf_thr=0.5, nms_iou_thr=0.6, contained_thr=0.8, save_fp=None, colors=None, labels=None, mask_thr=0.25):
    """Predict objects on an entire WSI using YOLOv5 model and an inference workflow.
    
    INPUTS
    ------
    fp : str
        file path to the WSI file
    gc : girderClient
        authenticated girder client, needed push results as annotations
    weights : str
        file path to .pt file with yolov5 weights
    mask : str or array-like
        binary mask to specify regions of WSI to inference on. Either pass the array directly or a file path to the mask file. The mask can be (and should be) at a smaller
        scale then the WSI but should be of the same aspect ratio. If aspect ratio is not the same the results will be unexpected. The width will be used to calculate the 
        downscale factor.
    doc_name : str
        tissue mask, tile, and prediction output will be pushed as annotations of this document name with the following appends: -tissue, -tiles, -predictions. 
        gc and wsi_id must be passed
    wsi_id : str
        WSI id, if passed the results will be pushed as annotations, must also pass gc
    exist_ok : bool
        if True, then a document of the same name as an existing one may be created
    tile_size : int
        size of square tiles used to break image into smaller regions
    stride : int
        stride when tiling WSIs
    nproc : int
        number of parallel processing - used to speed up tiling
    rgb_fill : tuple
        int tuple for the RGB color to use when filling parts of tiles not in mask or in WSI
    device : str
        ids of GPUs to use when predicting, i.e. '0,1,2', or leave as None to use all available
    conf_thr : float
        confidence threshold for model predictions, boxes with confidence lower than this are removed
    nms_iou_thr : float
        threshold when applying non max suppresion (done both within tiles and when merging predictions from adjacent tiles)
    contained_thr : float
        threshold of boxes to be removed if mostly contained in other larger boxes
    save_fp : str
        if not None then the predictions are saved yolo label file
    colors : list
        list of colors to use for the prediction boxes when pushing to the DSA. If None then default colors are used. The colors should be specified as 'rgb(#,#,#)' for RGB. 
        This is the line colors of the boxes. Boxes are pushed without any fill.
    labels : list
        list of labels for the predictions when pushing to the DSA, such as ['Pre-NFT', 'iNFT'] (default if None is passed).
    mask_thr : float
        amount of tissue in tile to be included, by fraction of area of tile
     
    RETURN
    ------
    tile_df : dataframe
        the tile dataframe
    pred_df : dataframe
        dataframe with each row a box prediction
    logs : time it takes to do each step, as a string
    """
    start_time = perf_counter()
    
    # track time it takes to do each step: tiling, predicting, merging, push to DSA, cleaning up, total] - save it to a text file, format in minutes
    # read the mask if passed as a file path
    if isinstance(mask, str):
        mask = imread(mask, grayscale=True)
        
    # calculate the scale factor between mask and WSI
    wsi_metadata = large_image.getTileSource(fp).getMetadata()
    
    if mask is not None:
        scale_mult = wsi_metadata['sizeX'] / mask.shape[1]  # mask scale -> wsi scale
        
    # tile the WSI
    t = perf_counter()
    
    tile_dir = abspath(f".temp/{re.sub(' +', '_', get_filename(fp))}")
    makedirs(tile_dir, exist_ok=True)
    print(Fore.CYAN + 'saving temporary tile images...' + Style.RESET_ALL)
    tile_df = tile_wsi(fp, join(tile_dir, 'images'), mask=mask, tile_size=tile_size, stride=stride, mask_thr=mask_thr, nproc=nproc, rgb_fill=rgb_fill)
    logs = f'Tiling: {perf_counter() - t}\n'
    
    # predict on the tiles
    print(Fore.CYAN + 'predicting tile labels...' + Style.RESET_ALL)
    t = perf_counter()
    predict(join(tile_dir, 'images'), tile_dir, weights, device=device, conf_thr=conf_thr, iou_thr=nms_iou_thr, im_size=tile_size)
    logs += f'Predicting: {perf_counter() - t}\n'
    
    # compile the predictions into a geo dataframe to allow merging overlapping predictions
    # merge the predictions using NMS
    print(Fore.CYAN + 'merging predictions...' + Style.RESET_ALL)
    t = perf_counter()
    
    pred_df = []
    
    for _, r in tile_df.iterrows():
        # check for label text file
        label_fp = im_to_txt_path(r.fp)
        if isfile(label_fp):
            # read the label boxes, shift to be in WSI space (0,0 = top left of WSI)
            for pred in read_yolo_label(label_fp, im_shape=tile_size, shift=(-r.x1, -r.y1), convert=True):  
                label, x1, y1, x2, y2, conf = pred
                pred_df.append([r.fp, label, x1, y1, x2, y2, conf, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), (x2-x1)*(y2-y1)])

    pred_df = GeoDataFrame(
        pred_df, 
        columns=[
            'fp', 'label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry', 
            'box_area'
        ]
    )
    
    pred_df = non_max_suppression(
        pred_df, nms_iou_thr
    ).sort_values(by='box_area', ascending=False).reset_index(drop=True)
    pred_df = remove_contained_boxes(pred_df, contained_thr)
    
    logs += f'Merging predictions: {perf_counter() - t}\n'
    
    # save the results to file
    del pred_df['geometry']
    
    if save_fp is not None:
        lines = ''

        for _, r in pred_df.iterrows():
            lines += f'{int(r.label)} {int(r.x1)} {int(r.y1)} {int(r.x2)} {int(r.y2)} {r.conf:.6f}\n'

        with open(save_fp, 'w') as fh:
            fh.write(lines.strip())
    
    # push results to DSA annotations
    if wsi_id is not None and gc is not None:
        # remove current docs of the same name if exist_ok is false
        if not exist_ok:
            # get the current annotation docs of WSI
            for doc in get_annotations_documents(gc, wsi_id):
                if doc['annotation']['name'] in (f'{doc_name}-tissue', f'{doc_name}-tiles', f'{doc_name}-predictions'):
                    gc.delete(f'/annotation/{doc["_id"]}')
        
        if mask is not None:
            # extract the tissue boundaries as DSA style points
            tissue_points = contours_to_points([(contour * scale_mult).astype(int) for contour in get_contours(mask, enclosed_contours=True)])
        
            # convert the points to elements and stick them in a document to push as DSA annotations
            tissue_els = []
            
            for pt in tissue_points:
                tissue_els.append({
                    'group': 'wsi-tissue',
                    'type': 'polyline',
                    'lineColor': 'rgb(0,255,0)',
                    'lineWidth': 4.0,
                    'closed': True,
                    'points': pt,
                    'label': {'value': 'wsi-tissue'},
                })

            _ = gc.post(f'/annotation?itemId={wsi_id}', json={'name': f'{doc_name}-tissue', 'description': '', 'elements': tissue_els})
            
        if len(pred_df):
            colors = COLORS if colors is None else colors
            labels = LABELS if labels is None else labels
            pred_els = []

            for _, r in pred_df.iterrows():
                tile_w, tile_h = r.x2 - r.x1, r.y2 - r.y1
                tile_center = [(r.x2 + r.x1) / 2, (r.y2 + r.y1) / 2, 0]
                label = int(r.label)

                pred_els.append({
                    'lineColor': colors[label],
                    'lineWidth': 2,
                    'rotation': 0,
                    'type': 'rectangle',
                    'center': tile_center,
                    'width': tile_w,
                    'height': tile_h,
                    'label': {'value': labels[label]},
                    'group': labels[label]
                })

            _ = gc.post(
                f'/annotation?itemId={wsi_id}', 
                json={
                    'name': f'{doc_name}-predictions', 
                    'description': '', 
                    'elements': pred_els
                }
            )
            
    t = perf_counter()
    rmtree(tile_dir)
    logs += f'Cleaning up: {perf_counter() - t}\n'
    
    logs += f'Total time: {perf_counter() - start_time}'
    return tile_df, DataFrame(pred_df), logs
