# Yolov5 utlities
# Functions:
# - get_devices
# - create_dataset_txt
# - read_yolo_label
# - non_max_suppression
# - create_datasets
# - remove_contained_boxes
# - draw_boxes
import torch
import numpy as np
import yaml
import shutil
import multiprocessing as mp
from geopandas import GeoSeries
from glob import glob
from pandas import read_csv, DataFrame
from typing import List, Union
import cv2 as cv

from ..box_and_contours import convert_box_type
from ..utils import get_filename, imread

from os import makedirs
from os.path import join


def get_devices(device=None):
    """Get the number of GPU devices, if available. 
    
    INPUTS
    ------
    device : str
        either a string in the format 0,1,2 for device ids, or None if it should be inferred by torch
    
    RETURNS
    -------
    device : str
        id of devices in format: 0,1,2,3
    n_devices : int
        number of devices
    
    """
    if device is None:
        # calcualte the device string, with all available devices
        device = ''
        n_devices = torch.cuda.device_count()
        
        if n_devices:
            for i in range(n_devices):
                device += str(i) + ','
            device = device[:-1]
        else:
            device = None
    else:
        n_devices = len(device.split(','))
        
    return device, n_devices


def create_dataset_txt(df: DataFrame, savepath: str, 
                       annotator: List[str] = None, dataset: List[str] = None, 
                       fp_col: str = 'fp'):
    """
    Create yolo txt file for datasest.
    
    Args:
        df: Image data.
        savepath: Save path.
        annotator: Filter input dataframe by annotators.
        dataset: Filter input dataframe by datasets.
        fp_col: Column in input dataframe that contains the image paths.
    
    """
    # filter images by parameters
    df = df.copy()
    
    if dataset is not None:
        df = df[df.dataset.isin(dataset)]
        
    if annotator is not None:
        df = df[df.annotator.isin(annotator)]

    # create text file
    if len(df):
        # for each image add filepath to text
        lines = ''
        
        for _, r in df.iterrows():
            lines += f'{r[fp_col]}\n'

        with open(savepath, 'w') as f: 
            f.writelines(lines.strip())
    else:
        print("No images found, can't create text file")
        
        
def read_yolo_label(filepath, im_shape=None, shift=None, convert=False):
    """Read a yolo label text file. It may contain a confidence value for the labels or not, will handle both cases
    
    INPUTS
    ------
    filepath : str
        the path of the text file
    im_shape : tuple or int (default: None)
        image width and height corresponding to the label, if an int it is assumed both are the same. Will scale coordinates
        to int values instead of normalized if given
    shift : tuple or int (default: None)
        shift value in the x and y direction, if int it is assumed to be the same in both. These values will be subtracted and applied
        after scaling if needed
    convert : bool (default: False)
        If True, convert the output boxes from yolo format (label, x-center, y-center, width, height, conf) to (label, x1, y1, x2, y2, conf)
        where point 1 is the top left corner of box and point 2 is the bottom corner of box
    
    RETURN
    ------
    coords : array
        coordinates array, [N, 4 or 5] depending if confidence was in input file
    
    """
    coords = []
    
    with open(filepath, 'r') as fh:
        for line in fh.readlines():
            if len(line):
                coords.append([float(ln) for ln in line.strip().split(' ')])
                
    coords = np.array(coords)
    
    # scale coords if needed
    if im_shape is not None:
        if isinstance(im_shape, int):
            w, h = im_shape, im_shape
        else:
            w, h = im_shape[:2]
            
        coords[:, 1] *= w
        coords[:, 3] *= w
        coords[:, 2] *= h
        coords[:, 4] *= h
        
    # shift coords
    if shift is not None:
        if isinstance(shift, int):
            x_shift, y_shift = shift, shift
        else:
            x_shift, y_shift = shift[:2]
            
        coords[:, 1] -= x_shift
        coords[:, 2] -= y_shift
        
    if convert:
        coords[:, 1:5] = convert_box_type(coords[:, 1:5])
        
    return coords


def non_max_suppression(df, thr):
    """Apply non-max suppression (nms) on a set of prediction boxes. 
    Source: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    
    INPUTS
    ------
    df : dataframe
        data for each box, must contain the x1, y1, x2, y2, conf columns with point 1 being top left of the box and point 2 and bottom
        right of box
    thr : float
        IoU threshold used for nms
    
    RETURN
    ------
    df : dataframe
        remaining boxes
    
    """
    df = df.reset_index(drop=True)  # indices must be reset
    dets = df[['x1', 'y1', 'x2', 'y2', 'conf']].to_numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]
        
    return df.loc[keep]


def create_datasets(df, save_dir, split_col='case', test_txt_path='', random_state=None, overlap_col='overlap', n_splits=3, valfrac=0.2, names=None):
    """Create dataset yaml and text files from tiles.csv
    
    INPUTS
    ------
    df : dataframe
        tiles csv file, each row a tile
    save_dir : str
        the directory to create yamls, texts, and csvs directories
    split_col : str (default: 'case')
        column to separate tiles by (i.e. case, roi_im_path, wsi_name, etc.)
    test_txt_path : str (default: '')
        path to a test text path, will be copied
    random_state : int (default: None)
        random state
    overlap_col : str (default: 'overlap')
        split df tiles into overlap and no-overlapping varients. Train uses overlap and val uses no-overlap, make this a bool col
    n_splits : int (default: 3)
        number of train / val splits to create
    valfract : float (default: 0.2)
        fraction of split_col that will be in validation
    names : list (default: None)
        defaults to ['Pre-NFT', 'iNFT'] but can give any list for the labels
    
    """
    np.random.seed(random_state)  # set random state for splitting train / val by split_col column
    
    if names is None:
        names = ['Pre-NFT', 'iNFT']
    nc = len(names)
    
    # create dirs
    yamls_dir = join(save_dir, 'yamls')
    txts_dir = join(save_dir, 'texts')
    csvs_dir = join(save_dir, 'csvs')
    
    makedirs(yamls_dir, exist_ok=True)
    makedirs(txts_dir, exist_ok=True)
    makedirs(csvs_dir, exist_ok=True)
    
    # get a list of "cases", though this may be a different col
    cases = sorted(df[split_col].unique().tolist())
    
    # split into a train and val based on overlap col
    train_df = df[df[overlap_col]]
    val_df = df[~df[overlap_col]]
    
    n_val = int(len(cases) * valfrac)  # number of cases in validation
    
    # copy the test path if given
    if test_txt_path != '':
        test_filename = get_filename(test_txt_path)
        shutil.copy(test_txt_path, join(txts_dir, test_filename + '.txt'))
        shutil.copy(test_txt_path.replace('/texts/', '/csvs/').replace('.txt', '.csv'), join(csvs_dir, test_filename + '.csv'))
        test_filename += '.txt'
    else:
        test_filename = ''
    
    yaml_dict = {'names': names, 'nc': nc, 'path': txts_dir, 'test': test_filename}
    
    # create each split
    for n in range(n_splits):
        np.random.shuffle(cases)
        split_dict = yaml_dict.copy()
        
        val_cases = cases[:n_val]
        train_cases = cases[n_val:]
        
        # create text files for train and val
        split_train = train_df[train_df[split_col].isin(train_cases)]
        split_val = val_df[val_df[split_col].isin(val_cases)]

        train_filename = f'train-n{n+1}.'
        val_filename = f'val-n{n+1}.'

        split_dict['train'] = train_filename + 'txt'
        split_dict['val'] = val_filename + 'txt'

        create_dataset_txt(split_train, join(txts_dir, train_filename + 'txt'))
        create_dataset_txt(split_val, join(txts_dir, val_filename + 'txt'))

        split_train.to_csv(join(csvs_dir, train_filename + 'csv'), index=False)
        split_val.to_csv(join(csvs_dir, val_filename + 'csv'), index=False)

        with open(join(yamls_dir, f'dataset-n{n+1}.yaml'), 'w') as fh:
            yaml.dump(split_dict, fh)
            

def remove_contained_boxes(df, thr):
    """Remove boxes contained in other boxes, or mostly contained. 
    
    INPUTS
    ------
    df : geodataframe
        info about each box
    thr : float
        the threshold of the box that must be contained by fraction of area to be remove
       
    RETURNS
    -------
    df : geodataframe
        the boxes that are left
    
    """
    rm_idx = []
    
    gseries = GeoSeries(df.geometry.tolist(), index=df.index.tolist())  # convert to a geoseries
    
    for i, geo in gseries.items():
        # don't check boxes that have already been removed
        if i not in rm_idx:
            r = df.loc[i]
            
            # remove boxes that don't overlap
            overlapping = df[
                (~df.index.isin(rm_idx + [i])) & ~((r.y2 < df.y1) | (r.y1 > df.y2) | (r.x2 < df.x1) | (r.x1 > df.x2))
            ]
            
            perc_overlap = overlapping.intersection(geo).area / overlapping.area  # percent of object inside the current geo
            
            # filter by the threshold
            overlapping = overlapping[perc_overlap > thr]
            
            rm_idx.extend(overlapping.index.tolist())
            
    return df.drop(index=rm_idx)


def draw_boxes(img: Union[np.array, str], label_fp, 
                        colors: tuple = ((0,0,255), (255,0,0))) -> np.array:
    """Draw YOLO labels on an image.
    
    Args:
        img: Image or image filepath.
        label_fp: File path to labels.
        color: Tuple of RGB colors to use.
        
    Returns:
        Image with boxes drawn.
        
    """
    if isinstance(img, str):
        img = imread(img)
        
    h, w = img.shape[:2]
    
    for box in read_yolo_label(label_fp, im_shape=(w, h), convert=True):
        label, x1, y1, x2, y2 = box[:5].astype(int)
        
        img = cv.rectangle(img, (x1, y1), (x2, y2), colors[label], 3)
        
    return img
