# General functions / classes
# functions:
# - imwrite
# - imread
# - load_yaml
# - print_opt
# - load_json
# - save_json
# - Timer
# - dict_to_opt
# - save_to_txt
# - im_to_txt_path
# - get_filename
# - create_cases_df
# - create_wsis_df
# - object_metrics
# - read_any_labels
# - draw_boxes
# - get_label_fp
import cv2 as cv
import yaml
from collections import namedtuple
from colorama import Fore, Style
import json
from time import perf_counter
from pandas import DataFrame
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import re
from typing import Tuple

from os import remove
from os.path import splitext, isfile, join


def imwrite(filepath, im, grayscale=False):
    """Write an image to file using opencv
    
    INPUTS
    ------
    filepath : str
        path to save image to
    im : array-like
        image array
    grayscale : bool
        if True then save image in grayscale, else save as RGB
    
    """
    if grayscale:
        cv.imwrite(filepath, im)
    else:
        cv.imwrite(filepath, cv.cvtColor(im, cv.COLOR_RGB2BGR))
        
        
def imread(impath, grayscale=False):
    """
    Read image from file using opencv, returns as RGB or grayscale.
    
    INPUTS
    ------
    impath : str
        path to image
    grayscale : bool
        if True return the image as grayscale
    
    RETURN
    ------
    numpy array, image
    
    """
    if grayscale:
        return cv.imread(impath, 0)
    else:
        return cv.cvtColor(cv.imread(impath), cv.COLOR_BGR2RGB)

    
def load_yaml(filepath='conf.yaml'):
    """
    Load a yaml file and return the content as a namedtuple.
    
    INPUTS
    ------
    filepath : str
        path to yaml file (default: 'conf.yaml')
    
    RETURN
    ------
    namedtuple
        yaml contents
    
    """
    with open(filepath, 'r') as f:
        cf =  yaml.safe_load(f)

    try:
        return dict_to_opt(cf)
    except ValueError as err:
        print(Fore.RED + f'all keys in \"{filepath}\" must be valid, make sure there is no dashes in the keys' + Style.RESET_ALL)
        raise

        
def print_opt(opt):
    """
    Print out the keys / value of named tuple.
    
    """
    # convert opt to dict
    try:
        opt = opt._asdict()
    except AttributeError:
        opt = vars(opt)
    
    printstr = ''
    for k, v in opt.items():
        printstr += f'{k}={v}, '
        
    if len(printstr):
        printstr = printstr[:-2]
        print('   opt:\t', end='')
        print(printstr)
        print()
    else:
        print('  opt is empty\n')

        
def load_json(path):
    """Load a json file.
    
    """
    with open(path, 'r') as fh:
        json_content = json.load(fh)
    return json_content


def save_json(path, data):
    """Save a variable to json file
    
    param: path is the path to save json file
    param: data, the variable to jsonify
    
    """
    with open(path, 'w') as fh:
        json.dump(data, fh)
        
        
class Timer():
    def __init__(self, fp):
        """Timer class."""
        self.fp = fp
        
        # start the document
        if not isfile(self.fp):
            with open(self.fp, 'w') as fh:
                fh.write('')
                
        self.running = False
        self.time = None
        
    def start(self):
        """Start the timer."""
        if not self.running:
            self.running = True
            self.time = perf_counter()
            
    def stop(self):
        """Stop the timer."""
        if self.running:
            self.running = False
            
            delta = perf_counter() - self.time
            
            with open(self.fp, 'a') as fh:
                fh.write(f'{delta:.0f}\n')
            
            self.time = None
        else:
            print(Fore.YELLOW, Style.BRIGHT, 
                  'No start timer, run .start() first.', Style.RESET_ALL)

            
def dict_to_opt(opt_dict):
    """Convert a dictionary into an opt style variable, with dot format to access fields"""
    # replace any keys with dashes to underscores
    valid_dict = {}
    
    for k, v in opt_dict.items():
        if isinstance(k, str):
            valid_dict[k.replace('-', '_')] = v
        else:
            valid_dict[k] = v
    
    return namedtuple("ObjectName", valid_dict.keys())(*valid_dict.values())


def save_to_txt(save_filepath, string):
    """Save a string to a text file"""
    with open(save_filepath, 'w') as fh:
        fh.writelines(string)
        
        
def im_to_txt_path(impath: str, txt_dir: str = 'labels'):
    """Replace the last occurance of /images/ to /labels/ in the given image path and change extension to .txt"""
    splits = impath.rsplit('/images/', 1)
    return splitext(f'/{txt_dir}/'.join(splits))[0] + '.txt'


def get_filename(path: str, prune_ext: bool = True, replaces_spaces: bool = False) -> str:
    """Get filename from a file path.
    
    Args:
        path: Filepath to get file name from.
        prune_ext: If True then the filename is returned without extension.
        replaces_spaces: If True then space characters are replaced with underscores, multiple sequential spaces are 
            replaced by a single underscore.
        
    Returns:
        Filename.
        
    """
    filename = path.split('/')[-1]
    
    if prune_ext:
        filename = splitext(filename)[0]
    
    if replaces_spaces:
        filename = re.sub(' +', '_', filename)
    
    return filename


def create_cases_df(annotations, cases_cols=None):
    """Create cases dataframe
    
    INPUTS
    ------
    annotations : dict
        keys are item ids, values are the items
    cases_cols : list (default: None)
        list of keys in meta to add to the dataframe as columns, if None then default is a list of columns
    
    RETURNS
    -------
    : dataframe
        metadata dataframe for the cases
    parent_id_map : dict
        map of the item name to ids
        
    """
    # loop through the inference cohort to get case information for the annotation cohort
    cases_df = []
    
    parent_id_map = {}  # map WSI name to the id of WSI / item in the inference cohort
    
    if cases_cols is None:
        cases_cols = [
            'Braak_stage', 'ABC', 'age_at_death', 'Clinical_Dx', 'Other_NP_Dx_AD', 'Other_NP_Dx_LBD', 'Other_NP_Dx_Misc_1', 'Other_NP_Dx_Misc_2',
            'Other_NP_Dx_Misc_3', 'Other_NP_Dx_Misc_4', 'Other_NP_Dx_TAU', 'Other_NP_Dx_TDP', 'Other_NP_Dx_Vascular', 'Primary_NP_Dx', 'race', 
            'region', 'sex', 'Thal'
        ]
    
    cases_added = []  # only add one entry per case
    
    for wsi_id, item in annotations.items():
        wsi_name = item['name']
        meta = item['meta'] if 'meta' in item else {}
        
        if 'Braak_stage' not in meta and 'Braak Stage' in meta:
            meta['Braak_stage'] = meta['Braak Stage']
            
        if 'Clinical_Dx' not in meta and 'Clinical Dx' in meta:
            meta['Clinical_Dx'] = meta['Clinical Dx']
            
        if 'Primary_NP_Dx' not in meta and 'Primary NP Dx' in meta:
            meta['Primary_NP_Dx'] = meta['Primary NP Dx']
            
        if 'age_at_death' not in meta and 'age at death' in meta:
            meta['age_at_death'] = meta['age at death']
        
        # map this WSI mage to its id
        parent_id_map[wsi_name] = wsi_id
        
        if meta['case'] not in cases_added:
            # add this case
            cases_added.append(meta['case'])
            row = [meta['case']]
            
            for case_col in cases_cols:
                row.append(meta[case_col] if case_col in meta else '')
            
            cases_df.append(row)
           
    # return as a dataframe, also return the map of the image name to parent id
    return DataFrame(data=cases_df, columns=['case'] + cases_cols), parent_id_map


def create_wsis_df(annotations, parent_id_map=None, meta_keys=None):
    """Create wsis dataframe
    
    INPUTS
    ------
    annotations : dict
        keys are item ids, values are the items. The items should have keys: name, meta, and scan_mag, cohort
    parent_id_map : dict (default: None)
        pass a parent id dict, mapping names to item ids. If None then it will be empty strings
    meta_keys : list (default: None)
        list of meta keys, default is None 
        
    RETURN
    ------
    : dataframe
        WSI dataframe data
    
    """
    if parent_id_map is None:
        parent_id_map = {}
        
    if meta_keys is None:
        meta_keys = []
        
    wsis_df = []
    
    # loop through all items in the annotated cohort
    for item_id, item in annotations.items():
        # add the info of this WSI
        wsi_name = item['name']
        meta = item['meta']
        
        parent_id = parent_id_map[wsi_name] if wsi_name in parent_id_map else ''
        
        row = [item['name'], item_id, parent_id] + [item[k] for k in ('scan_mag', 'cohort')]
        
        meta = item['meta'] if 'meta' in item else {}
        
        for k in meta_keys:
            row.append(meta[k] if k in meta else '')
        
        wsis_df.append(row)
        
    # return as a Dataframe
    return DataFrame(data=wsis_df, columns=['wsi_name', 'wsi_id', 'parent_id', 'scan_mag', 'cohort'] + meta_keys)


def object_metrics(
    matches: DataFrame, labels: list = None, bg_label: int = None) -> (np.array, float, float, np.array, dict
    ):
    """Given an dataframe of true and pred matches, output from match_labels(), calculate metrics. Mainly report
    the per-class F1 score, micro & macro F1 scores, average IoU score for each class when correctly predicting, 
    and the confusion matrix.
    
    Args:
        matches: Contains columns: true, pred, iou, x1, y1, x2, y2, px1, py1, px2, py2, conf.
    """
    # calculate the F1 scores
    true = matches['true'].tolist()
    preds = matches['pred'].tolist()
    
    if labels is None:
        labels = list(range(np.max(true + preds) + 1))
        
    if bg_label is None:
        bg_label = []
    else:
        bg_label = [bg_label]
        
    per_class_f1 = f1_score(true, preds, average=None, labels=labels)
    micro_f1 = f1_score(true, preds, average='micro', labels=labels)
    macro_f1 = f1_score(true, preds, average='macro', labels=labels)
    cm = confusion_matrix(true, preds, normalize='true', labels=labels + bg_label).T
    
    ious = {}
    
    for _, r in matches.iterrows():
        if r['true'] in labels and r['pred'] in labels and r['true'] == r['pred']:
            if r['true'] not in ious:
                ious[r['true']] = []
            ious[r['true']].append(r.iou)
            
    for k in list(ious.keys()):
        ious[k] = np.mean(ious[k])
        
    return per_class_f1, micro_f1, macro_f1, cm, ious


def delete_file(fp: str):
    """Delete a file and make sure it is deleted before continuing.
    
    Args:
        fp: Filepath.
        
    """
    while isfile(fp):
        try:
            remove(fp)
        except OSError:
            pass

        
def read_any_labels(fp: str, im_shape: Tuple[int, int]) -> np.array:
    """Read any label in either format and return it in non-YOLO format.
    
    Args:
        fp: Label filepath.
        im_shape: width, height of image.
        
    Returns:
        Boxes in non-yolo format (label, x1, y1, x2, y2, conf optional).
    
    """
    boxes = []
    
    if isfile(fp):
        with open(fp, 'r') as fh:
            for line in fh.readlines():
                line = [float(l) for l in line.strip().split(' ')]

                if max(line[1:5]) > 1:
                    # Already in non-yolo format
                    boxes.append(line)
                else:
                    xc, yc, bw, bh = line[1:5]
                    
                    bw2 = bw / 2
                    bh2 = bh / 2
                    
                    x1, y1, x2, y2 = xc - bw2, yc - bh2, xc + bw2, yc + bh2
                    line[1] = int(x1 * im_shape[0])
                    line[2] = int(y1 * im_shape[1])
                    line[3] = int(x2 * im_shape[0])
                    line[4] = int(y2 * im_shape[1])
                    
                    boxes.append(line)
                    
    return np.array(boxes)


def draw_boxes(img: np.array, boxes: np.array, lw: int = 10) -> np.array:
    """Draw NFT boxes on an image.
    
    Args:
        img: Image.
        boxes: Boxes in non-yolo format (label, x1, y1, x2, y2, conf optional).
        lw: Line with when drawing boxes.
        
    Returns:
        Image with drawn boxes.
        
    """
    img = img.copy()
    
    for box in boxes:
        label, x1, y1, x2, y2 = box[:5].astype(int)
        
        img = cv.rectangle(img, (x1, y1), (x2, y2), 
                           (255, 0, 0) if label else (0, 0, 255), lw)
        
    return img


def get_label_fp(fp: str, label_dir: str = None) -> str:
    """Get the text label file for an image.
    
    Args:
        fp: Filepath to image.
        label_dir: If passed then this is the directory wher the label file is,
            otherwise it is grabbed parallel to the filepath in labels dir.
            
    Returns:
        Label text filepath.
    
    """
    if label_dir is None:
        return im_to_txt_path(fp)
    else:
        return join(label_dir, get_filename(fp) + '.txt')
