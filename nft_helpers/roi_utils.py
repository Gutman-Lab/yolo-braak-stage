# Working with ROIs and their labels
# Functions:
# - create_roi_labels
# - select_consensus_labels
# - read_roi_txt_file
# - roi_labels_from_tiles
# - roi_labels_from_tiles_wrapper
# - combine_roi_labels
# - combine_rois_labels
# - update_tile_labels
# - tile_roi
# - tile_rois
# - tile_img
# - label_df_to_txt
# - label_array_to_df
# - merge_predictions
# - match_labels
import numpy as np
import multiprocessing as mp
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
from tqdm import tqdm
from pandas import concat, DataFrame
from collections import Counter
import cv2 as cv
from glob import glob

from .utils import save_to_txt, get_filename, imread, imwrite, im_to_txt_path
from .yolov5.utils import read_yolo_label, non_max_suppression, remove_contained_boxes
from .box_and_contours import line_to_xys, xys_to_line, corners_to_polygon

from os import makedirs, listdir, remove
from os.path import join, isfile, dirname


def create_roi_labels(df, label_map=None, shift=None, box_col='box_coords', label_col='label', save_filepath=None):
    """Given a dataframe of box annotation data, create label text file. All rows are assumed to be from the same ROI / image.
    
    INPUT
    -----
    df : dataframe
        each row an annotation in the same region of an image
    label_map : dict (default is None)
        map labels to int values, if None then all unique labels are alphabetized and given int label by that order
    shift : tuple (default is None)
        shift all box coordinates by this tuple of length two (x shift, y shift)
    box_col : str (default='box_coords')
        column in df with the coordinates of the annotations (boxes)
    label_col : str (default='label')
        column in df with the annotation label
    save_filepath : str (default is None)
        save labels to text file
        
    RETURN
    ------
    labels : str
        each line of the string contains the label, x1, y1, x2, y2 coordinates where point 1 is the top left corner of the annotation
        object and point 2 is the bottom right corner
    
    """
    labels = ''
    
    if save_filepath is not None and not save_filepath.endswith('.txt'):
        raise Exception('save filepath parameter should be a text file extension')
    
    # if label map is None, map them to int alphabetically
    if label_map is None:
        unique_labels = sorted(df[label_col].unique().tolist())
        
        label_map = {}
        for i, lb in enumerate(unique_labels):
            label_map[lb] = i
    else:
        # remove any annotations that are not in the map
        df = df[df[label_col].isin(list(label_map.keys()))]
        
    for _, r in df.iterrows():
        # convert the box coordinates and shift them if needed
        box_coords = line_to_xys(r[box_col])
        
        if shift is not None:
            box_coords += shift
            
        box_coords = xys_to_line(box_coords)
        
        labels += f'{label_map[r[label_col]]} {box_coords}\n'

    if len(labels):
        labels = labels[:-1]
        
        # save labels to text file
        if save_filepath is not None:
            save_to_txt(save_filepath, labels)
        
    return labels


def select_consensus_labels(df, strategy='majority'):
    """Return a list of labels for each row in the df parameter, by selecting the label based on consensus strategy provided
    
    INPUTS
    ------
    df : dataframe
        label dataframe, each row a unique object and each column an annotator, note that this works only with labels of 
        int form from 0, 1, ... , N
    strategy : list or str (default in 'majority')
        strategy for creating consensus labels. When passing in an int it will do an n-consensus strategy, where the label will be
        the highest positive int label that has at least n-number of occurrences, if no positive label reaches the n-consensus then
        the label is 0. For 'majority' consensus the label is given by the highest int label with the most occurences
    
    RETURN
    ------
    labels : dict
        keys are the strategy and values is a list of consensus labels for that strategy
    
    """
    # number of label annotators
    N = len(df.columns)
    
    if isinstance(strategy, str):
        strategy = [strategy]
        
    for strat in strategy:
        if isinstance(strat, int):
            if (strat < 1 or strat > N):
                raise Exception("consensus N majority strategy can't be greater then the number of annotators or less than 1")
        elif strat != 'majority':
            raise Exception('not an acceptable strategy')
            
    # return a list of labels for each strategy
    labels = {strat: [] for strat in strategy}
    
    for _, r in df.iterrows():
        r = r.to_numpy()
        
        # loop through each strategy
        for strat in strategy:
            if isinstance(strat, int):
                # for n-consensus take out the zero (background label)
                strat_r = r[r != 0].copy()

                # return both the unique values and the number of their occurences
                values, counts = np.unique(strat_r, return_counts=True)

                # get the highest occurence count
                max_count = max(counts)
                
                # the label is 0 if the max_count is not at lest of strat value
                if max_count < strat:
                    label = 0
                else:
                    # get all values with highest occurence count
                    max_values = values[counts == max_count]

                    # choose the highest value 
                    label = max(max_values)
                
                # add this label to appropriate list
                labels[strat].append(label)
            else:
                # majority strategy - for this keep background label
                values, counts = np.unique(r, return_counts=True)
                
                # get the highest occurence count
                max_count = max(counts)
                
                # get all values with highest occurence count
                max_values = values[counts == max_count]
                
                # choose the highest value 
                label = max(max_values)
                
                # add this label to appropriate list
                labels[strat].append(label)
                
    return labels


def read_roi_txt_file(filepath, xshift=0, yshift=0):
    """Read the ROI label boxes text file into an array
    
    INPUTS
    ------
    filepath : str
        path to text file
    xshift : int (default: 0)
        shift coordinates by this value (add this value)
    yshift : int (default: 0)
        shift coordinates by this value (add this value)
        
    RETURN
    ------
    boxes : array
        Nx5 array, where N is the number of boxes and columns are [label, x1, y1, x2, y2] with point 1 being top left and point 2 being
        bottom right corner of box. If there was a confidence value in the text file then each box also has a conf value at the end
    
    """
    boxes = []
    
    with open(filepath, 'r') as fh:
        for line in fh.readlines():
            line = [float(l) for l in line.strip().split(' ')]
            
            label, x1, y1, x2, y2 = [int(l) for l in line[:5]]
            
            # shift coordinates
            x1 += xshift
            x2 += xshift
            y1 += yshift
            y2 += yshift
            
            if len(line) > 5:
                boxes.append([label, x1, y1, x2, y2, line[5]])
            else:
                boxes.append([label, x1, y1, x2, y2])
            
    return np.array(boxes)


def roi_labels_from_tiles(df, tile_label_dir, save_path=None, nms_iou_thr=0.4, contains_thr=0.7):
    """Using a tiles.csv file, create ROI labels by applying NMS & contain-box remove algorithms.
    
    INPUTS
    ------
    df : dataframe
        tiles.csv dataframe
    tile_label_dir : str
        directory with the label text files for the tile images, note all images should be located in the same directory
    save_path : str (default: None)
        file path to save the label text file for the roi, if not given then it is not saved
    nms_iou_thr : float
        IoU threshold used in NMS
    contains_thr : float
        area threshold for contains algorithm, if a box is mostly contained in another box (by area) then it is removed
        
    RETURN
    ------
    label_df : dataframe
        boxes dataframe, each row a prediction
    
    """
    # compile the tile label coordinates into a GeoDataFrame
    label_df = []
    
    for _, r in df.iterrows():
        # get image size
        im_shape = (r.im_right - r.im_left, r.im_bottom - r.im_top)
        
        # read the label path if it exists
        label_path = join(tile_label_dir, get_filename(r.impath) + '.txt')
        
        if isfile(label_path):
            # read the text file
            xshift, yshift = r.im_left - r.roi_im_left, r.im_top - r.roi_im_top
            boxes = read_yolo_label(label_path, im_shape=im_shape, shift=(-xshift, -yshift), convert=True)
            
            for box in boxes:
                label, x1, y1, x2, y2 = box[:5].astype(int).tolist()
                geometry = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1,  y2)])
                
                label_df.append([label, x1, y1, x2, y2, box[5], geometry])
                
    label_df = GeoDataFrame(data=label_df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry'])
    
    # apply filters to remove redundant boxes - (1) remove via NMS & (2) apply remove contained boxes
    if len(label_df):
        label_df = non_max_suppression(label_df, nms_iou_thr)
        label_df = remove_contained_boxes(label_df, contains_thr)
    
    if save_path is not None:
        if len(label_df):
            lines = ''

            for _, r in label_df.iterrows():
                lines += f'{r.label} {r.x1} {r.y1} {r.x2} {r.y2} {r.conf}\n'

            with open(save_path, 'w') as fh:
                fh.write(lines.strip())
    
    # save the image
    return label_df


def roi_labels_from_tiles_wrapper(df, tiles_pred_dir, rois_pred_dir, nproc=20, nms_iou_thr=0.4, contains_thr=0.7):
    """Wrapper around the function roi_labels_from_tiles, which runs the function on all model & ROI pairs using parallel processing
    for faster completion.
    
    INPUTS
    ------
    df : dataframe
        the tiles dataframe
    tiles_pred_dir : str
        directory with model predictions of the tiles (<pred_dir>/model-name/...txt)
    rois_pred_dir : str
        directory to same predictions (<save_dir>/model-name/...txt)
    nproc : int
        number of processes to use for parallel processing
    nms_iou_thr : float
        IoU threshold used in NMS
    contains_thr : float
        area threshold for contains algorithm, if a box is mostly contained in another box (by area) then it is removed
        
    RETURN
    ------
    pairs : list
        list of 3-sized tuples - (roi_im_path, model tile label dir, roi label save filepath)
        
    """
    roi_im_paths = df.roi_im_path.unique()  # list of unique ROI image paths
    model_names = listdir(tiles_pred_dir)  # model dir names
    
    # create a list of model name / roi image path pairs
    pairs = []
    
    for model_name in model_names:
        roi_pred_dir = join(rois_pred_dir, model_name)
        makedirs(roi_pred_dir, exist_ok=True)
        
        for roi_im_path in roi_im_paths:
            # skip 
            save_path = join(roi_pred_dir, get_filename(roi_im_path) + '.txt')
            
            if not isfile(save_path):
                pairs.append([roi_im_path, join(tiles_pred_dir, model_name, 'labels'), save_path])
    
    if len(pairs):
        # run in parallel
        pool = mp.Pool(nproc)
        jobs = [pool.apply_async(func=roi_labels_from_tiles, args=(
            df[df.roi_im_path == pair[0]],
            pair[1],
            pair[2],
            nms_iou_thr,
            contains_thr,
        )) for pair in pairs]
        pool.close()
        
        for job in tqdm(jobs):
            _ = job.get()
            
    return pairs


def combine_roi_labels(roi_im_path: str, label_dirs: list, im_save_dir: str, iou_thr: float = 0.4, im_size: int = 500, 
                       min_agreement: int = 2, rgb_fill: tuple = (114, 114, 114), backups: dict = None, 
                       contained_thr: float = 0.7, nms_iou_thr: float = 0.4) -> GeoDataFrame:
    """Combine ROI predictions from multiple models into one using consensus voting.
    
    Args:
        roi_im_path: File path to ROI image.
        label_dirs: List of directories to check for prediction text files.
        im_save_dir: Directory to save images of the each consensus box.
        iou_thr: IoU threshold, used in determining if boxes point to the same object.
        im_size: Dimension of square images saved around each box.
        min_agreement: Minimum number of predictions that must point to the same object to give a consensus label. This 
            is class agnostic, meaning that this minimum number must be reached by a combination of all the positive 
            labels not the same label. A -1 is used for a box that does not meet the minimum agreement.
        rgb_fill: When saving box images near the edge, this RGB color is used to pad the image.
        backups: The keys are the box image paths, used to modify the box edges, label, and qc status for human-in-loop
            work. 
        contained_thr: Threshold of box contained in another before it is removed.
        nms_iou_thr: After merging boxes, clean up the boxes by removing overlapping boxes with NMS.
        
    Returns:
        consensus_boxes: Each row contains data about a consensus box.
    
    """
    n_models = len(label_dirs)  # number of models
    
    if backups is None:
        backups = {}
    
    makedirs(im_save_dir, exist_ok=True)
 
    # read the image and pad all around it
    half_size = int(im_size / 2)
    
    roi_im = None
    roi_label_filename = get_filename(roi_im_path) + '.txt'
    
    # compile all the boxes from ROIs into a single geodataframe to pass into parallel processing
    boxes_df = []
    
    for label_dir in label_dirs:
        # read roi label text file if it exists
        roi_label_path = join(label_dir, roi_label_filename)

        if isfile(roi_label_path):
            # get an array of the boxes in format: (label, x1, y1, x2, y2, conf)
            boxes = read_roi_txt_file(roi_label_path)

            for box in boxes:
                label, x1, y1, x2, y2 = box[:5].astype(int)
                conf = box[5]

                # add this each box as a new row
                boxes_df.append([label, x1, y1, x2, y2, conf, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), label_dir])
                
    boxes_df = GeoDataFrame(data=boxes_df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry', 'label_dir'])
    
    # find overlapping boxes from different models and add the box with a consensus vote, use -1 for a background class
    i_checked = []
    
    consensus_boxes = []
            
    for i, r in boxes_df.iterrows():
        # skip box if it already was checked
        if i in i_checked:
            continue
    
        i_checked.append(i)  # track this box as checked
        
        # filter boxes to check against: only of different label directory and not already checked
        subset = boxes_df[boxes_df.label_dir != r.label_dir]
        subset = subset[~subset.index.isin(i_checked)]
        
        # use IoU to determine if two boxes point to the same object
        intersection = subset.intersection(r.geometry)
        union = subset.union(r.geometry)
        ious = intersection.area.divide(union.area)
        
        # filter out low IoU boxes
        subset = subset[ious > iou_thr].sort_values(by='conf', ascending=False)
        
        # there should only be one unique label per label dir
        drop_idx = []
        for k, v in subset.label_dir.value_counts().items():
            if v > 1:
                label_dir_subset = subset[subset.label_dir == k]
                
                # add indices to drop
                drop_idx.extend(label_dir_subset.index.tolist()[1:])
              
        subset = subset.drop(index=drop_idx)  # drop indices
        i_checked.extend(subset.index.tolist())  # stop tracking all boxes that matched
        
        # add the current row as subset 
        subset = concat([DataFrame(data=[r]), subset], ignore_index=True)
        
        str_labels = subset['label'].tolist()
        str_labels = ' '.join(str(lb) for lb in str_labels + [-1] * (n_models - len(str_labels)))
        
        # use class-agnostic agreement
        if len(subset) < min_agreement:
            label = -1
            n_agreement = n_models - len(subset)
            conf = 1 - subset['conf'].mean()  # set it equal to 1 - the average of the prediction confidences of the positive predictions
        else:
            # get the label with the highest number of predictions, in case of ties take the label with highest int value
            label_counts = Counter(subset['label'].tolist())  # dict-type mapping the label to the number of occurences
            
            # get the highest number of occurences
            n_agreement = max(label_counts.values())
            
            # get a list of labels matching the highest count 
            label = []
            
            for k, v in label_counts.items():
                if v == n_agreement:
                    label.append(k)
                    
            # choose the highest value label
            label = sorted(label)[-1]
            subset = subset[subset['label'] == label]
            conf = subset['conf'].mean()
                        
        # randomly sample one of the boxes in the subset
        sample_r = subset.sample(random_state=r.x1).iloc[0]
        x1, y1, x2, y2 = sample_r['x1'], sample_r['y1'], sample_r['x2'], sample_r['y2']

        # calculate the coordinates for getting the image from ROI
        xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2) 
        left, top, right, bottom = xc-half_size, yc-half_size, xc+half_size, yc+half_size
        
        # save the image or skip it if it already exists
        im_path = join(im_save_dir, f'{get_filename(roi_im_path)}_imLeft-{left}_imTop-{top}_imRight-{right}_imBottom-{bottom}.png')
        
#         if not isfile(im_path):
#             if roi_im is None:
#                 roi_im = imread(roi_im_path)
#                 roi_im = cv.copyMakeBorder(roi_im, half_size, half_size, half_size, half_size, cv.BORDER_CONSTANT, value=rgb_fill)
            
#             # grab the image from ROI
#             im = roi_im[top+half_size:bottom+half_size, left+half_size:right+half_size].copy()
#             imwrite(im_path, im)
            
        # if there is a previous version - update the values
        old_label, old_x1, old_y1, old_x2, old_y2 = label, x1, y1, x2, y2
        
        if im_path in backups:
            old_r = backups[im_path]
            label, x1, y1, x2, y2 = old_r['label'], old_r['x1'], old_r['y1'], old_r['x2'], old_r['y2']
            qc = 'yes'
        else:
            qc = 'no'

        # add this consensus box
        consensus_boxes.append([
            roi_im_path, im_path, label, conf, im_size, x1, y1, x2, y2, left, top, right, bottom, n_agreement, 
            str_labels, qc, (x2 - x1) * (y2 - y1), old_label, old_x1, old_y1, old_x2, old_y2, 
            corners_to_polygon(x1, y1, x2, y2)
        ])
    
    df = GeoDataFrame(
        data=consensus_boxes, 
        columns=['roi_im_path', 'im_path', 'label', 'conf', 'im_size', 'x1', 'y1', 'x2', 'y2', 'im_left', 'im_top', 
                 'im_right', 'im_bottom', 'n_agreement', 'labels', 'qc', 'area', 'orig_label', 'orig_x1', 'orig_y1', 
                 'orig_x2', 'orig_y2', 'geometry']
    )
    
    pos_df = df[df.label >= 0]
    neg_df = df[df.label < 0]

    # filter out similar boxes
    if nms_iou_thr is not None:
        pos_df = non_max_suppression(pos_df, nms_iou_thr)
    if contained_thr is not None:
        pos_df = remove_contained_boxes(pos_df, contained_thr)
        
    return concat([pos_df, neg_df], ignore_index=True)


def combine_rois_labels(df: DataFrame, pred_dir: str, im_save_dir: str, iou_thr: float = 0.4, nproc: int = 20, 
                        im_size: int = 500, min_agreement: int = 2, rgb_fill: tuple = (114, 114, 114), 
                        backup_df: DataFrame = None, nms_iou_thr: float = 0.4, contained_thr: float = 0.7) -> GeoDataFrame:
    """Given a dataframe of ROI data and a set of directories of labels for those ROIs, create a consensus label for each ROI. 
    Combination of all ROI labels for a single ROI are combined into a consensus label file.
    
    Args:
        df: ROI dataframe.
        pred_dir: Directory of prediction text files matching the ROI image file paths.
        im_save_dir: Directory to save images around each box.
        iou_thr: IoU threshold for matching boxes from predictions.
        nproc: Number of parallel processes to use.
        im_size: Dimension of square image to save around each box.
        min_agreement: Minimum number of predictions that must agree to give consensus label.
        rgb_fill: Color to pad images when near edge.
        backup_df: Use to provide a cleaned up value of the box label and coordinates.
        nms_iou_thr: NMS IoU threshold applied after consensus boxes.
        contained_thr: Contained IoU threshold applied after consensus boxes.
    
    Returns:
        Dataframe of the consensus boxes with their labels and coordinates. Label -1 is for background boxes or boxes
        did not have enough agreement.
    
    """
    backups = {}
    
    if backup_df is not None:
        for _, r in backup_df.iterrows():
            if r.qc == 'yes':
                backups[r.im_path] = r
    
    makedirs(im_save_dir, exist_ok=True)
    
    # list all the prediction dirs
    label_dirs =  glob(join(pred_dir, '*'))
    
    # run in parallel
    pool = mp.Pool(nproc)
    jobs = [pool.apply_async(func=combine_roi_labels, args=(
        r.roi_im_path,
        label_dirs,
        im_save_dir,
        iou_thr,
        im_size,
        min_agreement,
        rgb_fill,
        backups,
        contained_thr,
        nms_iou_thr,
    )) for _, r in df.iterrows()]
    pool.close()
    
    boxes_df = [job.get() for job in tqdm(jobs)]
    
    return concat(boxes_df, ignore_index=True)


def _update_tile_labels(roi_im_path, tiles, annotations, rois_label_dir=None, area_thr=0.5):
    """Parallel processing wrapper on the update_tile_labels function"""
    # subset to tiles and annotations for this ROI
    roi_tiles = tiles[tiles.roi_im_path == roi_im_path]
    roi_annotations = annotations[annotations.roi_im_path == roi_im_path]

    # get an array of the coordinates and area
    roi_annotations = roi_annotations[['label', 'x1', 'y1', 'x2', 'y2', 'area', 'area', 'area']].to_numpy()

    # loop through each tile
    for _, tile in roi_tiles.iterrows():
        # shift the annotation coordinates to be relative to this tiles location
        tile_annotations = roi_annotations.copy()

        xshift, yshift = tile.im_left - tile.roi_im_left, tile.im_top - tile.roi_im_top
        tile_annotations[:, 1] -= xshift
        tile_annotations[:, 3] -= xshift
        tile_annotations[:, 2] -= yshift
        tile_annotations[:, 4] -= yshift

        # clip the annotation coords by this tile
        w, h = tile.im_right - tile.im_left, tile.im_bottom - tile.im_top

        tile_annotations[:, 1] = np.clip(tile_annotations[:, 1], 0, w)
        tile_annotations[:, 3] = np.clip(tile_annotations[:, 3], 0, w)
        tile_annotations[:, 2] = np.clip(tile_annotations[:, 2], 0, h)
        tile_annotations[:, 4] = np.clip(tile_annotations[:, 4], 0, h) 

        # calculate the new area
        tile_annotations[:, 6] = (tile_annotations[:, 3]-tile_annotations[:, 1]) * (tile_annotations[:, 4]-tile_annotations[:, 2])

        # filter the annotations by threshold
        tile_annotations[:, 7] = tile_annotations[:, 6] / tile_annotations[:, 5]
        tile_annotations = tile_annotations[tile_annotations[:, 7] >= area_thr, :]

        # add the text file
        label_path = im_to_txt_path(tile.impath)

        if len(tile_annotations):
            # create the label dir if it does not exist
            makedirs(dirname(label_path), exist_ok=True)

            lines = ''

            # save the file
            for ann in tile_annotations:
                label, x1, y1, x2, y2 = ann[:5]

                # calculate the center and box width and height
                xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h

                lines += f'{label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n'
            
            with open(label_path, 'w') as fh:
                fh.write(lines)
        else:
            if isfile(label_path):
                # remove this file
                remove(label_path)

        if rois_label_dir is not None:
            # create the label text file for the ROI as well
            roi_label_path = join(rois_label_dir, get_filename(roi_im_path) + '.txt')

            if len(roi_annotations):
                # get the width and height of the ROI
                roi_w, roi_h = tile.roi_im_right - tile.roi_im_left, tile.roi_im_bottom - tile.roi_im_top

                lines = ''

                # save the file
                for ann in roi_annotations:
                    label, x1, y1, x2, y2 = ann[:5]

                    # calculate the center and box width and height
                    xc, yc = (x1 + x2) / 2 / roi_w, (y1 + y2) / 2 / roi_h
                    bw, bh = (x2 - x1) / roi_w, (y2 - y1) / roi_h

                    lines += f'{label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n'

                with open(roi_label_path, 'w') as fh:
                    fh.write(lines)
            else:
                if isfile(roi_label_path):
                    # remove this ROI since it does not have any labels
                    remove(roi_label_path)

    
def update_tile_labels(tiles, annotations, rois_label_dir=None, area_thr=0.5, nproc=20):
    """Update the tile labels using annotation.csv file. Note, that this will overwrite current files and may delete some text files
    if the tile no longer has labels.
    
    INPUTS
    ------
    tiles : dataframe
        the tiles dataframe, each row a tile from the ROI
    annotations : dataframe
        the annotations dataframe, each row a object box in the ROI
    rois_label_dir : str (default: None)
        if not None, then roi label text files will be saved as well
    area_thr : float (default: 0.5)
        frac of original area of box that must be in tile to be included as a label
    nproc : int (default: 20)
        number of processes for parallel processing
    
    """
    if rois_label_dir is not None:
        makedirs(rois_label_dir, exist_ok=True)
        
    # loop by unique ROI image
    pool = mp.Pool(nproc)
    jobs = [pool.apply_async(func=_update_tile_labels, args=(roi_im_path, tiles, annotations, rois_label_dir, area_thr,)) 
            for roi_im_path in tiles.roi_im_path.unique()]
    pool.close()
    
    for job in tqdm(jobs):
        _ = job.get()
        
        
def tile_roi(r, save_dir, fill=(114, 114, 114), im_size=1280, stride=None, tile_frac_thr=0.2):
    """Tile an ROI
    
    INPUT
    -----
    r : pandas.Series
        data of the ROI
    save_dir : str
        directory to save images to
    fill : tuple (default: (114, 114, 114))
        RGB value to use when padding ROI image
    im_size : int (default: 1280)
        tile image size (square tiles)
    stride : int (default: None)
        stride value when tiling, if None it is by default set to im_size (no overlap between tiles)
    tile_fract_thr: float (default: 0.2)
        minimum fraction of tile (by area) that must be ROI region to include tile
        
    RETURN
    ------
    : dataframe
        each row containing information about a tile image saved
    
    """
    makedirs(save_dir, exist_ok=True)
    
    if stride is None:
        stride = im_size  # default behavior - no overlap
        
    # read the roi image
    roi_im = imread(r.roi_im_path)
    
    # create a binary mask of the ROI - since ROIs might have been rotated
    roi_h, roi_w = roi_im.shape[:2]
    roi_mask = cv.drawContours(
        np.zeros((roi_h, roi_w), dtype=np.uint8), [line_to_xys(r.roi_corners, shift=(r.roi_im_left, r.roi_im_top))], -1, 1, cv.FILLED
    )

    # mask the image outside of the ROI
    roi_im[np.where(roi_mask == 0)] = fill
    
    # pad the image and mask
    roi_im = cv.copyMakeBorder(roi_im, 0, im_size, 0, im_size, cv.BORDER_CONSTANT, value=fill)
    roi_mask = cv.copyMakeBorder(roi_mask, 0, im_size, 0, im_size, cv.BORDER_CONSTANT, value=0)
    
    # create x, y coordinates
    xys = list(((x, y) for x in range(0, roi_w, stride) for y in range(0, roi_h, stride)))
    overlap = True if stride < im_size else False
        
    img_df = []  # track all image data
    
    tile_area = im_size**2
    
    # loop through each tile coordinate
    for xy in xys:
        # grab tile
        x, y = xy
        im = roi_im[y:y+im_size, x:x+im_size].copy()
        mask = roi_mask[y:y+im_size, x:x+im_size]

        # skip if tile is not sufficiently in ROI
        if np.count_nonzero(mask) / tile_area  < tile_frac_thr:
            continue

        # calculate the coordinates of the top left corner of tile relative to the entire WSI
        x1, y1 = x + int(r.roi_im_left), y + int(r.roi_im_top)
        x2, y2 = x1 + im_size, y1 + im_size

        # save image and label text file
        clean_name = ' '.join(get_filename(r.wsi_name).split()).strip()
        im_savepath = join(save_dir, f'{clean_name}_id-{r.wsi_id}_left-{x1}_top-{y1}_right-{x2}_bottom-{y2}.png')

        # save image if it does not exist
        if not isfile(im_savepath):
            imwrite(im_savepath, im)
        
        im_r = r.copy()  # transfer the data from ROI and add some new data
        im_r['impath'] = im_savepath
        im_r['im_left'] = x1
        im_r['im_top'] = y1
        im_r['im_right'] = x2
        im_r['im_bottom'] = y2
        im_r['stride'] = stride
        im_r['overlap'] = overlap
        
        img_df.append(im_r)
        
    return DataFrame(img_df)


def tile_rois(df, save_dir, fill=(114, 114, 114), im_size=1280, stride=None, tile_frac_thr=0.2, nproc=20):
    """Tile a set of ROIs.
    
    INPUT
    -----
    df : dataframe
        ROIs dataframe, each row is an ROI data
    save_dir : str
        directory to save images to
    fill : tuple (default: (114, 114, 114))
        RGB value to use when padding ROI image
    im_size : int (default: 1280)
        tile image size (square tiles)
    stride : int (default: None)
        stride value when tiling, if None it is by default set to im_size (no overlap between tiles)
    tile_fract_thr: float (default: 0.2)
        minimum fraction of tile (by area) that must be ROI region to include tile
    nproc : int (default: 20)
        number of processes to use 
        
    RETURN
    ------
    : dataframe
        each row containing information about a tile image saved
    
    """
    makedirs(save_dir, exist_ok=True)
    
    if stride is None:
        stride = im_size  # default behavior - no overlap
    
    # process each image / WSI in parallel
    pool = mp.Pool(nproc)
    jobs = [pool.apply_async(
        func=tile_roi, args=(r, save_dir, fill, im_size, stride, tile_frac_thr)) for _, r in df.iterrows()
    ]
    pool.close()
    
    tiles_df = []
    
    for job in tqdm(jobs):
        tiles_df.append(job.get())
        
    # return as a single dataframe
    return concat(tiles_df, ignore_index=True)


def create_roi_mask(w: int, h: int, corners: list, value: int = 255, **kwargs: dict) -> np.array:
    """Create a binary mask given width & height and the region corners. 
    
    Args:
        w: Width of mask.
        h: Height of mask.
        corners: List of (x, y) tuples for the corners of the box.
        value: Value to use for the positive part of the mask.
        kwargs: Key-word arguments.
        
    Return:
        Binary mask that highlights the positive part of the mask.
    
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv.drawContours(mask, [corners], -1, value, cv.FILLED)
    
    return mask


def tile_img(img: np.array, mask: np.array = None, ts: int = 1280, stride: int = 960, area_thr: float = 0.2, 
             fill: (int, int, int) = (114, 114, 114), save_dir: str = None, prepend_name: str = 'tile', 
             **kwargs) -> (list, list, list):
    """Tile an image into smaller images.
    
    Args:
        img: Image to tile into smaller images.
        mask: Binary mask that defines which parts of the image are important. Tiles not in positive part of binary
            mask are not included. The amount of positive region that must be in tile to include is defined by the 
            area_thr parameter.
        ts: Size of the tile, square tiles.
        stride: Stride step when tiling. If this value is smaller than tile size then the tiles will be overlapping.
        area_thr: Fraction of pixels in tile that must be in mask to be included.
        fill: RGB color to fill areas that are not included in mask. 
        save_dir: Directory to save tile images to.
        prepend_name: When saving images use this string to prepend the filenames when saving tile images (i.e.
            prepend_name_x1-##_y1-##_x2-##_y2-##.png)
        **kwargs: Key-word arguments.
    
    Returns:
        List of tile images.
        List of top left corner of tile, in x, y coordinates.
        List of filepaths, empty if save_dir was not passed.
    
    """
    h, w = img.shape[:2]
        
    img = cv.copyMakeBorder(img, 0, stride, 0, stride, cv.BORDER_CONSTANT, value=fill)
    
    # if mask is passed then fill out non-roi regions
    if mask is not None:
        img[np.where(mask == 0)] = fill
        
        # pad on the edges to handle tiling going over the edge
        mask = cv.copyMakeBorder(mask, 0, stride, 0, stride, cv.BORDER_CONSTANT, value=0)
        
    count_thr = (ts * ts) * area_thr
    
    tiles = []
    xys = []
    fps = []
    
    for xy in list(((x, y) for x in range(0, w, stride) for y in range(0, h, stride))):
        # grab tile
        x, y = xy
        tile = img[y:y+ts, x:x+ts]

        # skip the tile if most of it is not in the mask
        if np.count_nonzero(mask[y:y+ts, x:x+ts]) < count_thr:
            continue
            
        # save the tile
        if save_dir is not None:
            fp = join(save_dir, f'{prepend_name}_x1-{x}_y1-{y}_x2-{x+ts}_y2-{y+ts}.png')
            imwrite(fp, tile)
            fps.append(fp)
        
        tiles.append(tile)
        xys.append(xy)
    
    return tiles, xys, fps


def label_df_to_txt(df: DataFrame, ts: int = None) -> str:
    """Convert a dataframe of predictions to a text file.
    
    Args:
        df: Predictions in each row, containing the label, conf, x1, y1, x2, y2.
        ts: If given, the output is formatted in yolo format, x-center, y-center, width, height normalized to the 
            size of images (ts).
    
    Returns:
        The formatted predictions boxes in string format.
    
    """
    output = ''
    
    for _, r in df.iterrows():
        if ts is None:
            output += f'{int(r.label)} {int(r.x1)} {int(r.y1)} {int(r.x2)} {int(r.y2)} {r.conf:0.6f}\n'
        else:
            x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
            
            xc, yc = (x1 + x2) / (2 * ts), (y1 + y2) / (2 * ts)
            w, h = (x2 - x1) / ts, (y2 - y2) / ts
            
            output += f'{int(r.label)} {xc:0.6f} {yc:0.6f} {w:0.6f} {h:0.6f} {r.conf:0.6f}\n'
            
    return output.strip()


def label_array_to_df(arr: np.array) -> GeoDataFrame:
    """Convert a label array ([[label, x1, y1, x2, y2, conf (optional)], ...]) to a GeoDataFrame.
    
    Args:
        arr: Array of labels.
        
    Returns:
        Dataframe with columns: label, x1, y1, x2, y2, and geometry.
        
    """
    df = []
    
    conf_flag = False
    
    for r in arr:
        if len(r) > 5:
            conf = r[5]
            conf_flag = True
            
        label, x1, y1, x2, y2 = r[:5]
        
        label = int(label)
        
        if conf_flag:
            df.append([label, x1, y1, x2, y2, conf, corners_to_polygon(x1, y1, x2, y2)])
        else:
            df.append([label, x1, y1, x2, y2, corners_to_polygon(x1, y1, x2, y2)])
            
    if conf_flag:
        return GeoDataFrame(df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry'])
    else:
        return GeoDataFrame(df, columns=['label', 'x1', 'y1', 'x2', 'y2', 'geometry'])

    
def merge_predictions(fps: list, xys: list, ts: int, iou_thr: float = 0.4, contained_thr: float = 0.7, 
                      label_dir: str = None) -> GeoDataFrame:
    """This function is to be used in tangent (after) the tile_img() function. Given a list of filepaths for prediction
    text files and their x, y coordinate location in a larger image, read in the boxes and merge them. The results are
    the prediction boxes for the larger image that the boxes belong to.
    
    Args:
        fps: File paths to images..
        xys: (x, y) corresponding to the file paths given. These denote the top left point of the predictions of each
            file. The boxes will be shifted to all be relative to a larger image.
        ts: Size of image for the prediction files. These are assumed to be square images.
        iou_thr: NMS IoU threshold.
        contained_thr: Fraction contained for removing contained boxes.
        label_dir: Directory with labels for the image filepaths. If not given the label file is assumed to be in a 
            directory /label/ below the image fp.
    
    Returns:
        Merge predictions in a dataframe format, each row is a prediction box.
    
    """
    # merge the predictions for the tiles into one
    labels = []  # to use the merge functions, format labels into a GeoDataFrame
    
    # loop through image file paths
    for fp, xy in zip(fps, xys):
        if label_dir is None:
            label_fp = im_to_txt_path(fn)
        else:
            label_fp = join(label_dir, get_filename(fp) + '.txt')
        
        if isfile(label_fp):
            x, y = xy
            for box in read_yolo_label(label_fp, im_shape=ts, shift=(-x, -y), convert=True):
                label, x1, y1, x2, y2, conf = box
                
                labels.append([label, x1, y1, x2, y2, conf, corners_to_polygon(x1, y1, x2, y2)])
                
    labels = GeoDataFrame(labels, columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry'])
    
    # merge boxes (tiling creates overlapping areas)
    if len(labels):
        labels = non_max_suppression(labels, iou_thr)
        labels = remove_contained_boxes(labels, contained_thr)
        
    return labels


def match_labels(true: np.array, preds: np.array, iou_thr: float = 0.4, bg_label: int = None) -> GeoDataFrame:
    """Match true labels against prediction labels. The format of the array inputs is: [label, x1, y1, x2, y2, conf]
    with the confidence being optional and not used.
    
    Args:
        true: Ground truth boxes.
        preds: Prediction boxes.
        iou_thr: IoU threshold to determine if boxes overlap.
        bg_label: The background label. If not given, then this will be inferred from the labels.
    
    Returns:
        Matched labels as a dataframe with true and preds column for the labels.
    
    """
    matches = []
    
    # convert to dataframes
    true = label_array_to_df(true)
    preds = label_array_to_df(preds)
    
    if bg_label is None:
        bg_label = np.max(true['label'].tolist() + preds['label'].tolist()) + 1
        
    # loop through each ground truth label
    i_matched = []
    
    for _, r in true.iterrows():
        # compare this ground truth box against all predictions that have not been matched
        not_matched = preds[~preds.index.isin(i_matched)]
        
        # calculate the intersection over union and threshold this
        intersections = not_matched.geometry.intersection(r.geometry).area
        unions = not_matched.geometry.union(r.geometry).area
        ious = intersections / unions
        ious = ious[ious > iou_thr]
        
        if len(ious):
            # get the highest IoU index
            ious = ious.sort_values(ascending=False)
            
            # get the index and value
            i = ious.index[0]
            iou = ious.iloc[0]
            
            mr = preds.loc[i]
            
            # add this to match to not check against against
            i_matched.append(i)
            
            # add to matches
            matches.append([
                r.label, mr.label, iou, r.x1, r.y1, r.x2, r.y2, mr.x1, mr.y1, mr.x2, mr.y2, mr.conf
            ])
        else:
            # there was not match so add it as such
            matches.append([
                r.label, bg_label, 0, r.x1, r.y1, r.x2, r.y2, -1, -1, -1, -1, -1
            ])
            
    # add predictions that did not match boxes
    for _, r in preds.drop(index=i_matched).iterrows():
        matches.append([
            bg_label, r.label, 0, -1, -1, -1, -1, r.x1, r.y1, r.x2, r.y2, r.conf
        ])
            
    return DataFrame(
        matches, columns=['true', 'pred', 'iou', 'x1', 'y1', 'x2', 'y2', 'px1', 'py1', 'px2', 'py2', 'conf']
    )
