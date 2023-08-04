# Script adds ROI validation for trained models.
from nft_helpers import compile_model_results, match_boxes
from nft_helpers.yolov5 import create_roi_preds
from nft_helpers.yolov5.utils import read_yolo_label
from nft_helpers.roi_utils import read_roi_txt_file
from nft_helpers.utils import load_yaml, im_to_txt_path, get_filename

from os.path import join, dirname, isdir, isfile
from shutil import rmtree
from pandas import read_csv, concat
import yaml
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score
)
from tqdm import tqdm
import json
from multiprocessing import Pool

np.set_printoptions(suppress=True)


def process_task_dir(r):
    """Process a task directory from an Series containing model results for that
    dataset.
    
    """
    # Create prediction ROI labels.
    roi_pred_dir = join(r.src, 'roi-labels')
    
    # Skip if it has already been run.
    if isdir(roi_pred_dir):
        return
    
#     if r.dataset == 'val' and r.model.startswith('iteration'):
    if r.dataset == 'val' and \
       (r.model.startswith('iteration') or 'models-consensus' in r.model):
        # Model-assisted-labeling true ROI labels change every iteration so 
        # calculating ROI metrics after the fact is not possible for the val
        # dataset.
        return
    
    # Get the filepath to the tile csv for this validate dataset.
    with open(join(dirname(dirname(r.src)), 'opt.yaml'), 'r') as fh:
        with open(yaml.safe_load(fh)['data'], 'r') as fh:
            data = yaml.safe_load(fh)

    txt_fp = join(data['path'], data[r.dataset])
    tiles = read_csv(join(
        dirname(dirname(txt_fp)), 'csvs', get_filename(txt_fp) + '.csv'
    ))

    # Create prediction ROI labels.
    roi_pred_dir = join(r.src, 'roi-labels')

    create_roi_preds(tiles, join(r.src, 'labels'), roi_pred_dir)

    # Match boxes between ground truth and pred ROI labels.
    matches = []

    for roi_fp in tiles.roi_fp.unique():
        # Get single tile to get ROI info.
        roi = tiles[tiles.roi_fp == roi_fp].iloc[0]
        
        # Read the true ROI labels.
        label_fp = im_to_txt_path(roi_fp)
        
        if r.dataset == 'val' and 'models-consensus' in r.model:
            # The true ROI labels are in specific directories.
            fn = get_filename(label_fp, prune_ext=False)
            label_fp = join(
                dirname(dirname(label_fp)), 
                'consensus', 
                r.model[0], 
                fn
            )

        if isfile(label_fp):
            # ROI labels are saved in two types of formats: non-YOLO format for 
            # ROIs used in annotator train / val datasets and the test-roi & 
            # test-external datasets and YOLO format for model-assisted-labeling
            # datasets.
            if r.model.startswith(('expert', 'novice')) or \
               r.dataset in ('test-roi', 'test-external-roi'):                            
                # Format: label, x1, y1, x2, y2 (not normalized).
                true = read_roi_txt_file(label_fp)
            else:
                # Format: label, xc, yc, bw, bh (normalized to ROI image shape).
                # Read and convert to non-yolo format.
                true = read_yolo_label(
                    label_fp, im_shape=(roi.roi_w, roi.roi_h), convert=True
                )
        else:
            # No true labels, no NFTs in ROI image.
            true = []

        # Read the predictions.
        pred_fp = join(roi_pred_dir, get_filename(roi_fp) + '.txt')

        if isfile(pred_fp):
            # Read this in the same format as true.
            pred = read_yolo_label(
                pred_fp, im_shape=(roi.roi_w, roi.roi_h), convert=True
            )
        else:
            # No predictions on this ROI.
            pred = []

        roi_matches = match_boxes(true, pred, labels=[0, 1])
        roi_matches['roi_fp'] = [roi_fp] * len(roi_matches)

        matches.append(roi_matches)

    # Combine the matches into a single dataframe.
    matches = concat(matches, ignore_index=True)
    matches.to_csv(join(r.src, 'roi-matches.csv'), index=False)    

    # Calculate metrics.
    roi_metrics = {}

    true = matches['true'].tolist()
    pred = matches['pred'].tolist()

    # F1 score for each class (e.g. analize as binary for this class)
    f1s = f1_score(true, pred, labels=[0, 1], average=None)
    roi_metrics['F1 score (Pre-NFT)'] = f1s[0]
    roi_metrics['F1 score (iNFT)'] = f1s[1]

    # Calculate avearge F1 scores.
    roi_metrics['micro F1 score'] = f1_score(true, pred, labels=[0, 1],
                                             average='micro')
    roi_metrics['macro F1 score'] = f1_score(true, pred, labels=[0, 1],
                                             average='macro')
    roi_metrics['weighted F1 score'] = f1_score(true, pred, labels=[0, 1],
                                             average='weighted')

    # Calculate the precision and recall for each class.
    precisions = precision_score(true, pred, labels=[0, 1], average=None)
    recalls = recall_score(true, pred, labels=[0, 1], average=None)
    roi_metrics['Precision (Pre-NFT)'] = precisions[0]
    roi_metrics['Precision (iNFT)'] = precisions[1]
    roi_metrics['Recall (Pre-NFT)'] = recalls[0]
    roi_metrics['Recall (iNFT)'] = recalls[1]

    # Get the confusion matrix (rows are true, columns are pred)
    cm = confusion_matrix(true, pred, labels=[-1, 0, 1])

    # Calculate the TP, FP, and FN for each class.
    roi_metrics['TP (Pre-NFT)'] = int(cm[1, 1])
    roi_metrics['TP (iNFT)'] = int(cm[2, 2])

    roi_metrics['FP (Pre-NFT)'] = int(cm[0, 1] + cm[2, 1])
    roi_metrics['FP (iNFT)'] = int(cm[0, 2] + cm[1, 2])

    roi_metrics['FN (Pre-NFT)'] = int(cm[1, 0] + cm[1, 2])
    roi_metrics['FN (iNFT)'] = int(cm[2, 0] + cm[2, 1])

    roi_metrics['cm'] = cm.astype(int).tolist()

    # Calculate the metrics when combining the two classes into one.
    matches.true = matches.true.replace({0: 1}).replace({-1: 0})
    matches.pred = matches.pred.replace({0: 1}).replace({-1: 0})

    # Calculate the confusion matrix.
    true = matches.true
    pred = matches.pred

    cm = confusion_matrix(true, pred, labels=[0, 1])
    roi_metrics['F1 score'] = f1_score(true, pred, labels=[0, 1])
    roi_metrics['Precision'] = precision_score(true, pred, labels=[0, 1])
    roi_metrics['Recall'] = recall_score(true, pred, labels=[0, 1])

    roi_metrics['TP'] = int(cm[1, 1])
    roi_metrics['FP'] = int(cm[0, 1])
    roi_metrics['FN'] = int(cm[1, 0])

    # Save the ROI metrics into pickle file.
    with open(join(r.src, 'roi-metrics.json'), 'w') as fh:
        json.dump(roi_metrics, fh)

    return
    

def main():
    """Main function."""
    cf = load_yaml()
    
    # Get validate results on the models.
    model_results = compile_model_results(join(cf.datadir, 'models'))
    model_results = model_results[model_results.label == 'all']

    with Pool(20) as pool:
        jobs = [
            pool.apply_async(
                func=process_task_dir,
                args=(r,)
            )
            for _, r in model_results.iterrows()
        ]
        
        # Run the jobs, does not return anything but saves results.
        for job in tqdm(jobs):
            _ = job.get()
        

if __name__ == '__main__':
    main()
