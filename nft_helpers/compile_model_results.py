# Compile the model results from directory.
from pandas import read_csv, DataFrame
from glob import glob
import re
from os.path import join, dirname, isfile
import numpy as np
import json

ROI_METRIC_KEYS = [
    'F1 score (Pre-NFT)', 'F1 score (iNFT)', 'micro F1 score', 'macro F1 score',
    'weighted F1 score', 'Precision (Pre-NFT)', 'Precision (iNFT)', 
    'Recall (Pre-NFT)', 'Recall (iNFT)', 'TP (Pre-NFT)', 'TP (iNFT)',
    'FP (Pre-NFT)', 'FP (iNFT)', 'FN (Pre-NFT)', 'FN (iNFT)', 'cm', 'F1 score',
    'Precision', 'Recall', 'TP', 'FP', 'FN'    
]


def compile_model_results(src_dir: str) -> DataFrame:
    """Get the results from validating on datasets using trained models.
    
    Args:
        src_dir: Directory with model directories.
        
    Returns:
        The results of validation of the models on various datasets. Results are reported for each class individually
    """
    results = []
    
    # list every validation terminal output file
    for fp in glob(join(src_dir, '**/validate/*/terminal_output.csv'), recursive=True):
        df = read_csv(fp)
        
        # Get the model name and keep track of the train / val split
        # i.e. expert1-n1, expert1-n2 are both model expert1 but two splits
        parts = fp.split('/')
        model = parts[-4]  # The model name
        
        # Model ends with n# for the split. Remove this
        if re.search('-n\d$', model):
            model, split = model[:-3], model[-1]
        else:
            model, split = model, 1
        
        # Remove the dataset from the end.
        model = model[:-8] if re.search('-dataset$', model) else model
        
        # Get the dataset the validation was run on.
        dataset = parts[-2]
        
        # get the best weights path
        weights = join('/'.join(parts[:-3]), 'weights/best.pt')
        
        # Read the ROI metrics - or if missing add a blank value.
        metric_fp = join(dirname(fp), 'roi-metrics.json')
        
        if isfile(metric_fp):
            with open(metric_fp, 'r') as fh:
                roi_metrics = json.load(fh)
        else:
            roi_metrics = {}
        
        # add to results for this model
        for _, r in df.iterrows():
            if r['class'] == 'Pre-NFT':
                f1 = roi_metrics.get('Pre-NFT F1')
            elif r['class'] == 'iNFT':
                f1 = roi_metrics.get('iNFT F1')
            else:
                f1 = np.nan
                
            if r['class'] == 'all':
                macro = roi_metrics.get('macro-F1')
                micro = roi_metrics.get('micro-F1')
            else:
                macro, micro = np.nan, np.nan
            
            row = [
                model, split, weights, dataset, r['class'], r['images'], 
                r['labels'], r['P'], r['R'], r['mAP50'], r['mAP50-95'], 
                dirname(fp)
            ]
            
            for k in ROI_METRIC_KEYS:
                row.append(roi_metrics[k] if k in roi_metrics else '')
                    
            results.append(row)
            
    return DataFrame(
        results, 
        columns=[
            'model', 'split', 'weights', 'dataset', 'label', 'images', 'labels',
            'P', 'R', 'mAP50', 'mAP50-95', 'src'
        ] + ROI_METRIC_KEYS
    )
