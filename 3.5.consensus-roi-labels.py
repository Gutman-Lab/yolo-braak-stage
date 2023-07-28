# Apply consensus voting on the model ROI predictions to get consensus labels.
from colorama import Fore, Style
from pandas import read_csv, concat, DataFrame
from typing import List
from geopandas import GeoDataFrame
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser

from os import listdir, makedirs
from os.path import join, isfile

from nft_helpers.utils import load_yaml, print_opt, get_filename, imread
from nft_helpers.yolov5.utils import (
    read_yolo_label, non_max_suppression, remove_contained_boxes
)
from nft_helpers.box_and_contours import corners_to_polygon


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, help='Dataset directory.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling'),
    )
    parser.add_argument('--iou-thr', type=float, default=0.4, 
                        help='IoU threshold.')
    parser.add_argument('--contained-thr', type=float, default=0.7,
                        help='Area threshold for removing contained boxes.')
    parser.add_argument('--nproc', type=int, default=20,
                        help='Parallel processes to use.')
    
    return parser.parse_args()


def consensus_labels(
    fps: List[str], n: int, iou_thr: float, contained_thr: float, 
    save_fp: str, img_fp: str
) -> DataFrame:
    """Combine a set of labels using voting strategy.
    
    Args:
        fps: Label filepaths.
        n: Number of labels that must be in agreement.
        iou_thr: IoU threshold when matching boxes.
        contained_thr: Remove contained box threshold.
        save_fp: Save consensus label to this filepath.
        img_fp: Filepath to image.
    
    Returns:
        Consensus data on boxes.
    
    """
    h, w = imread(img_fp).shape[:2]
    
    # Compile the boxes into a GeoDataFrame.
    boxes_df = []
    
    for i, fp in enumerate(fps):
        if isfile(fp):
            for box in read_yolo_label(fp, im_shape=(w, h), convert=True):
                label, conf = int(box[0]), box[5]
                x1, y1, x2, y2 = box[1:5].astype(int)
                
                boxes_df.append([
                    label, x1, y1, x2, y2, conf, i,
                    corners_to_polygon(x1, y1, x2, y2)
                ])
                
    boxes_df = GeoDataFrame(
        boxes_df, 
        columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'model', 'geometry']
    )
    
    # Return the empty dataframe if it is empty.
    if not len(boxes_df):
        return boxes_df
        
    # List of unique models - sorted.
    models = sorted(list(boxes_df.model.unique()))
    
    # Find matches between models.
    i_checked = []
    consensus_df = []
    
    N = len(boxes_df)
            
    for i, r in boxes_df.iterrows():
        # skip box if it already was checked
        if i in i_checked:
            continue
        i_checked.append(i)
        
        # Check this box against other labels boxes.
        subset = boxes_df[boxes_df.model != r.model]
        subset = subset[~subset.index.isin(i_checked)]
        
        # Subset to boxes that overlap sufficient with the current box.
        intersection = subset.intersection(r.geometry)
        union = subset.union(r.geometry)
        subset['iou'] = intersection.area.divide(union.area).tolist()
        
        subset = subset[
            subset.iou > iou_thr
        ].sort_values(by='iou', ascending=False)
        
        # Choose the box for each model / label that has the largest IoU.
        drop_idx = []
        
        for model, count in subset.model.value_counts().items():
            if count > 1:
                # Multiple boxes from this model overlap.
                model_subset = subset[subset.model == model]
                
                # Take the box with the highest IoU.
                drop_idx.extend(model_subset.index.tolist()[1:])

        # Drop indices.
        subset = subset.drop(index=drop_idx)  # drop indices
        
        # These boxes will not be tracked.
        i_checked.extend(subset.index.tolist())
        
        # Adding the current box.
        subset = concat([DataFrame(data=[r]), subset], ignore_index=True)
        
        # Get the labels as a space separated string, use -1 for background.
        str_labels = ''
        subset_models = subset.model.tolist()
        for model in models:
            if model in subset_models:
                str_labels += f'{subset[subset.model == model].iloc[0].label:.0f} '
            else:
                str_labels += '-1 '
        
        # Get the consensus label.
        label_counts = subset.label.value_counts().to_dict()
        
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        max_count = max(counts)
        
        if max_count >= n:
            n_agreement = max_count
            
            # Find the label that has the max count - go from highest label.
            for lb in sorted(labels, reverse=True):
                if label_counts[lb] == max_count:
                    consensus_label = lb
                    consensus_subset = subset[subset.label == lb]
                    consensus_conf = consensus_subset.conf.mean()
                    
                    # Get coordinates for one of the boxes.
                    rr = consensus_subset.sort_values(
                        by='conf', ascending=False
                    ).iloc[0]
                    
                    x1, y1, x2, y2 = rr.x1, rr.y1, rr.x2, rr.y2
                    break
        else:
            consensus_label = -1
            n_agreement = len(models) - len(subset)
            consensus_conf = -1  # no confidence when the label is the background
            rr = subset.sort_values(by='conf', ascending=False).iloc[0]
            x1, y1, x2, y2 = rr.x1, rr.y1, rr.x2, rr.y2
            
        # Add this to consensus.
        consensus_df.append([
            consensus_label, x1, y1, x2, y2, str_labels, n_agreement, 
            consensus_conf, corners_to_polygon(x1, y1, x2, y2)
        ])
        
        if len(i_checked) == N:
            break
        
                
    consensus_df = GeoDataFrame(
        consensus_df, 
        columns=['label', 'x1', 'y1', 'x2', 'y2', 'labels', 'n_agreement', 
                 'conf', 'geometry']
    )
    
    # Remove redundant boxes.
    consensus_boxes = consensus_df[consensus_df.label != -1] 
    
    if len(consensus_boxes):
        consensus_boxes = non_max_suppression(consensus_boxes, iou_thr)
        consensus_boxes = remove_contained_boxes(consensus_boxes, contained_thr)

        # Save the consensus boxes to an label file.
        labels = ''

        for _, r in consensus_boxes.iterrows():
            x1, y1, x2, y2  = r.x1, r.y1, r.x2, r.y2

            xc, yc = (x2 + x1) / 2 / w, (y2 + y1) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h

            labels += f'{r.label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n'

        with open(save_fp, 'w') as fh:
            fh.write(labels.strip())
                
    # return in normalized format.
    consensus_df.x1 /= w
    consensus_df.y1 /= h
    consensus_df.x2 /= w
    consensus_df.y2 /= h
    consensus_df['fp'] = [img_fp] * len(consensus_df)
    
    del consensus_df['geometry']
    return consensus_df
    

def main():
    """Main."""
    print(Fore.BLUE, Style.BRIGHT, 'Creating consensus ROI labels.\n', 
          Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    pred_dir = join(opt.dataset_dir, 'rois/predictions')
    models = sorted(listdir(pred_dir))
    
    N = len(models)
    
    roi_df = read_csv(join(opt.dataset_dir, 'rois.csv'))
    
    for n in range(1, N+1):
        # Check if boxes csv file for this dir exists.
        save_dir = join(opt.dataset_dir, f'rois/consensus/{n}')
        consensus_boxes_fp = join(save_dir, 'consensus-boxes.csv')
        
        if isfile(consensus_boxes_fp):
            print(Fore.YELLOW, Style.BRIGHT, f'Consensus labels for n={n} '
                  'exists, skipping.\n', Style.RESET_ALL)
            continue
        
        print(Fore.CYAN, Style.BRIGHT, f'Consensus of {n} model agreement:',
              Style.RESET_ALL)
        
        makedirs(save_dir, exist_ok=True)
        
        with Pool(opt.nproc) as pool:
            jobs = [
                pool.apply_async(
                    func=consensus_labels, 
                    args=(
                        [join(pred_dir, model, get_filename(r.fp) + '.txt') 
                         for model in models],
                        n,
                        opt.iou_thr,
                        opt.contained_thr,
                        join(save_dir, get_filename(r.fp) + '.txt'),
                        r.fp,
                    )
                ) 
                for _, r in roi_df.iterrows()
            ] 
            
            consensus_boxes_df = []
            
            for job in tqdm(jobs):
                consensus_boxes_df.append(job.get())
                
        _ = concat(
            consensus_boxes_df, ignore_index=True
        ).to_csv(consensus_boxes_fp, index=False)
        
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)
    

if __name__ == '__main__':
    main()
