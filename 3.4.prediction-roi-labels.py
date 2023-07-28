# Merge tile prediction labels into ROI prediction labels.
from colorama import Fore, Style
from pandas import read_csv
from tqdm import tqdm
from geopandas import GeoDataFrame
from multiprocessing import Pool
from argparse import ArgumentParser

from os import listdir, makedirs
from os.path import join, isfile, dirname

from nft_helpers.utils import (
    load_yaml, print_opt, im_to_txt_path, get_filename, imread
)
from nft_helpers.yolov5.utils import (
    read_yolo_label, non_max_suppression, remove_contained_boxes
)
from nft_helpers.box_and_contours import corners_to_polygon


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, help='Dataset directory.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument('--tile-size', type=int, default=1280, 
                        help='Tile size, assumed to be the same for all.')
    parser.add_argument('--iou-thr', type=float, default=0.4,
                        help='NMS IoU threshold.')
    parser.add_argument('--contained-thr', type=float, default=0.7,
                        help='Remove contained area threshold.')
    parser.add_argument('--nproc', type=int, default=10, 
                        help='Number of parallel processes to run.')
    
    return parser.parse_args()


def process_roi(roi_fp, roi_df, dataset_dir, tile_size, iou_thr, contained_thr):
    """Process ROI by creating an ROI label for each prediction model using 
    tile labels."""
    # Initiate as None to avoid reading large image if not needed.
    h, w = None, None
    
    # For this ROI we write in its prediction directory.
    pred_dir = dirname(
        im_to_txt_path(roi_df.iloc[0].fp, txt_dir='predictions')
    )

    # Get a list of model predictions (e.g. expert1, expert2, etc.).
    model_dirs = listdir(pred_dir)

    for model in model_dirs:
        # Look for ROI labels for this model.
        roi_dir = join(dataset_dir, f'rois/predictions/{model}')
        makedirs(roi_dir, exist_ok=True)
        
        roi_label_fp = join(roi_dir, get_filename(roi_fp) + '.txt')
        
        if isfile(roi_label_fp):
            continue  # Skip if the label already exists.
            
        # Read the image since it is needed not to get the shape.
        if h is None:
            h, w = imread(roi_fp).shape[:2]
        
        # Compile the tile labels into a GeoDataFrame.
        box_df = []

        for _, r in roi_df.iterrows():
            fn = get_filename(r.fp)

            label_fp = join(pred_dir, model, 'labels', fn + '.txt')

            if isfile(label_fp):
                for box in read_yolo_label(label_fp, im_shape=tile_size, 
                                           convert=True):
                    label, conf = int(box[0]), box[5]
                    x1, y1, x2, y2 = box[1:5].astype(int)
                    x1, y1, x2, y2 = x1 + r.x, y1 + r.y, x2 + r.x, y2 + r.y

                    box_df.append([
                        label, x1, y1, x2, y2, conf, 
                        corners_to_polygon(x1, y1, x2, y2)
                    ])
                    
        if len(box_df):
            box_df = GeoDataFrame(
                box_df, 
                columns=['label', 'x1', 'y1', 'x2', 'y2', 'conf', 'geometry']
            )
            
            # Merge overlapping boxes (caused by overlapping tiles).
            box_df = non_max_suppression(box_df, iou_thr)
            box_df = remove_contained_boxes(box_df, contained_thr)

            # Format the boxes to YOLO format and save.
            labels = ''

            for _, r in box_df.iterrows():
                xc, yc = (r.x1 + r.x2) / 2 / w, (r.y1 + r.y2) / 2 / h
                bw, bh = (r.x2 - r.x1) / w, (r.y2 - r.y1) / h
                labels += f'{r.label} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f} ' + \
                          f'{r.conf:.4f}\n'

            with open(roi_label_fp, 'w') as fh:
                fh.write(labels.strip())
    

def main():
    print(Fore.BLUE, Style.BRIGHT, 'Merge tile prediction labels into ROI ' 
          'prediction labels.\n', Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Tile metadata.
    tile_df = read_csv(join(opt.dataset_dir, 'tiles.csv'))
    
    # Parallel process to go through ROIs.
    with Pool(opt.nproc) as pool:
        jobs = [
            pool.apply_async(
                func=process_roi, 
                args=(
                    roi_fp, tile_df[tile_df.roi_fp == roi_fp], opt.dataset_dir,
                    opt.tile_size, opt.iou_thr, opt.contained_thr,
                )
            ) 
            for roi_fp in tile_df.roi_fp.unique()
        ]
        
        for job in tqdm(jobs):
            job.get()  # return nothing.
            
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)
    

if __name__ == '__main__':
    main()
