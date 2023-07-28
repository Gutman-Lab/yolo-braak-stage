# Update the tile labels using one of the consensus ROI labels.
from colorama import Fore, Style
from argparse import ArgumentParser
from glob import glob
from pandas import read_csv, DataFrame
from geopandas import GeoDataFrame
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from typing import List

from os import remove
from os.path import join, isfile

from nft_helpers.utils import (
    load_yaml, print_opt, get_filename, delete_file
)
from nft_helpers import update_roi_tile_labels


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument('--consensus-n', type=int, required=True,
                        help='Consensus directory (e.g. 1, 2, 3, etc.).')
    parser.add_argument(
        '--dataset-dir', type=str, help='Dataset directory.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument(
        '--area-thr', type=float, default=0.5,
        help='Fraction of object that must be in tile to include.'
    )
    parser.add_argument('--nproc', type=int, default=10, 
                        help='Number of parallel processes.')
    
    return parser.parse_args()

                
def main():
    """Main function"""
    print(Fore.BLUE, Style.BRIGHT, 'Updating tile labels.\n', Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Delete cache files.
    for cache_fp in glob(join(opt.dataset_dir, 'texts/*.cache')):
        delete_file(cache_fp)
            
    # For each ROI - get the shape of it (w, h)
    roi_df = read_csv(join(opt.dataset_dir, 'rois.csv'))
    roi_shapes = {r.fp: (r.w, r.h) for _, r in roi_df.iterrows()}
            
    # Get tile metadata.
    tile_df = read_csv(join(opt.dataset_dir, 'tiles.csv'))
    
    pred_dir = join(opt.dataset_dir, f'rois/consensus/{opt.consensus_n}')
    
    with Pool(opt.nproc) as pool:
        jobs = [
            pool.apply_async(
                func=update_roi_tile_labels, 
                args=(
                    tile_df[tile_df.roi_fp == roi_fp], 
                    join(pred_dir, get_filename(roi_fp) + '.txt'),
                    roi_shapes[roi_fp],
                    opt.area_thr,
                )
            ) 
            for roi_fp in tile_df.roi_fp.unique()]
        
        roi_df = [job.get() for job in tqdm(jobs)]
        
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)
        

if __name__ == '__main__':
    main()
