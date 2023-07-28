# Create test datasets.
from pandas import read_csv, DataFrame
from colorama import Fore, Style
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import cv2 as cv
import yaml
from argparse import ArgumentParser
from glob import glob

from os import makedirs, remove
from os.path import join, isfile, isdir

from nft_helpers.utils import print_opt, load_yaml, imread, delete_file
from nft_helpers.box_and_contours import line_to_xys, tile_im_with_boxes
from nft_helpers.roi_utils import read_roi_txt_file
from nft_helpers.yolov5.utils import create_dataset_txt


def parse_opt(cf):
    """CLIs"""
    ##
#     from nft_helpers.utils import dict_to_opt
#     return dict_to_opt({
#         'nproc': 5,
#         'box_thr': 0.5,
#         'im_size': 1280,
#         'stride': 960,
#         'groups': ['test-roi', 'external-roi']
#     })
    ##
    parser = ArgumentParser()
    parser.add_argument('--nproc', type=int, default=5, 
                        help='Parallel processes.')
    parser.add_argument(
        '--box-thr', type=float, default=0.5, 
        help='Threshold of box that must be in tile to keep as label for tile.'
    )
    parser.add_argument('--im-size', type=int, default=1280, 
                        help='Dimension of the square tiles.')
    parser.add_argument('--stride', type=int, default=960, 
                        help='Stride when tiling.')
    parser.add_argument(
        '--groups', type=str, nargs='+', default=['test-roi'],
        help='Dataset / ROI groups to add.'
    )
    
    opt = parser.parse_args()
    
    if opt.nproc > 5:
        raise Exception(
            'Keep CLI nproc less than or equal to 5 to avoid slow down.'
        )
        
    return opt


def tile_roi(r, save_dir, box_thr, im_size, stride, rgb_fill=(114, 114, 114)):
    """Tile ROIs and include labels on the tiles."""
    # read the ROI image
    roi_im = imread(r.fp)

    # create mask for rotated regions (0: not in ROI, 1: in ROI)
    roi_box = line_to_xys(r.roi_corners) - (r.x, r.y)
    roi_mask = np.zeros(roi_im.shape[:2], dtype=np.int8)
    roi_mask = cv.drawContours(roi_mask, [roi_box], -1, 1, cv.FILLED)
    
    # apply ROI mask to the image
    roi_im[np.where(roi_mask == 0)] = rgb_fill

    # get boxes coordinates for ROI
    boxes = read_roi_txt_file(r.roi_labels) if isfile(r.roi_labels) else []

    # Tile the ROIs.
    tile_paths = tile_im_with_boxes(
        roi_im, boxes, save_dir, rotated_mask=roi_mask, box_thr=box_thr, 
        tile_size=im_size, stride=stride, 
        savename=f'{r.wsi_id}-x{r.x}y{r.y}-'
    )

    rows = []
    
    for tile_path in tile_paths:
        im_path, x, y = tile_path
        rows.append([im_path, x, y, im_size, r.fp, r.x, r.y, r.w, r.h])
        
    return rows


def main():
    """Main function."""
    print(Fore.BLUE, Style.BRIGHT, 'Creating additional test datasets.\n', 
          Style.RESET_ALL)
    
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Create directory.
    save_dir = join(cf.datadir, 'datasets/test-datasets')
    makedirs(save_dir, exist_ok=True)
    
    # Filter out ROIs from annotated cohort - these are not for testing.
    rois_df = read_csv('csvs/labeled-rois.csv').replace(np.nan, '', regex=True)
    rois_df = rois_df[rois_df.roi_group.isin(opt.groups)]
    
    # Format the ROI columns.
    rois_df = rois_df.rename(columns={'roi_im_path': 'fp', 'roi_im_left': 'x',
                                   'roi_im_top': 'y'})
    rois_df['w'] = rois_df.roi_im_right - rois_df.x
    rois_df['h'] = rois_df.roi_im_bottom - rois_df.y

    # Save ROI data.
    rois_df.to_csv(join(save_dir, 'rois.csv'), index=False)
    
    print(Fore.CYAN, Style.BRIGHT, '  Tiling ROIs.', Style.RESET_ALL)        
    tiles_df = []
        
    with Pool(opt.nproc) as pool:
        jobs = [
            pool.apply_async(
                func=tile_roi, 
                args=(r, save_dir, opt.box_thr, opt.im_size, opt.stride,)
            )
            for _, r in rois_df.iterrows()
        ]

        for job in tqdm(jobs):
            tiles_df.extend(job.get())
        
    tiles_df = DataFrame(
        data=tiles_df, 
        columns=['fp', 'x', 'y', 'tile_size', 'roi_fp', 'roi_x', 'roi_y', 
                 'roi_w', 'roi_h']
    )
    tiles_df.to_csv(join(save_dir, 'tiles.csv'), index=False)
            
    # YOLO dataset directories.
    yamls_dir = join(save_dir, 'yamls')
    txts_dir = join(save_dir, 'texts')
    csvs_dir = join(save_dir, 'csvs')
    
    # Remove cache files.
    if isdir(txts_dir):
        # remove cache files
        for cache_fp in glob(join(txts_dir, '*.cache')):
            delete_file(cache_fp)
        
    makedirs(yamls_dir, exist_ok=True)
    makedirs(txts_dir, exist_ok=True)
    makedirs(csvs_dir, exist_ok=True)
    
    yaml_dict = {'names': ['Pre-NFT', 'iNFT'], 'nc': 2, 'path': txts_dir, 
                 'train': '', 'val': ''}

    for roi_group in rois_df.roi_group.unique():
        ann_df = tiles_df[tiles_df.roi_fp.isin(
            rois_df[rois_df.roi_group == roi_group].fp
        )]
        
        if not roi_group.startswith('test'):
            save_name = f'test-{roi_group}'
        else:
            save_name = roi_group
            
        ann_df.to_csv(join(csvs_dir, f'{save_name}.csv'), index=False)
        
        yaml_dict['test'] = f'{save_name}.txt'
        create_dataset_txt(ann_df, join(txts_dir, f'{save_name}.txt'))
        
        # save the yaml file
        with open(join(yamls_dir, f'{save_name}.yaml'), 'w') as fh:
            yaml.dump(yaml_dict, fh)
            
    print(Fore.GREEN, Style.BRIGHT, '\nDone!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
