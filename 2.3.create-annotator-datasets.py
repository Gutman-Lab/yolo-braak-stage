# Tile ROIs with labels and create YOLO dataset files.
from colorama import Fore, Style
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
import cv2 as cv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import yaml
from shutil import rmtree, copyfile
from sklearn.model_selection import train_test_split
from glob import glob

from nft_helpers.utils import print_opt, load_yaml, imread, get_filename
from nft_helpers.box_and_contours import line_to_xys, tile_im_with_boxes
from nft_helpers.roi_utils import read_roi_txt_file
from nft_helpers.yolov5.utils import create_dataset_txt

from os import makedirs
from os.path import join, isfile, isdir, dirname

# Ignore annotators that only annotated for inter-annotator analysis.
VALID_ANNOTATORS = ['expert1', 'expert2', 'expert3', 'expert4', 'expert5', 
                    'novice1', 'novice2', 'novice3']


def parse_opt():
    ##
#     from nft_helpers.utils import dict_to_opt
#     return dict_to_opt({
#         'im_size': 1280,
#         'box_thr': 0.5,
#         'stride': 960,
#         'save_dir': 'annotator-datasets',
#         'rgb_fill': (114, 114, 114),
#         'val_test_frac': 0.2,
#         'random_state': 24,
#         'tile_frac_thr': 0.2,
#         'nproc': 30,
#         'n_splits': 3,
#         'labels': ['Pre-NFT', 'iNFT']
#     })
    ##
    parser = ArgumentParser()
    parser.add_argument(
        '--im-size', type=int, default=1280, 
        help='size of tile used to partition ROIs, default, default: 1280'
    )
    parser.add_argument(
        '--box-thr', type=float, default=0.50, 
        help='% area threshold of object in tile to be included, default: 0.45'
    )
    parser.add_argument('--stride', type=int, default=960, 
                        help='when tiling with overlap, use this stride')
    parser.add_argument('--save-dir', type=str, default='annotator-datasets', 
                        help='directory to save tiles')
    parser.add_argument('--rgb-fill', nargs='+', default=(114, 114, 114), 
                        help='pad color')
    parser.add_argument(
        '--val-test-frac', default=0.2, type=float, 
        help='Fraction to hold out for val and test. Note that the val and test'
             ' will be the same size, i.e. half of this.'
    )
    parser.add_argument('--random-state', type=int, default=24, 
                        help='set random-state for splitting the data')
    parser.add_argument(
        '--tile-frac-thr', type=float, default=.20, 
        help='area threshold of tile that must be in ROI to be included'
    )
    parser.add_argument('--nproc', type=int, default=30, 
                        help='number of processors, for multiprocessing')
    parser.add_argument(
        '--n-splits', default=3, type=int, 
        help='number of train/val splits to create for each datasets'
    )
    parser.add_argument('--labels', default=['Pre-NFT', 'iNFT'], nargs='+', 
                        type=str, help='class labels, ordered')
    return parser.parse_args()


def create_dataset_files(df, save_dir, opt):
    """Create YOLO dataset files from a tiles dataframe."""
    yamls_dir = join(save_dir, 'yamls')
    txts_dir = join(save_dir, 'texts')
    csvs_dir = join(save_dir, 'csvs')
    
    makedirs(yamls_dir, exist_ok=True)
    makedirs(txts_dir, exist_ok=True)
    makedirs(csvs_dir, exist_ok=True)
    
    annotators = list(df.annotator.unique())
    experts = [ann for ann in annotators if ann.startswith('expert')]
    
    yaml_dict = {'names': opt.labels, 'nc': len(opt.labels), 'path': txts_dir}
    
    # Add test datasets.
    for test_fp in glob('/workspace/data/datasets/test-datasets/texts/*.txt'):
        fn = get_filename(test_fp)
        
        copyfile(test_fp, join(txts_dir, fn + '.txt'))
        copyfile(
            join(dirname(dirname(test_fp)), 'csvs', fn + '.csv'),
            join(csvs_dir, fn + '.csv')
        )
        
        if fn.startswith('test-'):
            yaml_dict[fn] = fn + '.txt'
        else:
            yaml_dict[f'test-{fn}'] = fn + '.txt'
    
    # loop through each annotator
    for annotator in list(df.annotator.unique()): # + ['experts', 'all']:
        # subset to only this annotator(s)
        if annotator == 'experts':
            ann_df = df[
                (df.annotator.isin(experts)) & (df.roi_group == 'ROIv1')
            ]
        elif annotator == 'all':
            # remove ROIv2s
            ann_df = df[df.roi_group == 'ROIv1']
        else:
            ann_df = df[df.annotator == annotator]
            
        # Split the dataset by WSI.
        wsi_names = sorted(list(ann_df.wsi_name.unique()))
        
        # get the number of that should be in test and val
        train, val_test = train_test_split(
            wsi_names, train_size=1 - opt.val_test_frac
        )
        test, val = train_test_split(val_test, train_size=0.5)
        
        # now get the names
        train_val_wsis, test_wsis = train_test_split(wsi_names, 
                                                     test_size=len(test))
        
        test_df = ann_df[ann_df.wsi_name.isin(test_wsis)]
        test_df.to_csv(join(csvs_dir, f'{annotator}-test.csv'), index=False)
        
        # create the dataset text file for the test data
        create_dataset_txt(test_df, join(txts_dir, f'{annotator}-test.txt'))
        yaml_dict['test'] = f'{annotator}-test.txt'
        
        for n in range(opt.n_splits):
            # get the splits train and val
            train_wsis, val_wsis = train_test_split(train_val_wsis, 
                                                    train_size=len(train))
            
            train_df = ann_df[ann_df.wsi_name.isin(train_wsis)]
            val_df = ann_df[ann_df.wsi_name.isin(val_wsis)]
            
            train_df.to_csv(join(csvs_dir, f'{annotator}-train-n{n+1}.csv'), 
                            index=False)
            val_df.to_csv(join(csvs_dir, f'{annotator}-val-n{n+1}.csv'), 
                          index=False)
            
            # create the train / val files
            create_dataset_txt(train_df, 
                               join(txts_dir, f'{annotator}-train-n{n+1}.txt'))
            create_dataset_txt(val_df, 
                               join(txts_dir, f'{annotator}-val-n{n+1}.txt'))
            
            yaml_dict['train'] = f'{annotator}-train-n{n+1}.txt'
            yaml_dict['val'] = f'{annotator}-val-n{n+1}.txt'
            
            # save the yaml file
            with open(join(yamls_dir, f'{annotator}-n{n+1}.yaml'), 'w') as fh:
                yaml.dump(yaml_dict, fh)

                
def tile_roi(r, save_dir, box_thr, im_size, stride, rgb_fill, cols):
    """Tile ROI."""
    rows = []

    # read the ROI image
    roi_im = imread(r.roi_im_path)

    # create mask for rotated regions (0: not in ROI, 1: in ROI)
    roi_box = line_to_xys(r.roi_corners) - (r.roi_im_left, r.roi_im_top)
    roi_mask = np.zeros(roi_im.shape[:2], dtype=np.int8)
    roi_mask = cv.drawContours(roi_mask, [roi_box], -1, 1, cv.FILLED)
    
    # apply ROI mask to the image
    roi_im[np.where(roi_mask == 0)] = rgb_fill

    # get boxes coordinates for ROI
    boxes = read_roi_txt_file(r.roi_labels) if isfile(r.roi_labels) else []

    # Tile ROI.
    tile_paths = tile_im_with_boxes(
        roi_im, boxes, save_dir, rotated_mask=roi_mask, box_thr=box_thr, 
        tile_size=im_size, stride=stride, pad_rgb=rgb_fill, 
        savename=f'{r.wsi_id}-x{r.roi_im_left}y{r.roi_im_top}-'
    )

    for tile_path in tile_paths:
        im_path, x, y = tile_path
        rows.append(
            [r[c] for c in cols] + [
                im_path, x+r.roi_im_left, y+r.roi_im_top, 
                x+r.roi_im_left+im_size,y+r.roi_im_top+im_size
            ]
        )

    return rows


def main():
    """Main function"""
    print(Fore.BLUE, Style.BRIGHT, 'Creating annotator datasets.\n', 
          Style.RESET_ALL)
    opt = parse_opt()
    print_opt(opt)
    
    np.random.seed(opt.random_state)
    
    cf = load_yaml()  # configuration variables
    
    # create directories to save images and labels
    save_dir = join(cf.datadir, 'datasets', opt.save_dir)
    makedirs(save_dir, exist_ok=True)
    
    # - Tile (divide it into smaller images) the ROIs for use in YOLOv5 models
    rois_df = read_csv(join(cf.codedir, 'csvs/labeled-rois.csv'))
    # rois_df = rois_df[rois_df.cohort == 'Annotated-Cohort']
    
    # Subset to annotators to train models for.
    rois_df = rois_df[rois_df.annotator.isin(VALID_ANNOTATORS)]
    
    # Tile the images if they have not been tiled before.
    tile_csv_fp = join(save_dir, 'tiles.csv')
    
    if isfile(tile_csv_fp):
        print(Fore.YELLOW, Style.BRIGHT, '  Reading tile data from file, '
              f'delete {tile_csv_fp} to run again instead.', Style.RESET_ALL)
        tiles_df = read_csv(tile_csv_fp)
    else:
        # columns to add from rois data to each tile entry
        cols = [
            'wsi_name', 'case', 'annotator', 'wsi_id', 'parent_id', 
            'Braak_stage', 'region', 'annotator_experience', 'scan_mag', 
            'roi_group', 'roi_im_path', 'roi_im_left', 'roi_im_top', 
            'roi_im_right', 'roi_im_bottom', 'url_to_roi', 'url_to_parent_roi', 
            'roi_corners', 'roi_labels'
        ]

        # Tile each ROI with multiprocessing.
        print(Fore.CYAN, Style.BRIGHT, '  Tiling ROIs.', Style.RESET_ALL)
        with Pool(opt.nproc) as pool:
            jobs = [
                pool.apply_async(
                    func=tile_roi, 
                    args=(r, save_dir, opt.box_thr, opt.im_size, opt.stride, 
                          opt.rgb_fill, cols)
                )
                for _, r in rois_df.iterrows()
            ]

            rows = []

            for job in tqdm(jobs):
                rows.extend(job.get())

        tiles_df = DataFrame(
            data=rows, 
            columns=cols + ['im_path', 'im_left', 'im_top', 'im_right', 
                            'im_bottom']
        )
        
        tiles_df = tiles_df.rename(columns={
            'roi_im_path': 'roi_fp', 'roi_im_left': 'roi_x', 
            'roi_im_top': 'roi_y', 'im_path': 'fp', 'tile_size': opt.im_size
        })
        tiles_df['roi_w'] = tiles_df.roi_im_right - tiles_df.roi_x
        tiles_df['roi_h'] = tiles_df.roi_im_bottom - tiles_df.roi_y
        tiles_df['x'] = tiles_df.im_left - tiles_df.roi_x
        tiles_df['y'] = tiles_df.im_top - tiles_df.roi_y

        tiles_df.to_csv(join(save_dir, 'tiles.csv'), index=False)
        
        rois_df = rois_df.rename(columns={
            'roi_im_path': 'fp', 'roi_im_left': 'x', 'roi_im_top': 'y'
        })
        rois_df['w'] = rois_df.roi_im_right - rois_df.x
        rois_df['h'] = rois_df.roi_im_bottom - rois_df.y
        
        rois_df.to_csv(join(save_dir, 'rois.csv'), index=False)

    # Create annotator dataset YOLO files (e.g. texts, yamls, etc.).
    print(Fore.CYAN, Style.BRIGHT, 
          '\n  Creating dataset files (yamls, texts, etc.).', Style.RESET_ALL)
    
    # Remove current dataset files.
    for rm_dir in (
        join(save_dir, 'csvs'), join(save_dir, 'texts'), join(save_dir, 'yamls')
    ):
        if isdir(rm_dir):
            rmtree(rm_dir)
        
    create_dataset_files(tiles_df, save_dir, opt)
    
    print(Fore.GREEN, Style.BRIGHT, '\nDone!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
