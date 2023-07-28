# Create dataset files.
from colorama import Fore, Style
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
from argparse import ArgumentParser

from os import makedirs
from os.path import join

from nft_helpers.utils import load_yaml, print_opt, get_filename
from nft_helpers.yolov5.utils import create_dataset_txt


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, help='Dataset directory.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument('--random-state', type=int, default=614, 
                        help='Random state.')
    parser.add_argument('--val-frac', type=float, default=0.1, 
                        help='Fraction of ROIs in validation dataset.')
    parser.add_argument('--n-splits', type=int, default=3, 
                        help='Number of train / validation splits to create.')
    opt = parser.parse_args()
    
    if opt.val_frac >= 1:
        raise Exception('val-frac CLI should be lower than 1.')
        
    return opt


def main():
    """Main function."""
    print(Fore.BLUE, Style.BRIGHT, 'Creating dataset files.\n', Style.RESET_ALL)
    
    # Read tile metadata.
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Read metadata to split the data by WSI.
    tile_df = read_csv(join(opt.dataset_dir, 'tiles.csv'))
    
    # Add _id column to each tile.
    for i, r in tile_df.iterrows():
        tile_df.loc[i, 'wsi_id'] = get_filename(r.fp).split('-x')[0]
    
    # Get list of unique ROIs - splitting data by unique ROIs.
    wsi_ids = sorted(list(tile_df.wsi_id.unique()))
    
    # Create directories to save files.
    txt_dir = join(opt.dataset_dir, 'texts')
    yaml_dir = join(opt.dataset_dir, 'yamls')
    csv_dir =join(opt.dataset_dir, 'csvs')
    
    makedirs(txt_dir, exist_ok=True)
    makedirs(yaml_dir, exist_ok=True)
    makedirs(csv_dir, exist_ok=True)
    
    # Get the labels.
    yaml_dict = {'nc': len(cf.labels), 'names': cf.labels, 'path': txt_dir,
                 'test': ''}
    
    # Seed random behaviour for reproducibility.
    np.random.seed(opt.random_state)
    
    # Create the splits!
    for n in range(1, opt.n_splits+1):
        # Split the ROIs into train and val.
        train_wsi_ids, val_wsi_ids = train_test_split(wsi_ids, 
                                                      test_size=opt.val_frac)
        
        # Create text files.
        train_df = tile_df[tile_df.wsi_id.isin(train_wsi_ids)]
        val_df = tile_df[tile_df.wsi_id.isin(val_wsi_ids)]
        
        # Save CSVs
        train_df.to_csv(join(csv_dir, f'train-n{n}.csv'), index=False)
        val_df.to_csv(join(csv_dir, f'val-n{n}.csv'), index=False)
        
        train_txt_fn = f'train-n{n}.txt'
        val_txt_fn = f'val-n{n}.txt'
        
        # Saves the text files.
        create_dataset_txt(train_df, join(txt_dir, train_txt_fn), fp_col='fp')
        create_dataset_txt(val_df, join(txt_dir, val_txt_fn), fp_col='fp')
        
        # Save yaml.
        yaml_dict['train'] = train_txt_fn
        yaml_dict['val'] = val_txt_fn
        
        with open(join(yaml_dir, f'n{n}.yaml'), 'w') as fh:
            yaml.dump(yaml_dict, fh)
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
