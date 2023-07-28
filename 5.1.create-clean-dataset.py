# Create a dataset of only cleaned up ROIs from model-assisted-labeling workflow.
from colorama import Fore, Style
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
from nft_helpers.utils import load_yaml

from os import makedirs
from os.path import join

from nft_helpers.utils import load_yaml, print_opt

    
def main():
    """Main function"""
    print(
        Fore.BLUE, Style.BRIGHT, 
        'Create dataset of tiles from only ROIs that have been cleaned up.\n', 
        Style.RESET_ALL
    )
    cf = load_yaml()
    
    # Read tile data.
    dataset_dir = join(cf.datadir, 'datasets/model-assisted-labeling')
    roi_df = read_csv(join(dataset_dir, 'model-assisted-labeling.csv'))
    roi_df = roi_df[roi_df.checked]
    wsis = sorted(list(roi_df.wsi_name.unique()))
    
    # Filter tiles by ROIs that have been cleaned up.
    tile_df = read_csv(join(dataset_dir, 'tiles.csv'))
    tile_df = tile_df[tile_df.roi_fp.isin(roi_df.fp)]
    
    # Add wsi name to the tiles.
    wsi_map = {r.fp: r.wsi_name for _, r in roi_df.iterrows()}
    
    for i, r in tile_df.iterrows():
        tile_df.loc[i, 'wsi_name'] = wsi_map[r.roi_fp]
    
    # Dirs to save new files to.
    yaml_dir = join(dataset_dir, 'yamls-extras')
    txt_dir = join(dataset_dir, 'texts')
    csv_dir = join(dataset_dir, 'csvs')
    makedirs(yaml_dir, exist_ok=True)
    
    savename = 'iteration8-cleaned-only'
    
    # Seeds the train / val splits by WSIs.
    np.random.seed(64)
    
    # Create three splits.
    for n in range(1, 4):
        train_wsis, val_wsis = train_test_split(wsis, test_size=0.1)

        # Split the tiles into train and val.
        train_tiles = tile_df[tile_df.wsi_name.isin(train_wsis)]
        val_tiles = tile_df[tile_df.wsi_name.isin(val_wsis)]

        train_name = f'{savename}-train-n{n}'
        val_name = f'{savename}-val-n{n}'

        train_tiles.to_csv(join(csv_dir, f'{train_name}.csv'), index=False)
        val_tiles.to_csv(join(csv_dir, f'{val_name}.csv'), index=False)

        # Create yaml and text files.
        yaml_dict = {
            'names': ['Pre-NFT', 'iNFT'], 'nc': 2, 'path': txt_dir, 
            'test': '', 'train': f'{train_name}.txt', 
            'val': f'{val_name}.txt', 'test-roi': 'test-roi.txt',
            'test-external-roi': 'test-external-roi.txt'
        }

        train_fps = ''
        val_fps = ''

        for _, r in train_tiles.iterrows():
            train_fps += f'{r.fp}\n'

        for _, r in val_tiles.iterrows():
            val_fps += f'{r.fp}\n'

        with open(join(txt_dir, f'{train_name}.txt'), 'w' ) as fh:
            fh.write(train_fps.strip())

        with open(join(txt_dir, f'{val_name}.txt'), 'w') as fh:
            fh.write(val_fps.strip())

        with open(join(yaml_dir, f'{savename}-n{n}.yaml'), 'w') as fh:
            yaml.dump(yaml_dict, fh)
            
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
