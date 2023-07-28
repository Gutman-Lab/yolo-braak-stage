# Split the datasets into regions.
from pandas import read_csv
from nft_helpers.utils import load_yaml
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from colorama import Fore, Style


def main():
    """Main function."""
    print(
        Fore.BLUE, Style.BRIGHT, 
        'Create dataset of tiles for each brain region.\n', 
        Style.RESET_ALL
    )
    cf = load_yaml()
    
    # Read ROI data.
    dataset_dir = join(cf.datadir, 'datasets/model-assisted-labeling')
    rois = read_csv(join(dataset_dir, 'rois.csv'))
    tiles = read_csv(join(dataset_dir, 'tiles.csv'))
    
    # Dirs to save new files to.
    yaml_dir = join(dataset_dir, 'yamls-extras')
    txt_dir = join(dataset_dir, 'texts')
    csv_dir = join(dataset_dir, 'csvs')
    
    # Map the hippocampus into one.
    rois.region = rois.region.replace(
        {'Left hippocampus': 'Hippocampus', 'Right hippocampus': 'Hippocampus',
         'Temporal cortex': 'Temporal', 'Occipital cortex': 'Occipital'}
    )
    
    # Add wsi name to the tiles.
    wsi_map = {r.fp: r.wsi_name for _, r in rois.iterrows()}
    
    for i, r in tiles.iterrows():
        tiles.loc[i, 'wsi_name'] = wsi_map[r.roi_fp]
    
    np.random.seed(64)
    
    # Read the test dataset info - to separate into new test datasets by region.
    test_dir = join(cf.datadir, 'datasets/test-datasets')
    
    test_rois = read_csv(join(test_dir, 'rois.csv'))
    test_rois.region = test_rois.region.replace(
        {'Left hippocampus': 'Hippocampus', 'Right hippocampus': 'Hippocampus',
         'Temporal cortex': 'Temporal', 'Occipital cortex': 'Occipital'}
    )
    
    test_tiles = read_csv(join(test_dir, 'csvs/test-roi.csv'))
    external_tiles = read_csv(join(test_dir, 'csvs/test-external-roi.csv'))
        
    # Add wsi name to the tiles.
    region_map = {r.fp: r.region for _, r in test_rois.iterrows()}
    
    for i, r in test_tiles.iterrows():
        test_tiles.loc[i, 'region'] = region_map[r.roi_fp]
        
    for i, r in external_tiles.iterrows():
        external_tiles.loc[i, 'region'] = region_map[r.roi_fp]
        
    for region in rois.region.unique():
        # Create test datasets.
        test_fps = ''
        external_fps = ''
        
        test_fn = f'test-roi-{region}'
        external_fn = f'test-external-roi-{region}'
        
        test_reg_tiles = test_tiles[test_tiles.region == region]
        external_reg_tiles = external_tiles[external_tiles.region == region]
        
        test_reg_tiles.to_csv(join(csv_dir, test_fn + '.csv'), index=False)
        external_reg_tiles.to_csv(join(csv_dir, external_fn + '.csv'), index=False)
        
        for _, r in test_reg_tiles.iterrows():
            test_fps += f'{r.fp}\n'
            
        for _, r in external_reg_tiles.iterrows():
            external_fps += f'{r.fp}\n'
            
        with open(join(txt_dir, f'{test_fn}.txt'), 'w' ) as fh:
            fh.write(test_fps.strip())

        with open(join(txt_dir, f'{external_fn}.txt'), 'w') as fh:
            fh.write(external_fps.strip())

        # Split the train and val datasets.
        region_rois = rois[rois.region == region]
        
        wsis = sorted(list(region_rois.wsi_name.unique()))
        
        savename = f'iteration8-{region}'
        
        # Create splits.
        for n in range(1, 4):
            train_wsis, val_wsis = train_test_split(wsis, test_size=0.1)

            # Split the tiles into train and val.
            train_tiles = tiles[tiles.wsi_name.isin(train_wsis)]
            val_tiles = tiles[tiles.wsi_name.isin(val_wsis)]

            train_name = f'{savename}-train-n{n}'
            val_name = f'{savename}-val-n{n}'
            
            train_tiles.to_csv(join(csv_dir, f'{train_name}.csv'), index=False)
            val_tiles.to_csv(join(csv_dir, f'{val_name}.csv'), index=False)

            # Create yaml and text files.
            yaml_dict = {
                'names': ['Pre-NFT', 'iNFT'], 'nc': 2, 'path': txt_dir, 
                'test': '', 'train': f'{train_name}.txt', 
                'val': f'{val_name}.txt', 'test-roi': test_fn + '.txt',
                'test-external-roi': external_fn + '.txt'
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
