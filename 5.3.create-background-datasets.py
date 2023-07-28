# Create a dataset with additional background ROIs.
from colorama import Fore, Style
from pandas import read_csv, concat
from os.path import join
from glob import glob
import yaml

from nft_helpers.utils import load_yaml, get_filename
from nft_helpers.girder_dsa import login
from nft_helpers import get_wsi_rois_wrapper, tile_roi_with_labels_wrapper


def main():
    """Main function."""
    print(
        Fore.BLUE, Style.BRIGHT, 
        'Create dataset with additional background ROIs.\n', Style.RESET_ALL
    )
    cf = load_yaml()
    
    # Authenticate girder client.
    gc = login(join(cf.dsaURL, 'api/v1'), username=cf.user, 
               password=cf.password)
    
    # Search the Inference-Cohort-1 WSIs.
    wsis = read_csv('csvs/wsis.csv')
    wsis = wsis[wsis.cohort == 'Inference-Cohort-1']
    
    # Create directories to save new images / label files.
    dataset_dir = join(cf.datadir, 'datasets/model-assisted-labeling')
    roi_dir = join(dataset_dir, 'background-rois')
    tile_dir = join(dataset_dir, 'background-tiles')
    
    # Dataset directories.
    csv_dir = join(dataset_dir, 'csvs')
    txt_dir = join(dataset_dir, 'texts')
    yaml_dir = join(dataset_dir, 'yamls-extras')
    
    # Download ROIs.
    print(Fore.CYAN, 'Download background ROIs.', Style.RESET_ALL)
    rois_df = get_wsi_rois_wrapper(
        gc, wsis.wsi_id.tolist(), roi_groups='background-roi', 
        docs='background-rois', save_dir=roi_dir, notebook=False
    )
    rois_df.to_csv(join(dataset_dir, 'background-rois.csv'), index=False)
    
    # Tile the ROIs.
    print(Fore.CYAN, 'Tile the new ROIs.', Style.RESET_ALL)
    tiles_df = tile_roi_with_labels_wrapper(
        rois_df.fp.tolist(), tile_dir, stride=960, notebook=False
    )

    # Get a map of ROIs to wsi name and ROI shape.
    tiles_df['roi_w'] = [0] * len(tiles_df)
    tiles_df['roi_h'] = [0] * len(tiles_df)
    
    roi_map = {}
    
    for _, r in rois_df.iterrows():
        roi_map[r.fp] = [r.wsi_name, r.w, r.h]
        
    for i, r in tiles_df.iterrows():
        tiles_df.loc[i, 'wsi_name'] = roi_map[r.roi_fp][0]
        tiles_df.loc[i, 'roi_w'] = roi_map[r.roi_fp][1]
        tiles_df.loc[i, 'roi_h'] = roi_map[r.roi_fp][2]
        
    tiles_df.to_csv(join(dataset_dir, 'background-tiles.csv'), index=False)
    
    # Add these tiles to the a current dataset - only on the train.
    for i, yaml_fp in enumerate(sorted(glob(join(
        dataset_dir, 'yamls-extras/iteration8-cleaned-only-n*.yaml'
    )))):
        with open(yaml_fp, 'r') as fh:
            data = yaml.safe_load(fh)
            
        # Read the tile csv for train.
        train_fn = get_filename(data['train'])
        train_df = read_csv(join(dataset_dir, 'csvs', train_fn + '.csv'))
        
        # Combine the tiles dataframes.
        train_df = concat([train_df, tiles_df], ignore_index=True)
        
        # Create a new train text file.
        fn = f'additional-background-rois-n{i+1}.'
        data['train'] = fn + 'txt'
        
        train_df.to_csv(join(csv_dir, fn + 'csv'))
        
        lines = ''
        
        for _, r in train_df.iterrows():
            lines += f'{r.fp}\n'
            
        with open(join(txt_dir, fn + 'txt'), 'w') as fh:
            fh.write(lines.strip())
            
        # Safe the new dataset yaml file.
        with open(join(yaml_dir, fn + 'yaml'), 'w') as fh:
            yaml.dump(data, fh)
            
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)
    

if __name__ == '__main__':
    main()
