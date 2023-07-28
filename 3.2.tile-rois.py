# Tile ROIs with labels.
from colorama import Fore, Style
from argparse import ArgumentParser
from pandas import read_csv
from os.path import join, isfile

from nft_helpers import tile_roi_with_labels_wrapper
from nft_helpers.utils import load_yaml, get_filename, print_opt


def parse_opt(cf):
    """Parse CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, 
        default=join(cf.datadir, 'datasets/model-assisted-labeling'),
        help='Dataset directory, containing rois.csv file.'
    )
    parser.add_argument('--tile-size', type=int, default=1280,
                        help='Tile size.')
    parser.add_argument('--stride', type=int, default=960, help='Stride.')
    parser.add_argument('--boundary-thr', type=float, default=0.2, 
                        help='Fraction of tile that must be in ROI to include.')
    parser.add_argument('--fill', type=int, default=(114, 114, 114),
                        help='RGB when padding image')
    parser.add_argument('--box-thr', type=float, default=0.5, 
                        help='Fraction of label box in tile to include.')
    parser.add_argument('--nproc', type=int, default=10,
                        help='Parallel processes.')
    parser.add_argument('--ignore-existing', action='store_true',
                        help='Ignore existing tiling and run again.')
    
    return parser.parse_args()
    
    
def main():
    print(Fore.BLUE, Style.BRIGHT, 'Tile ROIs with labels.\n', Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # ROI data
    roi_df = read_csv(join(opt.dataset_dir, 'rois.csv'))
    
    # Tile the ROIs is not done yet.
    tile_fp = join(opt.dataset_dir, 'tiles.csv')
    
    if isfile(tile_fp) and not opt.ignore_existing:
        print(Fore.YELLOW, Style.BRIGHT, 
              'Skipping tiling because tiles.csv exists, pass --ignore-existing' 
              ' to run again.', Style.RESET_ALL)
    else:
        # create a list of save directories for each unique ROI
        fps = roi_df.fp.tolist()
        save_dirs = []

        for fp in fps:
            save_dirs.append(join(opt.dataset_dir, 'tiles', get_filename(fp)))

        # Tile with parallel.
        tile_df = tile_roi_with_labels_wrapper(
            fps, save_dirs, tile_size=opt.tile_size, stride=opt.stride, 
            boundary_thr=opt.boundary_thr, nproc=opt.nproc, fill=opt.fill, 
            box_thr=opt.box_thr,
        )

        tile_df.to_csv(join(opt.dataset_dir, 'tiles.csv'), index=False)

    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)
    
    
if __name__ == '__main__':
    main()
