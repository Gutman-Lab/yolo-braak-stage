# Download the set of ROIs without labels.
from colorama import Fore, Style
from argparse import ArgumentParser
from os.path import join, isfile
from pandas import read_csv

from nft_helpers.utils import load_yaml, print_opt
from nft_helpers.girder_dsa import login
from nft_helpers import get_wsi_rois_wrapper


def parse_opt(cf):
    parser = ArgumentParser()
    parser.add_argument('--roi-groups', type=list, 
                        default=['ROIv1', 'ROIv2', 'ROIv3'],
                        help='List of ROI groups to download.')
    parser.add_argument('--docs', type=list, default=['annotations'],
                        help='List of annotation documents with ROIs.')
    parser.add_argument('--verbose', action='store_true', 
                        help='Prints out warnings.')
    parser.add_argument('--fill', type=tuple, default=(114,114,114),
                        help='Fill to use for rotated ROIs.')
    parser.add_argument(
        '--save-dir', type=str, help='Directory to download ROI files.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument('--nproc', type=int, default=3, 
                        help='Number of processes.')
    return parser.parse_args()


def main():
    print(Fore.BLUE, Style.BRIGHT, 'Downloading unlabeled ROIs from DSA.\n',
          Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    save_fp = join(opt.save_dir, 'rois.csv')
    
    gc = login(join(cf.dsaURL, 'api/v1'), username=cf.user, 
               password=cf.password)
    
    # Unlabeled dataset is in the inference cohort 1.
    wsi_df = read_csv('csvs/wsis.csv')
    wsi_df = wsi_df[wsi_df.cohort == 'Inference-Cohort-1']
    
    # Download the ROIs.
    roi_df = get_wsi_rois_wrapper(
        gc, wsi_df.wsi_id, roi_groups=opt.roi_groups, docs=opt.docs, 
        verbose=opt.verbose, fill=opt.fill, save_dir=join(opt.save_dir, 'rois'),
        nproc=opt.nproc
    )
    
    # save to ROI
    roi_df.to_csv(save_fp, index=False)
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
