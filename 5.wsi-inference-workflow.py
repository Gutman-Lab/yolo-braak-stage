from pandas import read_csv
from argparse import ArgumentParser
from colorama import Fore, Style
import torch
from glob import glob

from os import makedirs
from os.path import join, isfile

from nft_helpers.utils import load_yaml, print_opt, get_filename, imread
from nft_helpers.girder_dsa import login
from nft_helpers.yolov5 import wsi_inference


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument('--cohorts', type=str, nargs='+', required=True, 
                        help='Cohorts to inference on.')
    parser.add_argument('--weights', type=str, required=True, 
                        help='Weights to use.')
    parser.add_argument('--save-dir', type=str, required=True, 
                        help='Results save directory.')
    parser.add_argument('--docname', type=str, required=True,
                        help='Annotation doc name.')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold.')
    parser.add_argument('--iou-thr', type=float, default=0.4, 
                        help='NMS IoU threshold.')
    parser.add_argument('--contained-thr', type=float, default=0.7, 
                        help='Contained threshold.')
    parser.add_argument('--tile-size', type=int, default=1280, 
                        help='Tile size.')
    parser.add_argument('--stride', type=int, default=960, 
                        help='Stride when tiling.')
    parser.add_argument('--nproc', type=int, default=20, 
                        help='Parallel processes')
    return parser.parse_args()


def main():
    """Main function."""
    print(Fore.BLUE, Style.BRIGHT, 'Run inference workflow.', Style.RESET_ALL)
    
    # Cohorts to keep.
    cohorts = ['Inference-Cohort-1', 'Inference-Cohort-2', 'External-Cohort']
    
    # Dataset to save the WSI inference.
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Authenticate client to push results as annotations.
    gc = login(join(cf.dsaURL, 'api/v1'), username=cf.user, 
               password=cf.password)
    
    wsi_fp = join('csvs/wsis.csv')
    
    if isfile(wsi_fp):
        wsis_df = read_csv(wsi_fp)
    else:
        wsis_df = read_csv('csvs/wsis.csv')
        wsis_df = wsis_df[wsis_df.cohort.isin(cohorts)]
        wsis_df.to_csv(wsi_fp, index=False)

    case_fp = join('csvs/cases.csv')
    
    if isfile(case_fp):
        cases_df = read_csv(case_fp)
    else:
        cases_df = read_csv('csvs/cases.csv')
        cases_df = cases_df[cases_df.cohort.isin(cohorts)]
        cases_df.to_csv(case_fp, index=False)
        
    # Filter to only WSIs in cohorts specified.
    wsis_df = wsis_df[
        wsis_df.cohort.isin(opt.cohorts)
    ].sort_values(by='wsi_name').reset_index(drop=True)
    
    # Location to save results.
    save_dir = join(cf.datadir, 'wsi-inference/results', opt.save_dir)
    makedirs(save_dir, exist_ok=True)
    
    pred_dir = join(save_dir, 'inference')
    log_dir = join(save_dir, 'logs')
    
    # Location of masks.
    mask_dir = join(cf.datadir, 'wsi-inference/tissue-masks/masks')
    
    makedirs(pred_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)
    
    # Get a map of WSI names to local filepaths.
    wsi_map = {}
    for wsi_fp in glob(join(cf.wsidir, '**/*.*'), recursive=True):
        if wsi_fp.endswith(('.ndpi', '.svs')):
            wsi_map[get_filename(wsi_fp, prune_ext=False)] = wsi_fp
            
    # Double check that all WSIs have a local file
    for wsi_name in wsis_df.wsi_name:
        if wsi_name not in wsi_map:
            raise Exception(f'Could not find filepath for {wsi_name}.')
            
        if not isfile(join(mask_dir, get_filename(wsi_name) + '.png')):
            print(join(mask_dir, get_filename(wsi_name) + '.png'))
            raise Exception(f'Could not find mask file for {wsi_name}.')
            
    # Start inference
    N = len(wsis_df)
    
    for i, r in wsis_df.iterrows():
        # See if logs file exists.
        fn = get_filename(r.wsi_name)
        
        logs_fp = join(log_dir, fn + '.txt')
        
        if isfile(logs_fp):
            print(
                Fore.YELLOW, Style.BRIGHT, 
                f'({i+1} / {N}) Exists: skipping {r.wsi_name}.', Style.RESET_ALL
            )
            continue
            
        with open(logs_fp, 'w') as fh:
            fh.write('')
            
        print(
            Fore.CYAN, Style.BRIGHT, 
            f'({i+1} / {N}) Inferencing on {r.wsi_name}', Style.RESET_ALL
        )
        
         # Start the logs - track GPU usage.
        logs = 'GPUs:\n'

        device_count = torch.cuda.device_count()

        for device in range(device_count):
            logs += f'   {torch.cuda.get_device_name(device=device)}\n'
            
        logs += '\nTimes:\n'
        mask = imread(join(mask_dir, fn + '.png'), grayscale=True)
        
        # Run inference!
        logs += wsi_inference(
            wsi_map[r.wsi_name], gc, opt.weights, mask=mask, 
            doc_name=opt.docname, wsi_id=r.wsi_id, exist_ok=True,
            stride=opt.stride, tile_size=opt.tile_size, conf_thr=opt.conf,
            nms_iou_thr=opt.iou_thr, contained_thr=opt.contained_thr,
            save_fp=join(pred_dir, fn + '.txt'), nproc=opt.nproc
        )[2]

        with open(logs_fp, 'w') as fh:
            fh.write(logs)
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
