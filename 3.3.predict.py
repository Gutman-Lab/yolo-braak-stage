# Predict on the tiles using the best models!
from pandas import read_csv
from colorama import Fore, Style
from argparse import ArgumentParser

from os import listdir, makedirs
from os.path import join, isfile, isdir

from nft_helpers.utils import load_yaml
from nft_helpers.utils import print_opt
from nft_helpers import compile_model_results
from nft_helpers.yolov5 import predict


def parse_opt(cf):
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, help='Directory of dataset.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for predictions.')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use, i.e. 0,1,2')
    parser.add_argument('--iou-thr', type=float, default=0.4,
                        help='IoU threshold used for NMS.')
    parser.add_argument('--img-size', type=int, default=1280,
                        help='Size of images to predict on, resizes if needed.')
    
    return parser.parse_args()


def main():
    print(Fore.BLUE, Style.BRIGHT, 'Predicting on tiles with best models.\n',
          Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    model_csv_fp = join(opt.dataset_dir, 'best-annotator-models.csv')
    
    # Read the best models for each annotator.
    if isfile(model_csv_fp):
        model_results = read_csv(model_csv_fp)
        model_results = model_results.sort_values(by='model')
    else:
        model_results = compile_model_results(join(cf.datadir, 'models'))

        annotators = ['expert1', 'expert2', 'expert3', 'expert4', 'expert5', 
                       'novice1', 'novice2', 'novice3']

        idx = []  # track the indices of interest.

        for annotator in annotators:
            ann_df = model_results[
                (model_results.model == annotator) & \
                (model_results.dataset == 'test') & \
                (model_results.label == 'all')
            ].sort_values(by='mAP50-95', ascending=False)
            idx.append(ann_df.index[0])

        model_results = model_results[
            model_results.index.isin(idx)
        ].sort_values(by='model')
        
        model_results.to_csv(model_csv_fp, index=False)
    
    # Get list of tile directories.
    tile_dir = join(opt.dataset_dir, 'tiles')
    roi_dirs = sorted(listdir(tile_dir))
    
    for _, r in model_results.iterrows():
        # check if this model has a directory
        first_dir = join(tile_dir, roi_dirs[0], 'predictions', r.model)
        
        if isdir(first_dir):
            print(Fore.YELLOW, Style.BRIGHT, f'Skipping model \"{r.model}\",' 
                  'because predictions exist.', Style.RESET_ALL)
            continue
        else:
            makedirs(first_dir, exist_ok=True)
            print(Fore.CYAN, Style.BRIGHT, 
                  f'Predicting for model: \"{r.model}\".', Style.RESET_ALL)
            
            # Predict for all images in ROI
            for roi_dir in roi_dirs:
                # predict
                predict(
                    join(tile_dir, roi_dir, 'images'),
                    join(tile_dir, roi_dir, 'predictions', r.model),
                    r.weights,
                    device=opt.device,
                    conf_thr=opt.conf,
                    iou_thr=opt.iou_thr,
                    im_size=opt.img_size,
                )
                
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
