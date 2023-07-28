# Train models with "messy" labels (consensus of multiple model predictions).
from colorama import Fore, Style
from argparse import ArgumentParser
from os.path import isdir, join

from nft_helpers.utils import load_yaml, print_opt
from nft_helpers.yolov5 import train_models


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--consensus-n', type=int, required=True,
        help='Number of models in agreement for dataset used to train, affects '
             'the directory name of the model.'
    )
    parser.add_argument(
        '--dataset-dir', type=str, help='Dataset directory.',
        default=join(cf.datadir, 'datasets/model-assisted-labeling')
    )
    parser.add_argument('--hyp', type=str, help='Hyperparameter file.',
                        default=join(cf.codedir, 'hyps/hyp.yaml'))
    parser.add_argument('--epochs', type=int, default=150, 
                        help='Number of training epochs.')
    parser.add_argument(
        '--patience', type=int, default=20, 
        help='Number of epochs without improvement before early stopping.'
    )
    parser.add_argument('--weights', type=str, default='yolov5m6.pt', 
                        help='yolo weights to initiate the model with')
    parser.add_argument('--img', type=int, default=1280, 
                        help='size of image to train with, will resize to this')
    parser.add_argument('--batch-per-device', type=int, default=12, 
                        help='batch size per GPU')
    parser.add_argument(
        '--device', type=str, default=None, 
        help='cpu, 0, 1, 2, or None for all available cuda devices.')
    parser.add_argument('--iou_thr', type=float, default=0.4, 
                        help='NMS IoU threshold to use when validating')
    
    return parser.parse_args()
    
    
def main():
    """Main function"""
    print(Fore.BLUE, Style.BRIGHT, 'Train consensus labeled models.\n', 
          Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # check that yaml dir exists
    yamldir = join(opt.dataset_dir, 'yamls')
    
    if not isdir(yamldir):
        raise Exception(
            f'No \"yaml\" directory does not exist in \"{opt.dataset_dir}\".'
        )
        
    # run the training function
    train_models(
        yamldir, join(cf.datadir, 'models'), hyp=opt.hyp, epochs=opt.epochs, 
        weights=opt.weights, im_size=opt.img, device=opt.device,
        batch_per_device=opt.batch_per_device, iou_thres=opt.iou_thr, 
        prepend_name=f'{opt.consensus_n}-models-consensus-', 
        patience=opt.patience
    )
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL) 


if __name__ == '__main__':
    main()
