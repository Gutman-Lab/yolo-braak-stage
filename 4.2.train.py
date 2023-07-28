# Train next iteration of models (model-assisted-labeling workflow).
from colorama import Fore, Style
from argparse import ArgumentParser
from pandas import read_csv

from os import makedirs
from os.path import join, isfile

from nft_helpers.utils import load_yaml, print_opt
from nft_helpers.yolov5 import train_models
from nft_helpers import compile_model_results


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
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
    print(Fore.BLUE, Style.BRIGHT, 'Boiler Plate Code.\n', Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    # Read the model assisted labeling file
    df_fp = join(opt.dataset_dir, 'model-assisted-labeling.csv')
    
    if not isfile(df_fp):
        print(
            Fore.RED, Style.BRIGHT, 
            'Workflow not started.\n', 
            'Use section 1 in \"4.1.model-assisted-labeling.ipynb\"',
            Style.RESET_ALL
        )
        return
    
    df = read_csv(df_fp)
    
    # Get current interation
    iteration = df.iteration.max()
    
    # Make sure that the current iteration is all checked and updated
    df = df[df.iteration == iteration]
    
    if not all(df.checked):
        print(
            Fore.RED, Style.BRIGHT, 
            'ROIs need checking.\n', 
            'Use section 2 in \"4.1.model-assisted-labeling.ipynb\"',
            Style.RESET_ALL)
        return
    
    if not all(df.labels_updated):
        print(
            Fore.RED, Style.BRIGHT, 
            'Tile labels not updated.\n', 
            'Use section 3 in \"4.1.model-assisted-labeling.ipynb\"',
            Style.RESET_ALL)
        return

    # Train models.
    model_dir = join(cf.datadir, 'models/model-assisted-labeling')
    makedirs(model_dir, exist_ok=True)
    
    train_models(
        join(opt.dataset_dir, 'yamls'), model_dir, hyp=opt.hyp,
        epochs=opt.epochs, weights=opt.weights, im_size=opt.img, 
        device=opt.device, batch_per_device=opt.batch_per_device, 
        iou_thres=opt.iou_thr, prepend_name=f'iteration{iteration}-', 
        patience=opt.patience
    )
    
    # Compile model results
    model_results = compile_model_results(model_dir)
    model_results.to_csv(join(opt.dataset_dir, 'models.csv'), index=False)
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
