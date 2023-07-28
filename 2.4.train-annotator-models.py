# Train annotator models.
from glob import glob
import yaml
from colorama import Fore, Style
import shutil
import argparse

from nft_helpers.utils import get_filename, load_yaml, print_opt
from nft_helpers.yolov5 import train, val

from os import makedirs
from os.path import isdir, join, isfile, abspath


def parse_opt(cf):
    """CLIs"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str, default='annotator-datasets', 
        help='dir in <data_dir>/datasets/ that contains the yaml dir'
    )
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of epochs to train to')
    parser.add_argument('--weights', type=str, default='yolov5m6.pt', 
                        help='yolo weights to initiate the model with')
    parser.add_argument('--img', type=int, default=1280, 
                        help='size of image to train with, will resize to this')
    parser.add_argument('--hyp', type=str, help='Hyperparameters.',
                        default=join(cf.codedir, 'hyps/hyp.yaml'))
    parser.add_argument('--batch-per-device', type=int, default=12, 
                        help='batch size per GPU')
    parser.add_argument('--bn', action='store_true', 
                        help='use batch normalization during training')
    parser.add_argument(
        '--device', type=str, default=None, 
        help='cpu, 0, 1, 2, or set to None to get all available.'
    )
    parser.add_argument('--iou-thres', type=float, default=0.4, 
                        help='NMS IoU threshold to use when validating')
    parser.add_argument('--prepend-name', type=str, default='', 
                        help='name to prepend on all models with')
    parser.add_argument(
        '--model-dir', type=str, default='models', 
        help='Models are saved in your data directory in a directory of this '
             'name.'
    )
    parser.add_argument(
        '--patience', type=int, default=20, 
        help='Number of epochs without improvement before early stopping.'
    )
    
    return parser.parse_args()


def main():
    """Main functions."""
    print(Fore.BLUE, Style.BRIGHT, 
          'Training YOLOv5 models for annotator datasets.', Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf) 
    print_opt(opt)
    
    makedirs(join(cf.datadir, opt.model_dir), exist_ok=True)
    
    # hyperparameter files are all in the hyps dir
    hyp = abspath(opt.hyp)
    
    # check that yaml dir exists
    yamldir = join(cf.datadir, 'datasets', opt.dataset_dir, 'yamls')
    
    if not isdir(yamldir):
        raise Exception(f'yaml directory does not exist ({yamldir})')
    
    # Train for all yaml files.
    for yaml_filepath in sorted(glob(join(yamldir, '*.yaml'))):
        model_name = get_filename(yaml_filepath)
        
        # Skip this model if its directory exists.
        model_dir = join(
            cf.datadir, opt.model_dir, 
            f'{opt.prepend_name}{model_name}/{opt.prepend_name}{model_name}'
        )
        
        if isdir(model_dir):
            print(Fore.YELLOW, Style.BRIGHT, '\n  Skipping existing model ' 
                  f'"{opt.prepend_name}{model_name}".', Style.RESET_ALL)
        else:
            makedirs(model_dir, exist_ok=True)
            print(
                Fore.CYAN, Style.BRIGHT, 
                f'\n  Training model "{opt.prepend_name}{model_name}".', 
                Style.RESET_ALL
            )

            train(
                model_dir, yaml_filepath, hyp, epochs=opt.epochs, 
                weights=opt.weights, img=opt.img, 
                batch_per_device=opt.batch_per_device, bn=opt.bn, 
                device=opt.device, exist_ok=True, patience=opt.patience
            )

            # Look for model completion otherwise exit.
            if not isfile(join(model_dir, 'confusion_matrix.png')):
                # delete the directory raise exception
                shutil.rmtree(model_dir)
                raise Exception(f'model {model_dir} failed to train')

            # Run val script on available datasets (exclude Train).
            with open(yaml_filepath, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            
            # create directory for val.py results
            validate_dir = join(model_dir, 'validate')
            makedirs(validate_dir, exist_ok=True)
            
            # best model weights
            weights = join(model_dir, 'weights/best.pt')

            for task, value in yaml_dict.items():
                if value != '' and task.startswith(('val', 'test')):
                    task_dir = join(validate_dir, task)
                    makedirs(task_dir, exist_ok=True)

                    print(
                        Fore.CYAN, Style.BRIGHT, 
                        f'\n  Validating on "{task}" dataset using model '
                        f'{model_name}.', Style.RESET_ALL
                    )
                    
                    val(
                        weights, task_dir, yaml_filepath, img=opt.img, 
                        batch_per_device=opt.batch_per_device, 
                        device=opt.device, task=task, iou_thres=opt.iou_thres, 
                        exist_ok=True
                    )
        
    print(Fore.GREEN, Style.BRIGHT, '\nDone!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
