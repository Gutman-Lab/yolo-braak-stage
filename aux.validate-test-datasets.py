# Validate all available models on test datasets.
from colorama import Fore, Style
from argparse import ArgumentParser
from glob import glob

from os import listdir, makedirs
from os.path import join, isfile, isdir

from nft_helpers.utils import load_yaml, print_opt, get_filename
from nft_helpers.yolov5 import val


def parse_opt(cf):
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument('--img-size', type=int, default=1280, 
                        help='Size of images.')
    parser.add_argument('--batch-per-device', type=int, default=12, 
                        help='Batch size per GPU.')
    parser.add_argument(
        '--device', type=str, default=None, 
        help='cpu, 0, 1, 2, ... , or None for all available devices.'
    )
    parser.add_argument('--iou_thr', type=float, default=0.4, 
                        help='NMS IoU threshold to use when validating.')
    parser.add_argument(
        '--model-dirs', type=str, nargs='+', help='Model directories.',
        default=[
            join(cf.datadir, 'models'),
            join(cf.datadir, 'models/model-assisted-labeling')
        ]
    )
    return parser.parse_args()
    
    
def main():
    """Main function"""
    print(Fore.BLUE, Style.BRIGHT, 'Validate models on test dataset.\n',
          Style.RESET_ALL)
    cf = load_yaml()
    opt = parse_opt(cf)
    print_opt(opt)
    
    yaml_fps = {}
    
    for yaml_fp in sorted(glob(join(cf.datadir, 
                                    'datasets/test-datasets/yamls/*.yaml'))):
        yaml_fps[get_filename(yaml_fp)] = yaml_fp
    
    # list all the model directories
    for parent_dir in opt.model_dirs:
        for model in listdir(parent_dir):
            model_dir = join(parent_dir, model, model)

            # check if there is a terminal_output file, this signals the model is done training.
            if isfile(join(model_dir, 'terminal_output.csv')):
                # create validate dir
                validate_dir = join(model_dir, 'validate')
                makedirs(join(model_dir, 'validate'), exist_ok=True)

                # get the best weights
                weights = join(model_dir, 'weights/best.pt')

                # loop through each test dataset and evaluate if the directory does not exist
                for dataset, yaml_fp in yaml_fps.items():
                    task_dir = join(validate_dir, dataset)

                    if not isdir(task_dir):
                        makedirs(task_dir, exist_ok=True)

                        print(Fore.CYAN, Style.BRIGHT, f'\n  Model \"{model}\" on dataset \"{dataset}\"\n',
                              Style.RESET_ALL)

                        # validate
                        val(
                            weights, task_dir, yaml_fp, img=opt.img_size, 
                            batch_per_device=opt.batch_per_device, 
                            device=opt.device, task='test', iou_thres=opt.iou_thr, 
                            exist_ok=True
                        )
                    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)


if __name__ == '__main__':
    main()
