# Wrapper on train.py
from glob import glob
import shutil
from colorama import Fore, Style
import yaml

from .utils import get_devices
from ..utils import get_filename
from . import train, val

from os import makedirs
from os.path import join, abspath, isdir, isfile


def train_models(yamldir: str, save_dir: str, hyp: str, exist_ok: bool = False,
                 epochs: int = 10, weights: str = 'yolov5m6.pt', 
                 im_size: int = 1280, batch_per_device: int = 12, 
                 bn: bool = False, device: str = None, iou_thres: float = 0.4, 
                 prepend_name: str = '', patience: int = 100):
    """Train YOLOv5 models from a dataset directory.
    
    Args:
        yaml_dir: Dataset yaml directories.
        save_dir: Save directory.
        hyp: Hyperparameters file. 
        exist_ok: If True, train if directory exists.
        epochs: Number of epochs to train to.
        weights: Weights to start training from.
        im_size: Size of images to train on.
        batch_per_device: Batch size per device.
        bn: True to train with batch normalization.
        device: cpu, 0, 1, 2, ... , or None for all available devices.
        iou_thres: NMS IoU threshold.
        prepend_name: String to prepend to dataset name for model name.
        patience: Number of epochs without improvement before model is early 
            stopped.
    
    """
    makedirs(save_dir, exist_ok=True)  # create dir to save models
    
    hyp = abspath(hyp)  # hyp should be absolute path
    
    print(Fore.CYAN, f'saving models in \"{save_dir}\" directory.',
          Style.RESET_ALL)
    
    # Loop through each yaml file.
    for yaml_filepath in sorted(glob(join(yamldir, '*.yaml'))):
        model_name = prepend_name + get_filename(yaml_filepath)
        
        # check if model exists - if so then skip
        model_dir = join(save_dir, model_name, model_name)
        
        if isdir(model_dir) and not exist_ok:
            print(Fore.YELLOW, Style.BRIGHT, 
                  f'\n   model "{model_name}" already exists.',
                  Style.RESET_ALL)
        else:
            makedirs(model_dir, exist_ok=True)
            print(Fore.CYAN, f'\n   training model "{model_name}"...',
                  Style.RESET_ALL)

            train(
                model_dir, yaml_filepath, hyp, epochs=epochs, weights=weights, 
                img=im_size, batch_per_device=batch_per_device, bn=bn, 
                device=device, exist_ok=True, patience=patience
            )

            # sometimes a model may fail to train, catch these issues
            if not isfile(join(model_dir, 'confusion_matrix.png')):
                # delete the directory raise exception
                shutil.rmtree(model_dir)
                raise Exception(f'model {model_dir} failed to train')

            # run val.py on non-training datasets
            with open(yaml_filepath, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            
            # create directory for val.py results
            validate_dir = join(model_dir, 'validate')
            makedirs(validate_dir, exist_ok=True)
            
            # best model weights
            best_weights = join(model_dir, 'weights/best.pt')

            for task, value in yaml_dict.items():
                if value != '' and task.startswith(('val', 'test')):
                    task_dir = join(validate_dir, task)
                    makedirs(task_dir, exist_ok=True)

                    print(Fore.CYAN, f'\n\trunning val.py on {task} dataset '
                          '(model: {model_name})', Style.RESET_ALL)
                    val(
                        best_weights, task_dir, yaml_filepath, img=im_size, 
                        batch_per_device=batch_per_device, device=device, 
                        task=task, iou_thres=iou_thres, exist_ok=True
                    )
