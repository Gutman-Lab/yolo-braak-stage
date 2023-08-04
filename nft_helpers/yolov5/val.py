# Python wrapper for running ultralytics/yolov5/val.py function
from subprocess import Popen
from .utils import get_devices

from os import chdir, getcwd
from os.path import join, isdir, isfile, abspath


def val(weights, savedir, data, img=1280, batch_per_device=12, device=None, yolodir='/workspace/yolo', conf_thres=.001, iou_thres=.35, task='val', 
        verbose=True, save_txt=True, save_conf=True, agnostic_nms=True, exist_ok=False):
    """
    Validate trained model on a dataset with labels.
    
    INPUTS
    ------
    weights : str
        path the .pt weights of the YOLO model to load
    savedir : str
        directory to save validation results
    data : str
        path to yaml file containing the dataset path
    img : int (default: 1280)
        size of images to validate, will resize if needed
    batch_per_device : int (default: 12)
        batch size is calcualted by this parameter and the number of GPUs
    device : str (default: None)
        if None it will get the all the GPUs available, otherwise specify the id of the GPUs to use in this format: "0,1,2"
    yolodir : str (default: "/workspace/yolo")
        directory of the yolo repository, where train.py is
    conf_thres : float (default: 0.001)
        keep confidence threshold low to calculate the mAP metrics correctly
    iou_thres : float (default: 0.35)
        NMS IoU threshold
    task : str (default: val)
        dataset in yaml file to validate on
    verbose : bool (default: True)
        how much to output to terminal
    save_txt : bool (default: True)
        save predictions to label text files for each image
    save_conf : bool (default: True)
        add the confidence to the label text file if saved
    agnostic_nms : bool (default: True)
        evaluate NMS with agnostic label behaviour if True
    exist_ok : bool (default: False)
        overwrite results in savedir if it exists
    
    """
    if device is None:
        device, n_devices = get_devices(device)
    else:
        if device.endswith(','):
            device = device[:-1]
        n_devices = len(device.split(','))
    
    batch = batch_per_device * n_devices
    
    # split savedir into project and name
    dirs = abspath(savedir).split('/')
    
    if dirs[-1] == '':
        dirs = dirs[:-1]
        
    if len(dirs) == 1:
        raise Exception('savedir must not be in /')
        
    name = dirs[-1]
    project = '/'.join(dirs[:-1])
    
    if isdir(savedir) and not exist_ok:
        print(f'Not validating, {savedir} already exists')
    else:
        # setup the CLI string
        command = f'python val.py --data {data} --weights {weights} --batch {batch} --img {img} --conf-thres {conf_thres} --iou-thres {iou_thres}' + \
                  f' --task {task} --device {device} --project {project} --name {name} --save-results'

        if verbose:
            command += f' --verbose'
        if save_txt:
            command += f' --save-txt'
        if save_conf:
            command += f' --save-conf'
        if agnostic_nms:
            command += f' --agnostic-nms'
        if exist_ok:
            command += f' --exist-ok'

        # switch to the directory with the yolov5 train.py script
        starting_dir = getcwd()
        chdir(yolodir)

        process = Popen(command.split(' '))
        process.wait()

        # change back to repository
        chdir(starting_dir)
