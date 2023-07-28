from .utils import get_devices
from subprocess import Popen

from os import makedirs, getcwd, chdir
from os.path import abspath


def predict(source_dir, save_dir, weights, device=None, agnostic_nms=True, conf_thr=0.4, iou_thr=0.4, im_size=1280, 
            yolodir='/workspace/yolo', **kwargs):
    """Predict objects on a source directory of images using a YOLOv5 model
    
    INPUTS
    ------
    source_dir : str
        directory with images
    save_dir : str
        directory to save predictions - note that it will be saved in a labels directory. save_dir should be an absolute path or contain
        more then a single directory name, as it will be broken into the top level directory and the bottom path (i.e. if save_dir is
        mydir/hello-world/this it will be split into mydir/hello-world and this for use in the detect.py script)
    weights : str
        path to the .pt file with the weights to use
    device : str (default: None)
        device ids to use in format '0,1,2', if None then all available devices will be used. Alternatively, you can use a default set
        of pre-trained weights from ultralytics/yolov5, such as yolov5m6.pt
    agnostic_nms : bool (default: True)
        use agnostic NMS during predictions
    conf_thr : float (default: 0.4)
        threshold to remove low confidence predictions
    iou_thr : float (default: 0.4)
        IoU threshold used during NMS
    im_size : int (default: 1280)
        image size to use, will scale images to match this
    yolodir : str (default: '/workspace/yolo')
        directory with the detect.py script
    
    """
    # create the directory
    makedirs(save_dir, exist_ok=True)
    
    # split the save_dir into project and name
    if not save_dir.startswith('/'):
        save_dir = abspath(save_dir)
        
    save_dir = save_dir.split('/')
    project, name = '/'.join(save_dir[:-1]), save_dir[-1]
    
    # get the device if None
    if device is None:
        device = get_devices(device)[0]
        
    # create the bash command
    command = f'python detect.py --weights {weights} --source {source_dir} --img {im_size} --device {device} --save-txt --save-conf ' + \
                  f'--project {project} --name {name} --nosave --conf-thres {conf_thr} --iou-thres {iou_thr} --exist-ok'
    
    if agnostic_nms:
        command += ' --agnostic-nms'

    # get the current working dir 
    wdir = getcwd()
    
    chdir(yolodir)
    process = Popen(command.split(' '))
    process.wait()
    
    chdir(wdir)

