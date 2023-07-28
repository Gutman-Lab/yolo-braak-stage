# Python wrapper for running ultralytics/yolov5/train.py function
from subprocess import Popen
from .utils import get_devices

from os import chdir, getcwd, makedirs
from os.path import join, isdir, splitext, abspath


def train(savedir, data, hyp, epochs=100, weights='yolov5m6.pt', img=1280, save_period=-1, batch_per_device=12, bn=False, device=None,
                 yolodir='/workspace/yolo', exist_ok=False, patience=100):
    """
    Train a YOLO model using train.py script from yolo repository.
    
    INPUTS
    ------
    savedir : str
        directory to save models to - note that this path will be broken into project/name. If you pass a single name as a local directory
        the script will look for the absolute path, if it can't find it it will throw an error
    data : str
        path to yaml file specifying the train, val, test ... image path text files
    hyp : str
        path to hyperparameter yaml file
    epochs : int (default: 100)
        number of epochs to train to
    img : int (default: 1280)
        size of images to train to, if not this size they will be resized when training
    save_period : int (default: -1)
        save checkpoint after every number of epochs, or set to -1 to disable checkpoints
    batch_per_device : int (default: 12)
        batch size is calcualted by this parameter and the number of GPUs
    bn : bool (default: False)
        True to use batch normalization, potentially better performance but at a significant hit to time to train
    device : str (default: None)
        if None it will get the all the GPUs available, otherwise specify the id of the GPUs to use in this format: "0,1,2"
    yolodir : str (default: "/workspace/yolo")
        directory of the yolo repository, where train.py is
    patience : int (default: 100)
        Number of epochs without improvement before model is early stopped.
        
    """    
    if device is None:
        device, n_devices = get_devices(device)
    else:
        if device.endswith(','):
            device = device[:-1]
        n_devices = len(device.split(','))
        
    batch = batch_per_device * n_devices
    
    # split the savedir into project and name
    savedir = abspath(savedir).split('/')
    
    if savedir[-1] == '':
        savedir = savedir[:-1]
    project, name = '/'.join(savedir[:-1]), savedir[-1]
    
    makedirs(project, exist_ok=True)
    
    # check if save location (project / name) exists, if so append number
    savedir = join(project, name)
    
    if isdir(savedir) and not exist_ok:
        for i in range(2, 100):
            new_savedir = savedir + str(i)
            if not isdir(new_savedir):
                savedir = new_savedir                
                break
                
            if i == 99:
                raise Exception(f'too many models of the name {name} in project {project}')
         
    # switch to the directory with the yolov5 train.py script
    starting_dir = getcwd()
    chdir(yolodir)
        
    # if the training hangs on destroying the process, switch torch.distributed.run to torch.distributed.launch
    if n_devices == 1:
        bash_command = 'python'
    else:
        bash_command = f'python -m torch.distributed.run --nproc_per_node {n_devices}'
    
    bash_command += f' train.py --batch {batch} --weights {weights} --epochs ' + \
                    f'{epochs} --device {device} --project {project} --name {name} --img {img} --hyp {hyp} --save-period {save_period} ' + \
                    f'--data {data} --patience {patience}'
    if bn:
        bash_command += ' --sync-bn'
        
    if exist_ok:
        bash_command += ' --exist-ok'
    
    process = Popen(bash_command.split(' '))
    process.wait()
    
    # change back to repository
    chdir(starting_dir)
