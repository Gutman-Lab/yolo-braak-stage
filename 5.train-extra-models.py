# Train the models with only tiles from cleaned up ROIs.
from colorama import Fore, Style

from os import makedirs
from os.path import join

from nft_helpers.utils import load_yaml, print_opt
from nft_helpers.yolov5 import train_models


def main():
    """Main function"""
    print(Fore.BLUE, Style.BRIGHT, 'Train extra models.\n', Style.RESET_ALL)
    cf = load_yaml()

    # Set up parameters.
    hyp = '/workspace/code/hyps/hyp.yaml'
    epochs = 150
    patience = 20
    weights = 'yolov5m6.pt'
    img = 1280
    batch_per_device = 12
    device = None
    iou_thr = 0.4

    # Train models.
    model_dir = join(cf.datadir, 'models/model-assisted-labeling')
    makedirs(model_dir, exist_ok=True)
    
    train_models(
        join(cf.datadir, 'datasets/model-assisted-labeling/yamls-extras'), 
        model_dir, hyp=hyp, epochs=epochs, weights=weights, im_size=img, 
        device=device, batch_per_device=batch_per_device, iou_thres=iou_thr, 
        patience=patience
    )
    
    print(Fore.GREEN, Style.BRIGHT, 'Done!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
