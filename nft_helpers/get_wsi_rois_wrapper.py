from multiprocessing import Pool
from typing import Union, List, Tuple
from pandas import DataFrame, concat
from girder_client import GirderClient
from . import get_wsi_rois


def get_wsi_rois_wrapper(
    gc: GirderClient, 
    item_ids: List[str],
    annotation_groups: Union[str, List[str]] = None,
    roi_groups: Union[str, List[str]] = 'ROI',
    docs: Union[str, List[str]] = None,
    verbose: bool = False,
    fill: Tuple[int] = (114, 114, 114),
    save_dir: str = '.',
    mag: float = None,
    box_default: int = 100,
    nproc: int = 3,
    notebook: bool = False
) -> DataFrame:
    """Download ROI annotations as images and associated annotations. The image
    is the smallest bounding box on the ROI. This is a wrapper that can run 
    through multiple images using parallel processing.
    
    Args:
        gc: Girder client.
        item_ids: List of image ids.
        annotation_groups: Groups for annotations in ROIs. These will be 
            converted to int labels based on order provided. If ROIs don't have
            annotations of interest, then can leave as None.
        roi_groups: Groups of ROIs.
        docs: Annotation documents to check, if None then all are checked.
        verbose: Print out warnings.
        fill: Fill regions not in the ROI area.
        save_dir: Directory to save files.
        mag: Magnification to get image at. If None than it will be taken at the
            scan magnification.
        box_default: Default box size to draw when watershed fails to get an 
            object.
        notebook: Select which type of tqdm to use.
            
    Returns:
        Metadata for ROIs in a dataframe format.
        
    """
    if nproc > 5:
        raise Exception("nproc can't be over 5 to avoid DSA crashes.")
    
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        
    with Pool(nproc) as pool:
        jobs = [
            pool.apply_async(
                func=get_wsi_rois, 
                args=(gc, 
                      item_id, annotation_groups, roi_groups, docs, verbose,
                      fill, save_dir, mag, box_default,)) 
            for item_id in item_ids]
        
        roi_df = [job.get() for job in tqdm(jobs)]
        
    return concat(roi_df, ignore_index=True)
