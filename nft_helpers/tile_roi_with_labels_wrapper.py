# Tile multiple ROIs with labels using paralell processing.
from multiprocessing import Pool
from typing import List, Union, Tuple
from pandas import DataFrame, concat
from . import tile_roi_with_labels


def tile_roi_with_labels_wrapper(
    fps: List[str], save_dir: Union[str, List[str]], tile_size: int = 1280, 
    stride: int = None, boundary_thr: float = 0.2, nproc: int = 10,
    fill: Tuple[int] = (114, 114, 114), box_thr: float = 0.5, 
    notebook: bool = False
) -> DataFrame:
    """Tile an ROI image with labels.
    
    Args:
        fps: Image filepaths, should be in an '/images/ directory'.
        save_dir: Either a single location to create images and labels dir or
            a list of directories for each filepath passed.
        tile_size: Size of tile, uses square tiles only.
        stride: Stride to use when tiling, if None then it is set equal to 
            tile_size (no overlap between tiles).
        boundary_thr: If ROI has a boundary (for rotated ROIs) then a tile must
            have sufficient area in boundary to be included (0.0 - 1.0).
        nproc: Number of parallel processed to use.
        fill: RGB when padding image.
        box_thr: Area threshold of box that must be in a tile.
        notebook: Select which type of tqdm to use.
       
    Returns:
        Metadata of tiles saved.
        
    """
    if isinstance(save_dir, (list, tuple)):
        if len(save_dir) != len(fps):
            raise Exception('If save_dir is a tuple / list, then it must be the '
                            'same length as the number of filepaths.')
    else:
        save_dir = [save_dir] * len(fps)
    
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        
    with Pool(nproc) as pool:
        jobs = [
            pool.apply_async(
                func=tile_roi_with_labels, 
                args=(fp, sd, tile_size, stride, boundary_thr, fill, box_thr,)) 
            for fp, sd in zip(fps, save_dir)]
        
        tile_df = [job.get() for job in tqdm(jobs)]
        
    return concat(tile_df, ignore_index=True)
