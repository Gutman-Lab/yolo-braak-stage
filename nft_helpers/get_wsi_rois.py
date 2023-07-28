# Example for downloading ROIs from a single WSI.
from girder_client import GirderClient
from typing import Union, List, Tuple
from colorama import Fore, Style
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon, Point
from geopandas import GeoDataFrame
from pandas import DataFrame

from .girder_dsa import (
    get_annotations_documents, get_rectangle_element_coords, get_region,
    get_tile_metadata
)
from .utils import imwrite, imread
from .watershed import watershed_extraction

from os import makedirs
from os.path import join, isfile, isdir


def get_wsi_rois(
    gc: GirderClient, 
    item_id: str,
    annotation_groups: Union[str, List[str]] = None,
    roi_groups: Union[str, List[str]] = 'ROI',
    docs: Union[str, List[str]] = None,
    verbose: bool = True,
    fill: Tuple[int] = (114, 114, 114),
    save_dir: str = '.',
    mag: float = None,
    box_default: int = 100
) -> DataFrame:
    """Download ROI annotations as images and associated annotations. The image
    is the smallest bounding box on the ROI.
    
    Args:
        gc: Girder client.
        item_id: DSA ID of image.
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
            
    Returns:
        Metadata for ROIs in a dataframe format.
    
    """
    img_dir = join(save_dir, 'images')
    label_dir = join(save_dir, 'labels')
    boundary_dir = join(save_dir, 'boundaries')
        
    # Add some metadata.
    item = gc.getItem(item_id)
    meta = item['meta'] if 'meta' in item else {}
    case = meta['case'] if 'case' in meta else ''
    stage = meta['Braak_stage'] if 'Braak_stage' in meta else ''
    region = meta['region'] if 'region' in meta else ''
    
    if annotation_groups is None:
        annotation_groups = []
    elif isinstance(annotation_groups, str):
        annotation_groups = [annotation_groups]
    if isinstance(roi_groups, str):
        roi_groups = [roi_groups]
    if isinstance(docs, str):
        docs = [docs]
        
    # Catch some potential errors.
    if len(annotation_groups) != len(set(annotation_groups)):
        raise Exception('Annotation groups must be unique.')
    
    for gp in roi_groups:
        if gp in annotation_groups:
            raise Exception("Annotation and ROI groups can't overlap.")
    
    df  = []  # track ROI metadata
    
    # Get the annotations and compile as a GeoDataframe.
    roi_els = []
    ann_df = []
    
    docs = get_annotations_documents(
        gc, item_id, groups=annotation_groups + roi_groups, docs=docs
    )
    
    # Default behaviour: image to take for watershed is 3 times the default
    # size of box when watershed fails.
    ws_size = box_default * 3
    
    for doc in docs:
        for el in doc['annotation']['elements']:
            if el['group'] in roi_groups:
                if el['type'] in ('rectangle', 'polyline'):
                    roi_els.append(el)
                elif verbose:
                    print(
                        Fore.YELLOW, Style.BRIGHT, 
                        f"ROI group of wrong type ({el['type']})",
                        Style.RESET_ALL
                    )
            elif el['group'] in annotation_groups:
                if el['type'] == 'point':
                    x, y = np.array(el['center'][:2], dtype=int)
                    
                    # Use watershed to get the bounding box.
                    left = x - int(ws_size / 2)
                    top = y - int(ws_size / 2)
                    
                    img = get_region(gc, item_id, left=left, top=top, 
                                     width=ws_size, height=ws_size)
                    
                    box = watershed_extraction(img, box=True, as_str=False)
                    
                    if len(box):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = box_default, box_default
                        
                elif el['type'] == 'rectangle':
                    x, y = np.array(el['center'][:2], dtype=int)
                    w, h = int(el['width']), int(el['height'])
                else:
                    if verbose:
                        print(
                            Fore.YELLOW, Style.BRIGHT, 
                            f"Annotation group of wrong type ({el['type']})",
                            Style.RESET_ALL
                        )
                    continue
                    
                ann_df.append([annotation_groups.index(el['group']), x, y, w, h,
                               Point(x, y)])
                
    ann_df = GeoDataFrame(ann_df, columns=['label', 'x', 'y', 'w', 'h', 
                                           'geometry'])

    if not len(roi_els):
        if verbose:
            print(Fore.YELLOW, Style.BRIGHT, 'No ROI annotations found', 
                  Style.RESET_ALL)
        return DataFrame(
            df, 
            columns=['fp', 'x', 'y', 'mag', 'group', 'sf', 'w', 'h', 
                     'wsi_name', 'wsi_id', 'case', 'region', 'Braak_stage']
        )    
    elif verbose:
        print(f'\nFound {len(roi_els)} ROI & {len(ann_df)} annotations.')
        
    # get scale factor to go from scan mag to mag to get image
    if mag is None:
        mag = get_tile_metadata(gc, item_id)['magnification']
        sf = 1
    else:
        sf = mag / get_tile_metadata(gc, item_id)['magnification']
        
    # Create image dir
    makedirs(img_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    makedirs(boundary_dir, exist_ok=True)
        
    # Loop through each ROI
    for roi_el in roi_els:
        if roi_el['type'] == 'rectangle':
            # Get the coordinates
            roi_coords = get_rectangle_element_coords(roi_el)
        else:
            # Polyline coordinates
            roi_coords = np.array(roi_el['points'])[:, :2].astype(np.int32)
            
        # Add roi coordinates as a polygon in space
        roi_pol = Polygon(roi_coords)
        
        # get the min and max coords
        rx1, ry1 = np.min(roi_coords, axis=0)
        rx2, ry2 = np.max(roi_coords, axis=0)
        
        roi_w, roi_h = int((rx2 - rx1) * sf), int((ry2 - ry1) * sf)
                
        # format the filename and save path
        fn = f'{item_id}-x{rx1}y{ry1}.'
        img_fp = join(img_dir, fn + 'png')
        
        df.append([img_fp, rx1, ry1, mag, roi_el['group'], 1 / sf, roi_w, 
                   roi_h, item['name'], item_id, case, region, stage])
        
        # Either get the image from file or from new call.
        if isfile(img_fp):
            roi_img = imread(img_fp)
            roi_h, roi_w = roi_img.shape[:2]
        else:
            # Get the image.
            roi_img = get_region(
                gc, item_id, left=rx1, top=ry1, width=rx2-rx1, height=ry2-ry1,
                mag=mag
            )
            
            # Fill regions not in countour
            roi_coords = ((roi_coords - [rx1, ry1]) * sf).astype(int)
            boundary_mask = np.ones(roi_img.shape[:2])
            boundary_mask = cv.drawContours(
                boundary_mask, 
                [roi_coords],
                -1, 
                0,
                cv.FILLED
            )

            roi_img[boundary_mask == 1] = fill
            imwrite(img_fp, roi_img)  # save ROI image

            # Save boundary to text file.
            roi_h, roi_w = roi_img.shape[:2]

            with open(join(boundary_dir, fn + 'txt'), 'w') as fh:
                # Save a string of x, y points normalized to image width/height
                str_coords = ''

                for c in (roi_coords / [roi_w, roi_h]).flatten():
                    str_coords += f'{round(c, 4)} '

                fh.write(str_coords.strip())

        # Check annotations, center must be in ROI region
        roi_labels = ''

        # For all annotations in this ROI
        for i, r in ann_df[ann_df.within(roi_pol)].iterrows():
            xc, yc = (r.x - rx1) * sf / roi_w, (r.y - ry1) * sf / roi_h
            w, h = r.w * sf / roi_w, r.h * sf / roi_h

            roi_labels += f'{r.label} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n'

        if len(roi_labels):
            with open(join(label_dir, fn + 'txt'), 'w') as fh:
                fh.write(roi_labels.strip())
                
    return DataFrame(
        df, 
        columns=['fp', 'x', 'y', 'mag', 'group', 'sf', 'w', 'h', 'wsi_name', 
                 'wsi_id', 'case', 'region', 'Braak_stage']
    )
