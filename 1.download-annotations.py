# Download the annotations from DSA and create starting image and other files.
from pandas import read_csv, DataFrame, concat
import numpy as np
from argparse import ArgumentParser
import multiprocessing as mp
from colorama import Fore, Style
from tqdm import tqdm

from os import makedirs
from os.path import join, isfile, isdir

from nft_helpers.watershed import watershed_extraction
from nft_helpers.girder_dsa import (
    login, get_rectangle_element_coords, get_region, 
    get_collection_id, get_items, get_tile_metadata, backup_annotations
)
from nft_helpers.box_and_contours import line_to_xys, xys_to_line, pt_in_polygon
from nft_helpers.utils import (
    imwrite, load_yaml, print_opt, imread, load_json, create_cases_df, 
    create_wsis_df, get_filename
)


def parse_opt():
    """CLIs"""
    parser = ArgumentParser()
    parser.add_argument(
        '--img', type=int, default=500, 
        help='Size of image saved around annotations.'
    )
    parser.add_argument(
        '--nproc', type=int, default=3, 
        help='Parallel processes.'
    )
    parser.add_argument('--roi-groups', type=str, nargs='+', help='ROI groups.',
                        default=['ROIv1', 'ROIv2', 'test-roi'])
    parser.add_argument('--update-annotations', action='store_true',
                        help='Pull annotations from HistomicsUI.')
    opt = parser.parse_args()
    
    if opt.nproc > 5:
        raise Exception("Keep nproc CLI at 5 or less to avoid crashing DSA.")
        
    return opt


def process_wsi(r, annotations, gc, dsa_url, imsize, rois_dir, 
                annotations_im_dir, current_annotations, roi_groups):
    """Process a WSI by saving images of the ROI and point annotations."""
    # Split the annotations by groups.
    roi_elements = []
    pt_elements = []
    complete_elements = []

    for anndoc in annotations:
        for element in anndoc['annotation']['elements']:
            if element['type'] == 'point' and \
               element['group'] in ['Pre-NFT', 'iNFT']: 
                pt_elements.append(element)
            elif element['type'] == 'rectangle' and \
                 element['group'] in roi_groups:
                roi_elements.append(element)
            elif element['type'] == 'point' and \
                 element['group'] == 'ROI COMPLETE':
                complete_elements.append(element)

    # skip if no ROIs
    if not len(roi_elements):
        return [], []
    
    rois_df = []
    annotations_df = []
    
    # Half size of image.
    half_img = int(imsize / 2)
        
    # check each ROI
    for roi_element in roi_elements:
        roi_group = roi_element['group']
        roi_corners = get_rectangle_element_coords(roi_element)
        
        # calculate the coordinates of the smallest bounding box of the ROI
        roi_im_left, roi_im_top = np.min(roi_corners, axis=0)
        roi_im_right, roi_im_bottom = np.max(roi_corners, axis=0)
        
        # this is the width and height of the ROI, NOT the ROI bounding box image
        roi_width, roi_height = float(roi_element['width']), float(roi_element['height'])
        
        # create URL links for the ROI: to the this WSI and also its pair parent image (i.e. inference cohort)
        url_to_roi = join(
            dsa_url, 
            f"histomics#?image={r.wsi_id}&bounds={roi_im_left}%2C{roi_im_top}%2C{roi_im_right}%2C{roi_im_bottom}%2C0"
        )
        url_to_parent_roi = join(
            dsa_url, 
            f"histomics#?image={r.parent_id}&bounds={roi_im_left}%2C{roi_im_top}%2C{roi_im_right}%2C{roi_im_bottom}%2C0"
        )
        
        # catch any incomplete ROIs and stop the code if found - all should be done!
        # check ROI complete only for ROIv1, ROIv2, and test-roi, which are the ROIs that are annotated!
        if r.cohort in ('Annotated-Cohort', 'Inference-Cohort-2'):
            roi_incomplete = True

            for complete_i, complete_element in enumerate(complete_elements):
                if pt_in_polygon(int(complete_element['center'][0]), int(complete_element['center'][1]), roi_corners):
                    roi_incomplete = False
                    break

            if roi_incomplete:
                print(Fore.YELLOW + Style.BRIGHT, f'    Skipping incomplete ROI ({url_to_roi})', Style.RESET_ALL)
                continue
            
            del complete_elements[complete_i]  # remove this complete annotation
    
        # path of ROI image format: {wsi id}_left-{top left coordinate of ROI bounding box}_top-{bottom right "..."}
        # the coordinates are relative to the image (i.e. [0,0] is the top left corner of WSI)
        roi_im_path = join(rois_dir, f'{r.annotator}_id-{r.wsi_id}_left-{roi_im_left}_top-{roi_im_top}_right-{roi_im_right}_bottom-{roi_im_bottom}.png')
        
        # add ROI entry to data
        roi_corners_str = xys_to_line(roi_corners)
        rois_df.append([
            r.wsi_name, r.case, r.annotator, r.wsi_id, r.parent_id, r.Braak_stage, r.region, r.annotator_experience, 
            r.scan_mag, roi_im_path, roi_group, roi_im_left, roi_im_top, roi_im_right, roi_im_bottom, url_to_roi, 
            url_to_parent_roi, roi_corners_str, roi_width, roi_height, r.cohort
        ])
        
        roi_im = None  # set to None for this ROI
        
        # save the ROI image if it does not exist
        if not isfile(roi_im_path):
            roi_im = get_region(gc, r.wsi_id, roi_im_left, roi_im_top, right=roi_im_right, bottom=roi_im_bottom)
            imwrite(roi_im_path, roi_im) 
            
        # loop through all hp-tau inclusion annotations
        idx_to_remove = []  # after this ROI, remove the points that were here to avoid including a point multiple times
        
        for pt_i, pt_element in enumerate(pt_elements):
            pt_x, pt_y = int(pt_element['center'][0]), int(pt_element['center'][1])

            # check if point is in ROI
            if pt_in_polygon(pt_x, pt_y, roi_corners):
                idx_to_remove.append(pt_i)  # remove this point after

                # calculate the coordinates for image around this point
                im_left, im_right = pt_x - half_img, pt_x + half_img
                im_top, im_bottom = pt_y - half_img, pt_y + half_img

                # create the path of the image to save or check local copy
                im_path = join(annotations_im_dir, f'{r.wsi_id}-left_{im_left}-top_{im_top}-right_{im_right}-bottom_{im_bottom}.png')
                    
                im = None  # seed this variable for speed 
                
                if not isfile(im_path):
                    # need to get point image from ROI image or from DSA (if it is past the edge)
                    if im_left < roi_im_left or im_top < roi_im_top or im_right > roi_im_right or im_bottom > roi_im_bottom:
                        im = get_region(gc, r.wsi_id, im_left, im_top, right=im_right, bottom=im_bottom).astype(np.uint8)
                    else:
                        # check that ROI image is loaded
                        if roi_im is None:
                            roi_im = imread(roi_im_path)
                            
                        # get the image from the ROI
                        x1, y1 = im_left - roi_im_left, im_top - roi_im_top
                        im = roi_im[y1:y1+imsize, x1:x1+imsize]
                        
                    imwrite(im_path, im)  # save the image
                    
                # check if box boundary needs to be obtained from watershed or retrieved from previous backup file
                fn = get_filename(im_path)
            
                if fn in current_annotations:
                    box_coords = current_annotations[fn]['box_coords']
                    status = current_annotations[fn]['status']
                else:
                    # estimate box boundary with watershed
                    if im is None:
                        im = imread(im_path)
                    
                    box_coords = watershed_extraction(im, box=True, as_str=True)
                    status = 'needs checking'
                    
                    # if watershed failed - then do a dummy box of half the size
                    if len(box_coords):
                        # convert to string format
                        box_coords = xys_to_line(line_to_xys(box_coords) + (im_left, im_top))
                    else:
                        dummy_size = int(imsize / 4)
                        box_coords = f'{dummy_size + im_left} {dummy_size + im_top} {dummy_size+ half_img + im_left} {dummy_size + half_img + im_top}'

                # add custom URL to reach HistomicsUI view of point
                url_to_im = join(
                    dsa_url, f'histomics#?image={r.wsi_id}&bounds={im_left}%2C{im_top}%2C{im_right}%2C{im_bottom}%2C0'
                )

                # add this point data
                annotations_df.append([
                    r.wsi_name, r.case, r.annotator, r.wsi_id, r.parent_id, r.Braak_stage, r.region, 
                    r.annotator_experience, r.scan_mag, roi_group, roi_im_path, roi_im_left, roi_im_top, roi_im_right, 
                    roi_im_bottom, url_to_roi, url_to_parent_roi, roi_corners_str, roi_width, roi_height, 
                    pt_element['group'], im_left, im_top, im_right, im_bottom, box_coords, pt_x, pt_y, im_path, 
                    url_to_im, status, r.cohort
                ]) 

        # remove the points from this ROI from list
        for idx in sorted(idx_to_remove, reverse=True):
            del pt_elements[idx]
        
    return rois_df, annotations_df


def main():
    """Main function."""
    print(Fore.BLUE, Style.BRIGHT, 'Downloading ROI and annotation images plus '
          'metadata.\n', Style.RESET_ALL)
    
    opt = parse_opt()  # cli arguments
    cf = load_yaml()  # configuration variables
    print_opt(opt)
    
    if opt.nproc > 5:
        raise Exception(
            "nproc can't be greater than 5, otherwise DSA instance might crash."
        )
        
    print(
        Fore.YELLOW + Style.BRIGHT, '  Please note that this script does not '
        'overwrite existing images, delete annotations and rois dir to start '
        'from scratch!', Style.RESET_ALL
    )
    
    # create directories
    csvs_dir = join(cf.codedir, 'csvs')
    rois_dir = join(cf.datadir, 'rois/annotated-cohort/images')
    annotations_im_dir = join(cf.datadir, 'annotations')

    makedirs(csvs_dir, exist_ok=True)
    makedirs(rois_dir, exist_ok=True)
    makedirs(annotations_im_dir, exist_ok=True)
    
    # authenticate girder client 
    gc = login(join(cf.dsaURL, 'api/v1'), username=cf.user, 
               password=cf.password)
    
    # Load the annotations from file or pull them from DSA.
    annotations_path = join(cf.codedir, 'dsa-annotations.json')
    
    if isfile(annotations_path) and not opt.update_annotations:
        print(Fore.YELLOW + Style.BRIGHT, '\n  Reading annotations from file.',
              Style.RESET_ALL)
        annotations = load_json(annotations_path)
    else:
        print(Fore.CYAN, Style.BRIGHT, f'\n   Downloading annotations to local '
              'file.', Style.RESET_ALL)
        
        # Get the metadata and annotations for WSIs in project.
        annotations = backup_annotations(
            gc, cf.collection, exts=('.svs', '.ndpi'), docs=['annotations'], 
            save_fp=annotations_path
        )
        
    # Get information about each case, includes all metadata.
    case_df, parent_id_map = create_cases_df(annotations['Inference-Cohort-1'])
    case_df['cohort'] = ['Inference-Cohort-1'] * len(case_df)
    cases_df = [case_df]
    case_df = create_cases_df(annotations['Inference-Cohort-2'])[0]
    case_df['cohort'] = ['Inference-Cohort-2'] * len(case_df)
    cases_df.append(case_df)
    case_df = create_cases_df(annotations['External-Cohort'])[0]
    case_df['cohort'] = ['External-Cohort'] * len(case_df)
    cases_df.append(case_df)
    cases_df = concat(cases_df, ignore_index=True)
    cases_df.to_csv(join(csvs_dir, 'cases.csv'), index=False)
    
    # Get information about each WSI - this only includes minimal data.
    meta_keys = ['case', 'annotator', 'Braak_stage', 'annotator_experience', 
                 'region']
    wsi_df = create_wsis_df(annotations['Annotated-Cohort'], 
                            parent_id_map=parent_id_map, meta_keys=meta_keys)
    wsi_df['cohort'] = ['Annotated-Cohort'] * len(wsi_df)
    wsis_df = [wsi_df]
    wsi_df = create_wsis_df(annotations['Inference-Cohort-1'], 
                            meta_keys=meta_keys)
    wsi_df['cohort'] = ['Inference-Cohort-1'] * len(wsi_df)
    wsis_df.append(wsi_df)
    wsi_df = create_wsis_df(annotations['Inference-Cohort-2'], 
                            meta_keys=meta_keys)
    wsi_df['cohort'] = ['Inference-Cohort-2'] * len(wsi_df)
    wsis_df.append(wsi_df)
    wsi_df = create_wsis_df(annotations['External-Cohort'], meta_keys=meta_keys)
    wsi_df['cohort'] = ['External-Cohort'] * len(wsi_df)
    wsis_df.append(wsi_df)
    wsis_df = concat(wsis_df, ignore_index=True)
    wsis_df.to_csv(join(csvs_dir, 'wsis.csv'), index=False)
    
    # Load annotations.
    anns_csv_path = join(csvs_dir, 'annotations.csv')
    backup_csv_fp = join(csvs_dir, 'annotations-backup.csv')
    
    # Get coordinates of boxes and status from previous file.
    if isfile(backup_csv_fp):
        current_annotations = {}
        for _, r in read_csv(backup_csv_fp).iterrows():
            current_annotations[get_filename(r.im_path)] = {
                'box_coords': r.box_coords, 'status': r.status
            }
    else:
        current_annotations = {}
        
    # compile the annotations into a single dict
    all_annotations = {}
    
    for items in annotations.values():
        all_annotations.update(items)
                
    # Get annotations from all cohorts of interest.
    pool = mp.Pool(opt.nproc)
    jobs = [pool.apply_async(
        func=process_wsi, 
        args=(r, 
              all_annotations[r.wsi_id]['annotations'], 
              gc, 
              cf.dsaURL, 
              opt.img, 
              rois_dir, 
              annotations_im_dir, 
              current_annotations,
              opt.roi_groups,
             )) for _, r in wsis_df[wsis_df.cohort.isin(('Annotated-Cohort', 'Inference-Cohort-2', 'External-Cohort'))].iterrows()
           ]
    pool.close()
        
    print(Fore.CYAN, Style.BRIGHT, '\n  Saving ROI images and annotation images.', Style.RESET_ALL)
    rois_df = []
    annotations_df = []
    for job in tqdm(jobs):
        results = job.get()
        rois_df.extend(results[0])
        annotations_df.extend(results[1])
        
    # create the dataframes and save to file
    rois_df = DataFrame(
        data=rois_df, 
        columns=[
            'wsi_name', 'case', 'annotator', 'wsi_id', 'parent_id', 
            'Braak_stage', 'region', 'annotator_experience', 'scan_mag', 
            'roi_im_path', 'roi_group', 'roi_im_left', 'roi_im_top', 
            'roi_im_right', 'roi_im_bottom', 'url_to_roi', 'url_to_parent_roi', 
            'roi_corners', 'roi_width', 'roi_height', 'cohort'
        ]
    )
    annotations_df = DataFrame(
        data=annotations_df,
        columns=[
            'wsi_name', 'case', 'annotator', 'wsi_id', 'parent_id', 
            'Braak_stage', 'region', 'annotator_experience', 'scan_mag', 
            'roi_group', 'roi_im_path', 'roi_im_left', 'roi_im_top', 
            'roi_im_right', 'roi_im_bottom', 'url_to_roi', 'url_to_parent_roi', 
            'roi_corners', 'roi_width', 'roi_height', 'label', 'im_left', 
            'im_top', 'im_right', 'im_bottom', 'box_coords', 'pt_x', 'pt_y', 
            'im_path', 'url_to_im', 'status', 'cohort'
        ]
    )
    
    rois_df.to_csv(join(csvs_dir, 'rois.csv'), index=False)
    annotations_df.to_csv(join(csvs_dir, 'annotations.csv'), index=False)
    print(Fore.GREEN, Style.BRIGHT, '\nDone!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
