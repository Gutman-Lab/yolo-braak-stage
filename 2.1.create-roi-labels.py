# Create a label text file for each annotated ROI, adding consensus ROIs for testing
from colorama import Fore, Style
from pandas import read_csv, DataFrame
from geopandas import GeoDataFrame
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from argparse import ArgumentParser

from os import makedirs
from os.path import join, isfile

from nft_helpers.utils import (
    load_yaml, im_to_txt_path, print_opt, save_to_txt, delete_file
)
from nft_helpers.roi_utils import create_roi_labels, select_consensus_labels
from nft_helpers.box_and_contours import line_to_xys, xys_to_line


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--iou-thr', type=float, default=0.25, 
                        help='IoU threshold used to match annotations between annotators.')
    
    return parser.parse_args()


def match_annotations(df: DataFrame, label: str, subset_group: str, compare_group: str, threshold: float = 0.5, 
                      bg_label: str = 'background') -> DataFrame:
    """Match annotations given a list of annotation data in dataframe format.
    
    Args:
        df: Annotations style dataframe.
        label: Column in df parameter with the label.
        subset_group: Column to check annotations against, same group.. 
        compare_group: Column to check annotations against, different group.
        threshold: IoU threshold when matching boxes.
        bg_label : Label to give when no matching box by annotator.
        
    Returns:
        output_df: Each matching set of annotations is a row, contains the metadata from the input df, selected 
            randomly from one of the matching annotations in a set. Two new columns are added: labels which
            includes the labels separated by semi-colon and the groups which is the same for all rows and 
            contains the group order for the labels.
    
    """
    # add a geometry column and convert to geodataframe
    geodf = GeoDataFrame(df.copy())
    
    for i, r in geodf.iterrows():
        geometry = line_to_xys(r.box_coords)
        x1, y1, x2, y2 = geometry[0, 0], geometry[0, 1], geometry[1, 0], geometry[1, 1]
        geodf.loc[i, 'geometry'] = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    
    # for each annotation add a row
    labels = []
    metadata_idx = []  # return a matching dataframe with the metadata for the matching points
    
    # get a list of compare groups (this will be the columns of the output dataframe)
    groups = geodf[compare_group].unique()
    
    # check each subset group
    for subset in tqdm(geodf[subset_group].unique()):
        subset_df = geodf[geodf[subset_group] == subset]
        
        checked_idx = []
        
        # loop through each annotation in subset
        for i, r in subset_df.iterrows():
            # skip this annotation if it already has been matched
            if i in checked_idx:
                continue
            
            checked_idx.append(i)
            
            # compare only against annotations not of the same compare group
            compare_df = subset_df[(subset_df[compare_group] != r[compare_group]) & (~subset_df.index.isin(checked_idx))]
            comapre_df = GeoDataFrame(compare_df)
            
            # calculate the IoU between this annotation (r) and all others of this subset and different compare group
            intersection = compare_df.intersection(r.geometry)
            union = compare_df.union(r.geometry)
            ious = intersection.area.divide(union.area)
            
            # threshold by IoUs
            matched = compare_df[ious > threshold]
            
            row = []  # for this match, add the labels as string
            r = r.copy()
            
            for gp in groups:
                # add the label for the annotator in loop
                if gp == r[compare_group]:
                    row.append(r[label])
                else:
                    gp_rows = matched[matched[compare_group] == gp]

                    n_matches = len(gp_rows)
                    if n_matches == 1:
                        matched_r = gp_rows.iloc[0]
                        row.append(matched_r[label])
                        checked_idx.append(matched_r.name)
                    elif n_matches > 1:
                        # select the match with highest IoU
                        matched_ious = ious[gp_rows.index]
                        highest_iou = matched_ious.max()

                        # get the dataframe index for the highest iou
                        highest_idx = matched_ious[matched_ious == highest_iou].index[0]

                        # select this index from matched rows
                        matched_r = gp_rows.loc[highest_idx]

                        row.append(matched_r[label])
                        checked_idx.append(matched_r.name)
                    else:
                        row.append(bg_label)
            
            labels.append(';'.join(row))
            metadata_idx.append(i)
    
    output_df = df.loc[metadata_idx].reset_index(drop=True)
    output_df['labels'] = labels
    output_df['groups'] = [';'.join(groups)] * len(output_df)
    
    return output_df


def create_roi_labels(df, label_map=None, shift=None, box_col='box_coords', label_col='label', save_filepath=None):
    """Given a dataframe of box annotation data, create label text file. All rows are assumed to be from the same ROI / image.
    
    INPUT
    -----
    df : dataframe
        each row an annotation in the same region of an image
    label_map : dict (default is None)
        map labels to int values, if None then all unique labels are alphabetized and given int label by that order
    shift : tuple (default is None)
        shift all box coordinates by this tuple of length two (x shift, y shift)
    box_col : str (default='box_coords')
        column in df with the coordinates of the annotations (boxes)
    label_col : str (default='label')
        column in df with the annotation label
    save_filepath : str (default is None)
        save labels to text file
        
    RETURN
    ------
    labels : str
        each line of the string contains the label, x1, y1, x2, y2 coordinates where point 1 is the top left corner of the annotation
        object and point 2 is the bottom right corner
    
    """
    labels = ''
    
    if save_filepath is not None and not save_filepath.endswith('.txt'):
        raise Exception('save filepath parameter should be a text file extension')
    
    # if label map is None, map them to int alphabetically
    if label_map is None:
        unique_labels = sorted(df[label_col].unique().tolist())
        
        label_map = {}
        for i, lb in enumerate(unique_labels):
            label_map[lb] = i
    else:
        # remove any annotations that are not in the map
        df = df[df[label_col].isin(list(label_map.keys()))]
        
    for _, r in df.iterrows():
        # convert the box coordinates and shift them if needed
        box_coords = line_to_xys(r[box_col])
        
        if shift is not None:
            box_coords += shift
            
        box_coords = xys_to_line(box_coords)
        
        labels += f'{label_map[r[label_col]]} {box_coords}\n'

    if len(labels):
        labels = labels[:-1]
        
        # save labels to text file
        if save_filepath is not None:
            save_to_txt(save_filepath, labels)
    else:
        # no longer nay labels so should remove.
        delete_file(save_filepath)
        
    return labels


def main():
    """Main function."""
    print(Fore.BLUE, Style.BRIGHT + 'Creating label text files for ROIs.\n', 
          Style.RESET_ALL)
    
    cf = load_yaml()
    opt = parse_opt()
    print_opt(opt)
    
    label_map = {'Pre-NFT': 0, 'iNFT': 1}

    # save location for roi labels
    save_dir = join(cf.datadir, 'rois/annotated-cohort/labels')
    makedirs(save_dir, exist_ok=True)
    
    rois_labels = []  # list of roi data
    
    print(Fore.CYAN, Style.BRIGHT, 'Creating ROI label text files.', 
          Style.RESET_ALL)
    
    # Loop through each ROI image
    rois_df = read_csv(join(cf.codedir, 'csvs/rois.csv'))
    annotations = read_csv(join(cf.codedir, 'csvs/annotations.csv'))
    
    for _, r in tqdm(rois_df.iterrows(), total=len(rois_df)):
        # annotations in this ROI
        roi_annotations = annotations[annotations.roi_im_path == r.roi_im_path]
        
        # images/filename.png --> labels/filename.txt
        txt_filepath = im_to_txt_path(r.roi_im_path)
        r['roi_labels'] = txt_filepath
        
        # add this ROI to labeled-rois
        rois_labels.append(r)
                
        # Create ROI label.
        if len(roi_annotations):
            _ = create_roi_labels(
                roi_annotations, label_map=label_map, 
                shift=(-r.roi_im_left, -r.roi_im_top), 
                save_filepath=txt_filepath
            )
        
    print(Fore.CYAN, Style.BRIGHT, '\nCreate consensus ROI label text files.',
          Style.RESET_ALL)
    
    # Get the matching annotations for inter-anntotator agreement rois.
    matching_annotations_fp = join(cf.codedir, 'csvs/matching-annotations.csv')
    
    if isfile(matching_annotations_fp):
        matching_annotations = read_csv(matching_annotations_fp)
    else:
        print(
            Fore.CYAN, Style.BRIGHT, '   Matching annotations from different '
            'annotators.\n\n', Style.RESET_ALL
        )
        
        matching_annotations = match_annotations(
            annotations[annotations.roi_group == 'ROIv2'], 'label', 
            'url_to_parent_roi', 'annotator', threshold=opt.iou_thr
        ).drop(
            columns=['annotator', 'wsi_id', 'Braak_stage', 
                     'annotator_experience', 'url_to_roi', 'label', 'status']
        )

        matching_annotations.to_csv(matching_annotations_fp, index=False)
        
    # extract a dataframe of labels for each annotator
    annotators = matching_annotations.iloc[0].groups.split(';')

    matching_labels = []
        
    for annotation_labels in matching_annotations.labels:
        # convert annotations to int format
        int_labels = []

        for lb in annotation_labels.split(';'):
            if lb == 'background':
                int_labels.append(0)
            else:
                int_labels.append(label_map[lb] + 1)

        matching_labels.append(int_labels)

    matching_labels = DataFrame(data=matching_labels, columns=annotators)
        
    # subset to only experts
    experts = [ann for ann in annotators if ann.startswith('expert')]
    matching_labels = matching_labels[experts]
    
    # remove annotations that had no expert annotations
    expert_idx = np.sum(matching_labels.to_numpy(), axis=1) != 0
    matching_annotations = matching_annotations[expert_idx]
    matching_labels = matching_labels[expert_idx]
    
    # set of strategies to create consensus labels, multiple version of this will exist
    strategies = list(range(1, len(matching_labels.columns)+1))+ ['majority']
    
    # Loop by the url of the inter-annotator agreement rois, this will list exactly the 15 rois.    
    for url_to_parent_roi in tqdm(rois_df[rois_df.roi_group == 'ROIv2'].url_to_parent_roi.unique()):
        r = rois_df[rois_df.url_to_parent_roi == url_to_parent_roi].iloc[0]
        
        # modify values specific for annotator to reflect that this is a consensus roi
        r['wsi_id'] = r['parent_id']
        r['url_to_roi'] = r['url_to_parent_roi']
        r['Braak_stage'] = -1
        r['parent_id'] = ''
        r['url_to_parent_roi'] = ''
        r['annotator_experience'] = 'expert-consensus'
        
        # subset to annotations and labels for this ROI
        roi_annotations = matching_annotations[matching_annotations.url_to_parent_roi == url_to_parent_roi].reset_index(drop=True).copy()
        roi_labels = matching_labels[matching_annotations.url_to_parent_roi == url_to_parent_roi]
        
        # use this array to calculate consensus label based on various strategies
        consensus_labels = select_consensus_labels(roi_labels, strategy=strategies)

        for strat in strategies:
            strat_r = r.copy()
            
            annotator = f'consensus-n{strat}' if isinstance(strat, int) else f'consensus-{strat}'
            strat_r['annotator'] = annotator
            
            strat_annotations = roi_annotations.copy()
            
            strat_annotations['label'] = [lb - 1 for lb in consensus_labels[strat]]
            
            # remove rows that are not labeled in this strat
            strat_annotations = strat_annotations[strat_annotations.label >= 0]
            
            txt_filepath = join(
                    save_dir, f'{annotator}_id-{r.wsi_id}_left-{r.roi_im_left}_top-{r.roi_im_top}_right-{r.roi_im_right}_bottom-' + \
                   f'{r.roi_im_bottom}.txt'
            )
            
            # add the filepath to the label text file for this ROI
            strat_r['roi_labels'] = txt_filepath

            rois_labels.append(strat_r)
            
            if len(roi_annotations):
                _ = create_roi_labels(strat_annotations, shift=(-r.roi_im_left, -r.roi_im_top), save_filepath=txt_filepath)
    
    rois_labels = DataFrame(data=rois_labels).reset_index(drop=True)
    
    # save to csv directory
    rois_labels.to_csv(join(cf.codedir, 'csvs/labeled-rois.csv'), index=False)
    
    print(Fore.GREEN, Style.BRIGHT + '\nDone!', Style.RESET_ALL)

    
if __name__ == '__main__':
    main()
