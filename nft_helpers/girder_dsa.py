# Working with the DSA through the girder client
# functions:
# - login
# - get_rectangle_element_coords
# - recursive_search
# - clean_annotation_documents
# - get_annotations_documents
# - get_tile_metadata
# - get_collection_id
# - get_region
# - get_items
# - download_item_rois
# - download_rois
# - backup_annotations
from girder_client import GirderClient, HttpError
from copy import deepcopy
import numpy as np
from PIL import Image
from io import BytesIO
from pandas import DataFrame, concat
from tqdm import tqdm
import multiprocessing as mp
import json
from colorama import Fore, Style
from typing import Union, List

from .utils import imwrite, get_filename
from .box_and_contours import xys_to_line
from .watershed import watershed_extraction

from os import makedirs
from os.path import join, isfile


def login(apiurl, username=None, password=None):
    """Authenticate a girder client instance using username and password.
    Parameters
    ----------
    apiurl : str
        the DSA instance api url
    username : str (default: None)
        username to authenticate client with, if None then interactive authentication is used
    password : str (default: None)
        password to authenticate client with, if None then interactive authentication is used
    Return
    ------
    gc : girder_client.GirderClient
        authenticated girder client instance
    """
    gc = GirderClient(apiUrl=apiurl)

    if username is None or password is None:
        interactive = True
    else:
        interactive = False

    gc.authenticate(username=username, password=password, interactive=interactive)

    return gc


def _rotate_point_list(point_list, rotation, center=(0, 0)):
    """Rotate a list of x, y points around a center location.
    Adapted from: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/annotations_and_masks/annotation_and_mask_utils.py
    INPUTS
    ------
    point_list : list
        list of x, y coordinates
    rotation : int or float
        rotation in radians
    center : list
        x, y location of center of rotation
    RETURN
    ------
    point_list_rotated : list
        list of x, y coordinates after rotation around center
    """
    point_list_rotated = []

    for point in point_list:
        cos, sin = np.cos(rotation), np.sin(rotation)
        x = point[0] - center[0]
        y = point[1] - center[1]

        point_list_rotated.append((int(x * cos - y * sin + center[0]), int(x * sin + y * cos + center[1])))

    return point_list_rotated


def get_rectangle_element_coords(element):
    """Get the corner coordinate from a rectangle HistomicsUI element, can handle rotated elements.
    Adapted from: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/annotations_and_masks/annotation_and_mask_utils.py
    INPUTS
    ------
    element : dict
        rectangle element, in HistomicsUI format
    RETURN
    ------
    corner_coords : array
        array of shape [4, 2] for the four corners of the rectangle in (x, y) format
    """
    # element is a dict so prevent referencing
    element = deepcopy(element)

    # calculate the corner coordinates, assuming no rotation
    center_x, center_y = element['center'][:2]
    h, w = element['height'], element['width']
    x_min = center_x - w // 2
    x_max = center_x + w // 2
    y_min = center_y - h // 2
    y_max = center_y + h // 2
    corner_coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    # if there is rotation rotate
    if element['rotation']:
        corner_coords = _rotate_point_list(corner_coords, rotation=element['rotation'], center=(center_x, center_y))

    corner_coords = np.array(corner_coords, dtype=np.int32)

    return corner_coords


def recursive_search(gc, name, groups=None, _metadata=None, _id=None):
    """Recursive function for getting all items in a DSA collection, keeping the structure of the collection folders / items and including both
    folder / item metadata as well as HistomicsUI annotations.
    
    INPUTS
    ------
    gc : girder client, authenticated to get the collection data
    name : str
        name of the collection
    groups : list
        filter out annotation documents by only including those that contain a set of annotation groups
    _metadata & _id should not be passed, these parameters are used in the recursive calls only
    
    
    RETURN
    ------
    data : dict
        top level is a dictionary for info on the collection, the folders key contains the list of folders at the top level, each folder has folders and / or
        items within. An item might have annotations as well
    
    """    
    if _metadata is None:
        _metadata = {}
    
    data = {'name': name, 'folders': [], 'meta': _metadata}
    
    if _id is None:
        parent_type = 'collection'
        
        # find collection
        for col in gc.listCollection():
            if col['name'] == name:
                _id = col['_id']
                break
                
        if _id is None:
            return f'no collection \"{name}\", could not retrieve id'
    else:
        parent_type = 'folder'
        items = []
        
        for item in gc.listItem(_id):
            meta = item['meta'] if 'meta' in item else {}
            annotations = get_annotations_documents(gc, item['_id'], groups=groups)
            
            mag = ''
            if item['name'].endswith(('.ndpi', '.svs')):
                tile_metadata = get_tile_metadata(gc, item['_id'])
                
                if 'magnification' in tile_metadata:
                    mag = tile_metadata['magnification']
                    
            items.append({'name': item['name'], '_id': item['_id'], 'meta': meta, 'annotations': annotations, 'mag': mag})
            
        data['items'] = items
        
    data['_id'] = _id
            
    # get list of folders
    for fld in gc.listFolder(_id, parentFolderType=parent_type):
        # for each fld recursive search
        meta = fld['meta'] if 'meta' in fld else {}
        
        data['folders'].append(recursive_search(gc, fld['name'], groups=groups, _metadata=meta, _id=fld['_id']))
        
    return data


def get_annotations_documents(gc: GirderClient, item_id: str, clean: bool = None, docs: list = None, groups: list = None) -> list:
    """Get Histomics annotations for an image.
    
    Args:
        gc: Authenticated girder client.
        item_id: Item id.
        clean: Deprecated, used to set to clean up document. This is now always done.
        docs: Only include documents with given names.
        groups : Only include annotation documents that contain at least one annotation of these set of groups.
       
    Returns:
        List of annotation documents.
     
    """
    if clean is not None:
        print(Fore.YELLOW, Style.BRIGHT, 'The \"clean\" parameter is deprecated, documents are always cleaned.', 
              Style.RESET_ALL)
        
    ann_docs = []
    
    # Get information about annotation documents for item. 
    for ann_doc in gc.get(f'annotation?itemId={item_id}'):
        # Filter out bad documents and constrain which documents to get by inputs.
        if docs is not None:
            if ann_doc['annotation']['name'] not in docs:
                continue  # ignoring documents not in doc list
                
        # Filter out documents with no annotation groups
        if 'groups' not in ann_doc or not len(ann_doc['groups']):
            continue
            
        # If constrained to a set of annotation groups, ignore documents without any groups in the list
        if groups is not None:
            no_groups = True
            for grp in ann_doc['groups']:
                if grp in groups:
                    no_groups = False
                    break
            
            if no_groups:
                continue
                
        # Get the full document with elements.
        ann_doc = gc.get(f"annotation/{ann_doc['_id']}")
        
        # Remove any elements not in group or elements with no groups.
        filtered_els = []
        new_groups = set()
            
        for element in ann_doc['annotation']['elements']:
            if 'group' not in element:
                continue
            
            if groups is None or element['group'] in groups:
                filtered_els.append(element)
                new_groups.add(element['group'])
                    
        ann_doc['groups'] = list(new_groups)
        ann_doc['annotation']['elements'] = filtered_els
            
        # append this doc
        if len(filtered_els):
            ann_docs.append(ann_doc)
        
    return ann_docs


def get_tile_metadata(gc, itemid):
    """Get the tile source metadata for an item with a large image associated with it.
    Parameters
    ----------
    gc : girder_client.GirderClient
        an authenticated girder client session
    itemid : str
        WSI DSA item id
    Return
    ------
    metadata : dict
        the metadata for large image associated
    
    """
    metadata = gc.get(f'item/{itemid}/tiles')

    return metadata


def get_collection_id(gc, collection_name):
    """Get the collection ID by providing the collection name."""
    collection_id = None
    
    for col in gc.listCollection():
        if col['name'] == collection_name:
            return col['_id']
        
    print(f'collection {collection_name} does not exist')


def get_region(gc, item_id, left, top, height=None, width=None, right=None, bottom=None, mag=None, rgb=True, pad=None):
    """Get a region of a WSI image. Specify width and height or alternative give the right and bottom coordinate. Always
    give the left and top coordinate. Don't mix and match height / width with right / bottom.
    Parameters
    ----------
    gc : girder_client.GirderClient
        an authenticated girder client session
    item_id : str
        WSI DSA item id
    left : int
        left or minimum x coordinate of region in the wsi at native mag
    top : int
        top or minimum y coordinate of region in the wsi at native mag
    height : int (default: None)
        height of region to extract
    width : int (default: None)
        width of region to extract
    right : int (default: None)
        right or maximum x coordinate of region in the wsi at native mag
    bottom : int (default: None)
        bottom or maximum y coordinate of region in the wsi at native mag
    mag : float (default: None)
        magnification to pull region in
    rgb : bool (default: True)
        if True then the image will be returned as an RGB, removing any extra channels if present. Note that if less
        then three channels are present then the function will fail
    pad : int (default: None)
        if not None then image will be padded to match size given (helps pad regions overlapping edge of wsi). Pass an
        int from 0 (black) to 255 (white) with intermediate values being a hue of gray. Note that padding always removes
        any extra channels - so only works for RGB images.
    Return
    ------
    region_im : ndarray
        region image
    """
    # width and height but both be not None else right and bottom must be both not None otherwise exception
    if width is not None and height is not None:
        w, h = width, height

        if mag is None:
            get_url = f'{item_id}/tiles/region?left={left}&top={top}&regionWidth={width}&regionHeight={height}&units' \
                      f'=base_pixels&exact=false&encoding=PNG&jpegQuality=100&jpegSubsampling=0'
        else:
            get_url = f'{item_id}/tiles/region?left={left}&top={top}&regionWidth={width}&regionHeight={height}&units' \
                      f'=base_pixels&magnification={mag}&exact=false&encoding=PNG&jpegQuality=100&jpegSubsampling=0'
    elif right is not None and bottom is not None:
        w, h = right - left, top - bottom

        if mag is None:
            get_url = f'{item_id}/tiles/region?left={left}&top={top}&right={right}&bottom={bottom}&units=base_pixels' \
                      f'&exact=false&encoding=PNG&jpegQuality=100&jpegSubsampling=0'
        else:
            get_url = f'{item_id}/tiles/region?left={left}&top={top}&right={right}&bottom={bottom}&units=base_pixels' \
                      f'&magnification={mag}&exact=false&encoding=PNG&jpegQuality=100&jpegSubsampling=0'
    else:
        raise Exception('You must pass width / height or right / bottom parameters to get a region')

    resp_content = gc.get('item/' + get_url, jsonResp=False).content
    region_im = np.array(Image.open(BytesIO(resp_content)))

    if pad is not None:
        region_im = region_im[:, :, :3]
        x_pad = w - region_im.shape[1]
        y_pad = h - region_im.shape[0]
        if x_pad != 0 or y_pad != 0:
            region_im = np.pad(region_im, ((0, y_pad), (0, x_pad), (0, 0)), 'constant', constant_values=pad)
        return region_im

    if rgb:
        return region_im[:, : , :3]
    else:
        return region_im
    
    
def get_items(gc, parent_id):
    """Recursively gets items in a collection or folder parent location.
    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client instance
    parent_id : str
        the id of the collection / folder to get all items under
    Return
    ------
    items : list
        list of all items under the parent location
    """
    try:
        items = gc.get(f'resource/{parent_id}/items?type=collection&limit=0&sort=_id&sortdir=1')
    except HttpError:
        items = gc.get(f'resource/{parent_id}/items?type=folder&limit=0&sort=_id&sortdir=1')

    return items


def download_item_rois(gc, item_id, annotations, groups, save_dir, metas=None, url='https://computablebrain.emory.edu'):
    """Download all ROIs from a single DSA WSI and save them locally, return the metadata as a Dataframe.
    
    INPUTS
    ------
    gc : girder_client.GirderClient
        authenticated girder client to get images
    item_id : str
        item DSA _id
    annotations : dict
        contains the metadata for the item, including the name, meta dict, and annotations list
    groups : str or list
        annotation groups to save for the ROIs
    save_dir : str
        directory to save ROIs
    metas : list (default: None)
        list of metadata keys to include, if not found then an empty string is used
    url: str (default: 'https://computablebrain.emory.edu/')
        URL to the DSA containing the images - to include a URL to the ROI
        
    RETURN
    ------
    df : dataframe
        each row contains information about the ROIs saved
    
    """
    if metas is None:
        metas = []
    
    df = []
    
    makedirs(save_dir, exist_ok=True)
    
    meta = annotations['meta'] if 'meta' in annotations else {}  # metadata for the item
    ann_docs = annotations['annotations'] if 'annotations' in annotations else []
    name = annotations['name']
    mag = annotations['scan_mag'] if 'scan_mag' in annotations else ''

    # loop through the ROI annotations
    for ann_doc in ann_docs:
        # check if at least one of the elements in the document is in the group list
        if list(set(groups) & set(ann_doc['groups'])):
            # loop through the elements
            for element in ann_doc['annotation']['elements']:
                el_group = element['group']
                if el_group in groups:
                    corners = get_rectangle_element_coords(element)  # corners of the ROI - migth be rotated

                    # calculate the coordinates of the smallest bounding box of the ROI
                    x1, y1 = np.min(corners, axis=0)
                    x2, y2 = np.max(corners, axis=0)

                    # this is the width and height of the ROI, NOT the ROI bounding box image
                    w, h = float(element['width']), float(element['height'])

                    # URL link to the ROI in HitsomicsUI
                    url_to_roi = join(url, f"histomics#?image={item_id}&bounds={x1}%2C{y1}%2C{x2}%2C{y2}%2C0")

                    # create a unique filepath to save the ROI image
                    # clean up the name
                    clean_name = ' '.join(get_filename(name).split()).strip()
                    savepath = join(save_dir, f"{clean_name}_id-{item_id}_left-{x1}_top-{y1}_right-{x2}_bottom-{y2}.png")

                    # get & save the ROI image locally if it does not exist
                    if not isfile(savepath):
                        img = get_region(gc, item_id, x1, y1, right=x2, bottom=y2)
                        imwrite(savepath, img) 
                    
                    item_data = [name, item_id, mag, savepath, url_to_roi, x1, y1, x2, y2, w, h, xys_to_line(corners), el_group]
                    
                    # add any metas
                    for m in metas:
                        item_data.append(meta[m] if m in meta else '')
                    
                    df.append(item_data)  # add the item data
                            
    df = DataFrame(data=df, columns=['wsi_name', 'wsi_id', 'scan_mag', 'roi_im_path', 'url_to_roi', 'roi_im_left', 'roi_im_top',
                                    'roi_im_right', 'roi_im_bottom', 'roi_width', 'roi_height', 'roi_corners', 'roi_group'] + metas)
    return df


def download_rois(gc, id_list, annotations, groups, save_dir, metas=None, url='https://computablebrain.emory.edu', nproc=20):
    """Download ROIs from the DSA using parallel processing, and return a dataframe with the ROI data.
    
    INPUTS
    ------
    gc : girder_client.GirderClient
        authenticated girder client to get images
    id_list : list
        list of image ids in the DSA
    annotations : dict
        keys are image ids, values are DSA item data including the meta key
    groups : str or list
        annotation groups to save for the ROIs
    save_dir : str
        directory to save ROIs
    metas : list (default: None)
        list of metadata keys to include, if not found then an empty string is used
    url: str (default: 'https://computablebrain.emory.edu/')
        URL to the DSA containing the images - to include a URL to the ROI
    nproc : int (default: 10)
        number of processes to use when applying parallel processing on the images
        
    RETURN
    ------
    rois_df : dataframe
        each row contains information about the ROIs saved
        
    """
    if isinstance(groups, str):
        groups = [groups]
        
    rois_df = []  # add to this list
    
    # process each image / WSI in parallel
    pool = mp.Pool(nproc)
    jobs = [pool.apply_async(
        func=download_item_rois, args=(gc, item_id, annotations[item_id], groups, save_dir, metas, url,)) for item_id in id_list
    ]
    pool.close()
    
    for job in tqdm(jobs):
        rois_df.append(job.get())
        
    # return as a single dataframe
    return concat(rois_df, ignore_index=True)


def backup_annotations(gc: GirderClient, collection_name: str, save_fp: str = None,
                       exts: tuple = ('.svs', '.ndpi', '.czi'), docs: list = None) -> dict:
    """Create a backup file of the DSA annotations for image items in a collection. Includes item metadata and DSA 
    annotations. Additionally, it adds tile metadata for the large image.
    
    Args:
        gc: An authenticated girder client to the DSA client.
        collection_name: Function works for a single collection.
        save_fp: File path to save dictionary of annotations, it will save them as a json file.
        exts: Image extensions to include.
        docs: List of document names to include, if None then all will be included.
        
    Returns:
        DSA annotations in dictionary format, with the first level of keys being the folders names right under the 
        collection level. Inside these is a dictionary of items (for images only) with the key being the item id and 
        the values being dictionaries with keys: "name", "meta", "annotations", "cohort", "mag"
    
    """   
    # backup annotations
    annotations = {}

    # list all the cohort folders
    folders = list(gc.listFolder(get_collection_id(gc, collection_name), parentFolderType='collection'))
    for i, cohort_fld in enumerate(folders):
        print(f"{i+1} of {len(folders)} folders (\"{cohort_fld['name']}\"):")
        cohort = cohort_fld['name']
        annotations[cohort] = {}
        
        cohort_annotations = annotations[cohort]  # reference
        
        # list all the items in the cohort
        for item in tqdm(get_items(gc, cohort_fld['_id'])):
            # only include image items
            if item['name'].endswith(exts):
                # add name of image, metadata, cohort, scan magnification of image, and annotation documents
                cohort_annotations[item['_id']] = {
                    'name': item['name'], 
                    'meta': item['meta'], 
                    'cohort': cohort, 
                    'annotations': get_annotations_documents(gc, item['_id'], docs=docs), 
                    'scan_mag': get_tile_metadata(gc, item['_id'])['magnification']
                }
                
    if save_fp is not None:
        with open(save_fp, 'w') as fh:
            json.dump(annotations, fh)
    
    print(Fore.GREEN + Style.BRIGHT + '   Done!' + Style.RESET_ALL)
    return annotations


def convert_rgb_to_str(rgb):
    """Convert a tuple of rgb or rgba values into a HistomicsUI string"""
    if len(rgb) == 4:
        return f'rgba({int(rgb[0])},{int(rgb[1])},{int(rgb[2])},{float(rgb[3]):.2f})'
    else:
        return f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})'


def upload_annotation_doc(gc, wsi_id, df, doc_name='default', labels=None, colors=None, lw=4, overwrite_doc=False):
    """Upload a DSA annotation document from a dataframe of box objects data.
    
    INPUTS
    ------
    gc : girder_client.GirderClient
        aunthenticated girder client to push the annotations
    wsi_id : str
        DSA id of wsi
    df : dataframe
        each row contains label and coordinate of the box - label, x1, y1, x2, y2, conf
    doc_name : str (default: 'default')
        document name, will not overwrite!
    labels : list (default: None)
        label list to map int labels, if a label not found it will be ignored
    colors : list (default: None)
        list of RGB colors to use for each label. If None then it default to blue, red, and green for int labels 0, 1, 2 and black for all others.
    lw : int (default: 4)
        line width of the annotations
    overwrite_doc : bool (default: False)
        if True, then before pushing the document the previous documents will be ovewritten
    
    """
    # delete doc of same name
    if overwrite_doc:
        for ann in gc.get(f'/annotation/item/{wsi_id}'):
            if ann['annotation']['name'] == doc_name:
                gc.delete(f"/annotation/{ann['_id']}")
    
    if colors is None:
        colors = {0: 'rgb(0,0,255)', 1: 'rgb(255,0,0)', 2: 'rgb(0,255,0)'}
    else:
        colors = {i: convert_rgb_to_str(cl) for cl in enumerate(colors)}
        
    labels = {i: lb for i, lb in enumerate(labels)}
    
    doc = {'name': doc_name, 'description': '', 'elements': []}
    
    for _, r in df.iterrows():
        label = r.label
        
        if label in labels:
            color = colors[label] if label in colors else (0, 0, 0)
            
            label = labels[label]
            
            # calculate the center of the box and the width and height
            x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
            xc, yc = (x1 + x2) / 2, (y1 +  y2) / 2
            
            
            doc['elements'].append(
                {
                    'group': label,
                    'type': 'rectangle',
                    'lineColor': color,
                    'lineWidth': lw,
                    'rotation': 0,
                    'label': {'value': label},
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2, 0],
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            )
        
    # only push if there are annotations
    if len(doc['elements']):
        _ = gc.post(f'/annotation?itemId={wsi_id}', json=doc)

        
def point_to_box_annotations(
    gc: GirderClient, item_id: str, docs: Union[str, List[str]],
    groups: Union[str, List[str]] = None, no_watershed: bool = False, 
    default_box: int = 100, new_doc: str = 'boxes', notebook=False
):
    """Convert point annotations to boxes. 
    
    Args:
        gc: Girder client.
        item_id: DSA item ID.
        docs: Annotation documents to look at.
        groups: Only look at this annotation groups.
        no_watershed: If True then default box size will be used for all points.
        default_box: Default size of box to use when watershed fails or is not
            used.
        new_doc: A new doc is pushed that is a copy of the original doc but with
            this string added to it. Example: original docname: annotations, 
            new docname: annotations-boxes.
        notebook: Use tqdm for notebook.
        
    """
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        
    if isinstance(docs, str):
        docs = [docs]
    if groups is not None and isinstance(groups, str):
        groups = [groups]
        
    docs = get_annotations_documents(
        gc, item_id, docs=docs
    )
    
    # Image taken to do watershed is 3 times the size of the default box.
    ws_size = default_box * 3
    
    for i, doc in enumerate(docs):
        print(f"Annotation doc {i+1} of {len(docs)}")
        point_elements = []
        elements = []
    
        for element in doc['annotation']['elements']:
            del element['id']
            
            if (groups is None or element['group'] in groups) and \
                element['type'] == 'point':
                # Analyze
                point_elements.append(element)
            else:
                elements.append(element)
                        
        for element in tqdm(point_elements):
            xc, yc = np.array(element['center'][:2], dtype=int)
            
            # Convert to rectangle / box annotation. Default setting.
            el = {
                    'center': [int(xc), int(yc), 0],
                    'fillColor': element['fillColor'],
                    'group': element['group'],
                    'height': default_box,
                    'width': default_box,
                    'label': {'value': element['label']['value']},
                    'lineColor': element['lineColor'],
                    'lineWidth': element['lineWidth'],
                    'type': 'rectangle'
                }
            
            if not no_watershed:
                # Use watershed to estimate bounding box..
                left = xc - int(ws_size / 2)
                top =  yc - int(ws_size / 2)

                img = get_region(gc, item_id, left=left, top=top, 
                                 width=ws_size, height=ws_size)
                
                box = watershed_extraction(img, box=True)
                
                if len(box):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    el['width'] = int(w)
                    el['height'] = int(h)
                    el['center'] = [
                        int(x1 + x2) / 2 + left,
                        int(y1 + y2) / 2 + top,
                        0
                    ]
                    
            elements.append(el)
                
        _ = gc.post(
            f'/annotation?itemId={item_id}', 
            json={'name': f"{doc['annotation']['name']}-{new_doc}", 
                  'description': '', 'elements': elements}
        )
