# Working with object coordinates - such as boxes and contours
# functions:
# - get_contours
# - contour_to_line
# - xys_to_line
# - line_to_xys
# - pt_in_polygon
# - tile_im_with_boxes
# - convert_box_type
# - corners_to_polygon
import numpy as np
import cv2 as cv
from .utils import imwrite, save_to_txt

from os import makedirs
from os.path import join, isfile

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_contours(mask, enclosed_contours=False, min_points=3):
    """Get object contours from a binary mask using the OpenCV method.

    INPUTS
    ------
    mask : array-like
        binary mask to extract contours from - note that it must be in np.uint8 dtype, bool dtype won't work and this
        function will try to convert it to numpy uint8 form
    enclosed_contours : bool (default: False)
        if True then contours enclosed in other contours will be returned, otherwise they will be filtered
    min_points : int (default: 3)
        the minimum number of points a contour must have to be included

    RETURN
    ------
    contours : list
        list of object contours

    """
    # convert to uint8 dtype if needed
    if mask.dtype == 'bool':
        mask = mask.astype(np.bool)

    # extract contours - note that a default method is used for extracting contours
    contours, h = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # filter out contours that don't have enough points
    temp_contours = []
    for contour in contours:
        if contour.shape[0] >= min_points:
            temp_contours.append(contour)
    contours = temp_contours

    # filter out enclosed contours if needed
    if not enclosed_contours:
        temp_contours = []
        for i, contour in enumerate(contours):
            if h[0, i, 3] == -1:
                temp_contours.append(contour)
        contours = temp_contours

    return contours


def contour_to_line(contour):
    """convert a contour in opencv format to a string of x y points (i.e. x1 y1 x2 y2 x3 y3 ... xn yn)"""
    return ' '.join(contour.flatten().astype(str))


def xys_to_line(xys, shift=None):
    """Convert an array of xy coordiantes to a string of xy coordinates with spaces
    
    INPUTS
    ------
    xys : array-like
        shape (N, 2) with N being the number of points and each row having x, y coordinates
    shift : tuple / list (default: None)
        lenght of 2, corresponding to an x and y shift (subtracted) to the xys array
        
    RETURN
    ------
    : str
        the coordiantes flattened in a string as x1 y1 x2 y2 x3 y3 ... xN yN
        
    """
    if shift is not None:
        if len(shift) != 2:
            raise Exception('shift parameter must be of lenght 2')
            
        xys = (xys - shift).astype(int)
    
    return ' '.join([str(v) for v in xys.flatten()])


def line_to_xys(line, shift=None):
    """Convert a line of x y coordinates separated by spaces.
    
    INPUTS
    ------
    line : str
        x y coordinates for points
    shift : tuple / list (default: None)
        lenght of 2, shift the coordinates (subtracted) to the xys array
        
    RETURN
    ------
    xys : array-like
        shape (N, 2) with N being the number of points and each row having x, y coordinates
        
    """
    xys = np.reshape(np.array([int(c) for c in line.split(' ')]), (-1, 2))
    
    if shift is not None:
        if len(shift) != 2:
            raise Exception('')
            
        xys = (xys - shift).astype(int)
        
    return xys


def pt_in_polygon(pt_x, pt_y, vertices):
    """Provide the x, y coordinates of a point and the vertices in a polygon and return True if the point
    is in the polygon or False otherwise.
    
    INPUTS
    ------
    pt_x, pt_y : int or float
        x, y coordinates if the point
    vertices : list
        list of x, y tuples, the vertices of the polygon
        
    RETURN
    ------
    True or False
    
    """
    point = Point(pt_x, pt_y)
    polygon = Polygon(vertices)
    
    return polygon.contains(point)


def tile_im_with_boxes(im, boxes, save_dir, savename='', rotated_mask=None, box_thr=0.45, tile_size=1280, stride=960, 
                       pad_rgb=(114, 114, 114), rotated_thr=0.2, ignore_existing=False):
    """Tile an image containing box annotations. The image should be rectangular, but rotated images are supported. In the case of 
    rotated images a mask may be passed to identify the region of the image that is the rectangular region of interest. All other regions
    will be ignored.
    
    INPUTS
    ------
    im : array-like
        and RGB or gray-scale image that will be tiled, or broken into smaller regions
    boxes : list
        list of annotations boxes in image, each box has format [label, x1, y1, x2, y2] where point 1 is the top left corner of the box
        and point 2 is the bottom right corner of box. All coordinates are relative to (0, 0) being the top left corner of the image
    save_dir : str
        dir to create image and labels directories to save the images and label text file combos for the tiles
    savename : str (default: '')
        each tile image and label text file will include the coordinates of the tile, but will be prepended by the value of this 
        parameter, by default the prepend is an empty string
    rotated_mask : array-like (default: None)
        mask to specify region of interest inside image, for rotated images only (1: inside ROI, 0: outside of ROI)
    box_thr : float (default: 0.45)
        percentage of box (by area) that must be in tile to be included (from its original size)
    tile_size : int (default: 1280)
        size of tiles
    stride : int (default: 960)
        amount distance to travel between adjacent tiles, if it is less than tile_size then there will be overlap between tiles
    pad_rgb : tuple (default: (114, 114, 114))
        RGB to use when padding the image
    rotated_thr : float (default: 0.2)
        for rotated images, percentage of tile area that must be in rotated region to be included
    ignore_existing : bool (default: False)
        if True, then images will not save if a file of the name already exists
        
    RETURN
    ------
    save_paths : list
        list of image file paths that were saved, also includes x, y coordinate for that image
        
    """
    # create save locations
    im_dir = join(save_dir, 'images')
    label_dir = join(save_dir, 'labels')
    
    makedirs(im_dir, exist_ok=True)
    makedirs(label_dir, exist_ok=True)
    
    # create a mask for the boxes, with each box having a unique label
    height, width = im.shape[:2]
    boxes_mask = np.zeros((height, width), dtype=np.uint8)
    
    # for each box, track its area
    box_area_thrs = {}
    boxes_labels = {}
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[1:]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        box_area_thrs[i+1] = (x2 - x1) * (y2 - y1) * box_thr
        boxes_labels[i+1] = box[0]
        boxes_mask = cv.drawContours(boxes_mask, np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]), -1, i+1, cv.FILLED)
        
    # pad the edges of image
    im = cv.copyMakeBorder(im, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, value=pad_rgb)

    save_paths = []
    
    # calculate the area of a tile in rotated image to include
    area_thr = tile_size ** 2 * rotated_thr
    
    for x in range(0, width, stride):
        for y in range(0, height, stride):
            # ignore this tile if it is not sufficiently in rotated image
            if rotated_mask is not None and np.count_nonzero(rotated_mask[y:y+tile_size, x:x+tile_size]) < area_thr:
                continue
                
            filename = f'{savename}x{x}y{y}.'  # filename
            im_path = join(im_dir, filename + 'png')
                
            # grab tile
            tile_mask = boxes_mask[y:y+tile_size, x:x+tile_size].copy()
            tile_im = im[y:y+tile_size, x:x+tile_size, :].copy()

            # track lines to write to file for labels
            lines = ''

            # check each unique int label in tile
            for i in np.unique(tile_mask):
                if i > 0:
                    # ignore this object if not enough in tile
                    if np.count_nonzero(tile_mask == i) > box_area_thrs[i]:
                        # add the line for this object: class x_center y_center width height
                        # # normalize all values to 0 - 1 by dividing the tile size
                        yxs = np.where(tile_mask == i)

                        ymin, xmin = np.min(yxs, axis=1)
                        ymax, xmax = np.max(yxs, axis=1)

                        xcenter, ycenter = (xmax + xmin) / 2 / tile_size, (ymax + ymin) / 2 / tile_size
                        box_w, box_h = (xmax - xmin) / tile_size, (ymax - ymin) / tile_size

                        lines += f'{boxes_labels[i]} {xcenter} {ycenter} {box_w} {box_h}\n'

            # save image and label text file
            save_paths.append([im_path, x, y])  # append save name and coords to return
            
            # save the image only if it does not exist, or ignore_existing is False
            if not isfile(im_path) or not ignore_existing:
                imwrite(im_path, tile_im)
            
            # save its label text file if there are labels for tile
            if len(lines):
                save_to_txt(join(label_dir, filename + 'txt'), lines)
    
    return save_paths


def convert_box_type(box):
    """Convert a box type from YOLO format (x-center, y-center, box-width, box-height) to (x1, y1, x2, y2) where point 1 is the
    top left corner of box and point 2 is the bottom right corner
    
    INPUT
    -----
    box : array
        [N, 4], each row a point and the format being (x-center, y-center, box-width, box-height)
        
    RETURN
    ------
    new_box : array
        [N, 4] each row a point and the format x1, y1, x2, y2
        
    """
    # get half the box height and width
    half_bw = box[:, 2] / 2
    half_bh = box[:, 3] / 2
    
    new_box = np.zeros(box.shape, dtype=box.dtype)
    new_box[:, 0] = box[:, 0] - half_bw
    new_box[:, 1] = box[:, 1] - half_bh
    new_box[:, 2] = box[:, 0] + half_bw
    new_box[:, 3] = box[:, 1] + half_bh
    
    return new_box


def contours_to_points(contours):
    """Convert a list of opencv contours (i.e. contour shape is (num_points, 1, 2) with x, y order) to a list of x,y point in format ready
    to push as DSA annotations. This form is a list of lists with [x, y, z] format where the z is always 0
    
    INPUTS
    ------
    contours : list
        list of numpy arrays in opencv contour format
        
    """
    points = []
    
    for contour in contours:
        
        points.append([[float(pt[0][0]), float(pt[0][1]), 0] for pt in contour])
        
    return points
        
    
def corners_to_polygon(x1: int, y1: int, x2: int, y2: int) -> Polygon:
    """Return a Polygon from shapely with the box coordinates given the top left and bottom right corners of a 
    rectangle (can be rotated).
    
    Args:
        x1, y1, x2, y2: Coordinates of the top left corner (point 1) and the bottom right corner (point 2) of a box.
        
    Returns:
        Shapely polygon object of the box.
        
    """
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
