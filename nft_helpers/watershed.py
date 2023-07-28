# Watershed
# functions:
# - watershed_extraction
# - apply_watershed
import numpy as np
from scipy import ndimage as ndi
import cv2 as cv
from .box_and_contours import get_contours, contour_to_line

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    area_closing, binary_erosion, binary_dilation, area_opening, square, diamond, disk, star
)
from skimage.segmentation import watershed as skwatershed
from skimage.feature import peak_local_max


def _binary_operations(im, directions):
    """Run multiple binary operations on a single image.

    INPUTS
    ------
    im : array-like
        grayscale image to run binary operations on
    directions : str
        string separated by -, o: opening, c: closing, d: dilation, e: erosion, s: square kernel, d: diamon kernel,
        k: disk kernel, t: star kernel

    RETURN
    ------
    im : array-like
        image after binary operations

    """
    for order in directions.split('-'):
        # check type
        if order[0] in ('d', 'e'):
            # get kernel from second letter
            if order[1] == 's':
                kernel = square(int(order[2]))
            elif order[1] == 'd':
                kernel = diamond(int(order[2]))
            elif order[1] == 'k':
                kernel = disk(int(order[2]))
            elif order[1] == 't':
                kernel = star(int(order[2]))
            else:
                raise Exception('Directions not correct, kernel not a valid option.')

            if order[0] == 'd':
                im = binary_dilation(im, footprint=kernel)
            else :
                im = binary_erosion(im, footprint=kernel)
        elif order[0] in ('o', 'c'):
            # get area size
            area_threshold = int(order[1:])

            if order[0] == 'o':
                im = area_opening(im, area_threshold=area_threshold)
            else:
                im = area_closing(im, area_threshold=area_threshold)

    return im


def watershed(im, binary_instructions='es3-ds3-c500', blur=3, sigma=3, edge_blur=True, threshold=None, objects=1):
    """Detect central object in an image using a customized watershed approach.

    INPUTS
    ------
    im : array-like
        rgb image containing object to detect
    binary_instructions : str (default: "es3-ds3-c500")
        string of binary operation to apply, if None then none are applied. Look at binary_operations for detailed
        instructions of this parameter
    blur : int (default: 3)
        size of gaussian blur to use, 0 means none is used
    sigma : int (default: 5)
        size of gaussian kernel to use for blurring - affects detection of peaks
    edge_blur : bool (default: True)
        if True apply a small gaussian blur on final binary mask to clean up the sharp edges
    threshold : float (default: None)
        if None then Otsu's thresholding is used - otherwise specify a float from 0 to 1 for thresholding
    objects : int (default: 1)
        maximum number of ojbects to detect, they n top objects are chosen based on their eucledian distance to center

    RETURNS
    -------
    : array-like
        binary mask of detected object
    int_images : list
        list of intermediate images to return, useful in understanding results - (grayscale, Otsu's binary, binary
        operations, gaussian blurs,

    """
    int_images = []

    # central point of image
    center_x, center_y = im.shape[1] // 2, im.shape[0] // 2

    # grayscale rgb image
    gray_im = rgb2gray(im)

    int_images.append(gray_im)
    
    if threshold is None:
        # binarize image using Otsu's thresholding
        binary_im = gray_im < threshold_otsu(gray_im)
    else:
        binary_im = gray_im < threshold
    int_images.append(binary_im)
    
    # clean up mask using binary operations
    if binary_instructions is not None:
        binary_im = _binary_operations(binary_im, binary_instructions)
    int_images.append(binary_im)
    
    blur_im = 1 - gaussian(gray_im, sigma=(sigma, sigma))
    
    peaks = peak_local_max(blur_im, exclude_border=False)  # peaks for watershed
    
    # peak mask is same shape as image, pixels where watershed peaks were located are marked as True
    peak_mask = np.zeros(blur_im.shape, dtype=bool)
    peak_mask[tuple(peaks.T)] = True  # note that a peak is usually a single dot

    # convert the bool mask of watershed peaks to a unique int label mask, True pixels not adjacent to each other
    # are given unique int labels
    peak_mask = ndi.label(peak_mask)[0]

    # calcualte the euclidean distance map using the binary mask
    distance_map = ndi.distance_transform_edt(binary_im)

    # apply watershed - subject to binary mask, distance heatmap, and peak mask
    # returns a watershed mask with ALL blobs found
    watershed_mask = skwatershed(-distance_map, peak_mask, mask=binary_im, watershed_line=False)

    int_images.append(watershed_mask)

    # check if there is a blob in center of image
    blob_label = watershed_mask[center_y, center_x]

    if objects and blob_label:
        # filter the watershed mask to only contain the center blob
        blob_ys, blob_xs = np.where(watershed_mask == blob_label)
    else:
        # if no center blob found then find blob closest to center
        # alternative take the markers for the top n objects near center
        # min_dist = 1e20  # very large distance
        distances = []
        labels = []
        for m in np.unique(watershed_mask):
            if m != 0:
                if m == blob_label:
                    dist = 0
                else:
                    m_ys, m_xs = np.where(watershed_mask == m)
                    m_ymin, m_ymax = np.min(m_ys), np.max(m_ys)
                    m_xmin, m_xmax = np.min(m_xs), np.max(m_xs)
                    center = (m_xmax + m_xmin) // 2, (m_ymax + m_ymin) // 2
                    dist = np.linalg.norm(np.array([center_y, center_x]) - np.array(center))

                labels.append(m)
                distances.append(dist)
                # if dist < min_dist:
                #     min_dist = dist
                #     blob_label = m
        if len(distances):
            distances, labels = (list(t) for t in zip(*sorted(zip(distances, labels))))

            blob_ys, blob_xs = np.where(np.isin(watershed_mask, labels[:objects]))
        else:
            blob_ys, blob_xs = [], []

    # create a binary mask of the object and blur slightly to make edges smoother
    mask = np.zeros(gray_im.shape, np.uint8)
    mask[blob_ys, blob_xs] = 255

    if edge_blur:
        mask = (gaussian(mask, 1) > 0).astype(np.uint8) * 255

    return mask, int_images


def watershed_extraction(im, box=True, as_str=False):
    """Apply a customized watershed method to an image to extract an object at its center. Returns the contours of the object as well as 
    the smallest bounding box.
    
    INPUTS
    ------
    im : array-like
        rgb image
    box : bool
        if True then contours are returned as top left and bottom right coordinates
    as_str : bool
        if True then it is returned as a string
        
    """
    h, w = im.shape[:2]
    xc, yc = int(w / 2), int(h / 2)
    
    ws_mask = watershed(im)[0]

    # erode mask 3 times
    for i in range(3):
        ws_mask = binary_erosion(ws_mask, footprint=square(3)).astype(np.uint8) * 255

    # extract contours
    contours = get_contours(ws_mask)

    # take the contour that includes the center point only
    for contour in contours:
        mask = np.zeros([h, w], dtype=np.uint8)
        mask = cv.drawContours(mask, [contour], -1, 255, cv.FILLED)

        # if contour contains center then use this contour
        if mask[yc, xc]:
            if box:
                # convert to just the top left and bottom right corner of bounding box
                x1, y1 = np.min(contour, axis=0)[0]
                x2, y2 = np.max(contour, axis=0)[0]
                
                if as_str:
                    return f'{x1} {y1} {x2} {y2}'
                else:
                    return x1, y1, x2, y2
            else:
                # convert contour to string of x y coordinate and save to text file if there was a contour containing the center
                if as_str:
                    return contour_to_line(contour)
                else:
                    return contour
        
    return ''
