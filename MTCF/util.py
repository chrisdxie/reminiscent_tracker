""" Utilities for MTCF code.

    We use the "centered rectangle format", which represents a bounding box rectangle as:
        (center_x, center_y, width, height)
"""

import sys
import numpy as np
import os
import glob
import cv2
import json
import cyvlfeat

def hog(image, cell_size, variant='UoCTTI', n_orientations=9,
        directed_polar_field=False, undirected_polar_field=False,
        bilinear_interpolation=False, verbose=False):
    """ This code builds on top of cyvlfeat (https://github.com/menpo/cyvlfeat)
        NOTE: this was last tested on February 12, 2019. After then, it is possible 
              that cyvlfeat has correctly fixed it's code (there was a pull request
              to fix the code in 2016, but it seems it never went through...)

        However, the hog.py code provided has a mistake due to a confusion of
        row/col-major (Numpy/C uses row-major, MATLAB uses col-major)

        This code fixes the issue while calling the cyvlfeat.hog.hog function.

        @param image: a [H x W x 3] RGB array of np.float32
        @param cell_size: The size of the cells that the image should be decomposed into. Assumed
            to be square and this only a single integer is required.

        The rest of the parameter descriptions can be found in the cyvlfeat/hog/hog.py file.
    """

    # rearranging the data to the same way matlab stores images
    channels = [x.squeeze() for x in np.split(image, 3, axis=2)]
    image_shape = image.shape
    image = np.hstack([c.ravel() for c in channels]).reshape(image_shape)

    return cyvlfeat.hog.hog(image, cell_size, 
                            variant=variant, 
                            n_orientations=n_orientations,
                            directed_polar_field=directed_polar_field, 
                            undirected_polar_field=undirected_polar_field,
                            bilinear_interpolation=bilinear_interpolation, 
                            verbose=verbose)

def VOT_video_statistics(video_path):
    """ Given a VOT video directory, calculate some statistics of the video
    """
    first_frame = cv2.imread(video_path + '00000001.jpg')
    frame_height = first_frame.shape[0] 
    frame_width = first_frame.shape[1]
    num_frames_total = len(glob.glob1(video_path,"*.jpg"))

    return {'fh': frame_height, 'fw': frame_width, 'n_frames': num_frames_total}

def OTB_video_statistics(video_path):
    """ Given an OTB video directory, calculate some statistics of the video
    """
    config = json.load(open(video_path + 'cfg.json'))
    first_frame_filename = get_image_filename(int(config['startFrame']), 'otb')
    if 'Board' in video_path: # Hack for OTB video 
        first_frame_filename = '0' + first_frame_filename
    first_frame = cv2.imread(video_path + 'img/' + first_frame_filename)
    frame_height = first_frame.shape[0] 
    frame_width = first_frame.shape[1]
    num_frames_total = config["endFrame"] - config["startFrame"] + 1

    return {'fh': frame_height, 'fw': frame_width, 'n_frames': num_frames_total}

def stderr_print(string):
    """ Print to stderr. Useful for VOT software, where normal printing doesn't work
    """
    print(string, file=sys.stderr)

def resize_image(img, fx, fy, interpolation='zoom'):
    """ Resizes an image with fx, fy (x ratio, y ratio)

        Default interpolation uses cv2.INTER_LINEAR (good for zooming)
    """
    if interpolation == 'zoom':
        interp = cv2.INTER_LINEAR
    elif interpolation == 'shrink':
        interp = cv2.INTER_AREA
    else:
        raise Exception("Interpolation should be one of: ['zoom', 'shrink']")
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)

def load_image_with_resize(imagefile, fx=1, fy=1):
    """ Load image and perform some preprocessing
    """
    image = cv2.imread(imagefile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, fx, fy)
    return image

def extract_patch_from_slices(img, width_slice, height_slice):
    """ Performs MATLAB-like subwindow extraction from a 2D image (w/ multiple channels)

        Note: The following are equivalent
            Matlab: A([1,3,5],[1,3,5])       # returns a view of the data
            Numpy:  A[[0,2,4],:][:,[0,2,4]]  # returns a copy of the data
        Both return a [3 x 3] matrix

        @param width_slice: 1-D numpy array or list with elements in [0, ..., W-1]
        @param width_slice: 1-D numpy array or list with elements in [0, ..., H-1]
    """
    return img[height_slice, ...][:, width_slice, :]

def get_image_filename(file_num, dataset):
    """ A helper function to get filenames for OTB/VOT videos
    """
    if dataset == 'vot':
        return '%08d.jpg' % file_num
    elif dataset == 'otb':
        return '%04d.jpg' % file_num
    else:
        raise Exception('Dataset {0} unrecognized'.format(dataset))

def CLE(rect1, rect2):
    """ Calculates Center Location Error of two centered rectangles
    """
    c1 = np.array(rect1[:2])
    c2 = np.array(rect2[:2])
    return np.linalg.norm(c1 - c2)

def centered_rectangle_to_ltrb_rectangle(region):
    """ Transforms a centered rectangle format to 
        ltrb format: (left, top, right, bottom)
    """

    center_x, center_y, width, height = region
    left = center_x - width/2.
    top = center_y - height/2.
    right = center_x + width/2.
    bottom = center_y + height/2.
    return np.array([left, top, right, bottom])

def IoU(rect1, rect2):
    """ Calculates IoU of two rectangles in centered rectangle format.
    """
    # Convert to ltrb format for simpler processing
    rect1 = centered_rectangle_to_ltrb_rectangle(rect1)
    rect2 = centered_rectangle_to_ltrb_rectangle(rect2)

    intersection = max( min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]), 0 ) * \
                   max( min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]), 0 )

    # A1 + A2 - I
    union = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + \
            (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - \
            intersection 

    return float(intersection) / max(union, .00001)


def multidim_argmax(M):
    """ M is some multidimensional array. Return the index of the (first) argmax over all elements of M
    """
    return np.unravel_index(M.argmax(), M.shape)

def multidim_argmin(M):
    """ M is some multidimensional array. Return the index of the (first) argmin over all elements of M
    """
    return np.unravel_index(M.argmin(), M.shape)

def multidim_argmax_avg(M):
    """ M is some multidimensional array.

        Return the average of the indices of the maximum:

        E.g.

        M = | 1 2 3 4 |
            | 5 8 8 5 |
            | 5 8 8 5 |
            | 4 3 2 1 |

        multidim_argmax_avg(M) = (1.5, 1.5)
    """
    maximum = M.max()
    indices = np.argwhere(M == maximum)
    return np.mean(indices, axis=0)

def normalize(M):
    """ Take all values of M and normalize it to range [0, 1]
    """
    M = M.astype(np.float32)
    return (M.astype(np.float32) - M.min()) / (M.max() - M.min())

def build_matrix_of_indices(width, height):
    """ Builds a [height, width, 2] numpy array containing indices. 

        @return: 3d array b such that b[..., 0] is y-indices, b[..., 1] is x-indices
    """

    return np.indices((height, width)).transpose(1,2,0)

