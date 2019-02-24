""" This code is for parsing annotations from OTB/VOT videos

    Use parseVOTAnnotation/parseOTBAnnotation functions to parse the annotations.
    If you have similar style annotation files, you can potentially use 
        parseVOTStyleAnnotation/parseOTBStyleAnnotation functions.
"""

import numpy as np
import xml.etree.ElementTree
import json

def parseVOTStyleAnnotation(file_name):

    f = open(file_name, 'r')
    bounding_boxes = {}

    # Go through the file line by line and calculate center, height and width
    frame_num = 1
    for line in f.readlines():
        if line == "":
            continue

        # split by commas
        line = line.strip().split(',')
        values = map(float, line)
        if len(values) == 8:
            label_type = 'polygon'
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = values
        elif len(values) == 4:
            label_type = 'rectangle'
            left, top, width, height = values
            # print "Ran into Rectangle GT format for {0}".format(file_name)
        elif len(values) == 1: # this happens when VOT restarts. Just put whatever here
            label_type = 'rectangle'
            left, top, width, height = [0, 0, 1, 1]
        else:
            raise Exception("Not sure what ground truth label format is for video: {0}".format(file_name))

        # Get max x, min x, max y, min y
        if label_type == 'polygon':
            max_x = max([X1, X2, X3, X4])
            min_x = min([X1, X2, X3, X4])
            max_y = max([Y1, Y2, Y3, Y4])
            min_y = min([Y1, Y2, Y3, Y4])
        elif label_type == 'rectangle':
            min_x = left
            min_y = top
            max_x = left + width
            max_y = top + height

        # Calculate center
        bx = (max_x + min_x)/2.0
        by = (max_y + min_y)/2.0

        # Calculate height and width
        height = (max_y - min_y)
        width = (max_x - min_x)

        bounding_boxes[frame_num] = {'bx': bx, 'by': by, 'height': height, 'width': width}

        frame_num += 1

    return bounding_boxes

def parseOTBStyleAnnotation(file_name):

    gtFile = open(file_name, 'rb')
    gtLines = gtFile.readlines()
    gtRect = []
    for line in gtLines:
        if '\t' in line:
            gtRect.append(map(int,line.strip().split('\t')))
        elif ',' in line:
            gtRect.append(map(int,line.strip().split(',')))
        elif ' ' in line:
            gtRect.append(map(int,line.strip().split(' ')))
    gt = np.array(gtRect)

    # Calculate center
    bx = gt[:,0] + gt[:,2]/2.
    by = gt[:,1] + gt[:,3]/2.

    # Calculate height and width
    height = gt[:,3]
    width = gt[:,2]

    bounding_boxes = { i+1 : {'bx': bx[i], 'by': by[i], 'height': height[i], 'width': width[i]} for i in xrange(gt.shape[0]) }

    return bounding_boxes


def parseVOTAnnotation(video_path):

    file_name = video_path + 'groundtruth.txt'

    return parseVOTStyleAnnotation(file_name)

def parseOTBAnnotation(video_path):

    file_name = video_path + 'groundtruth_rect.txt'

    return parseOTBStyleAnnotation(file_name)


