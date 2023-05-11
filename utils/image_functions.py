#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:39:14 2022

@author: dirac
"""
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['return_opened', 'stdize_img_size', 'histogram_per_ch']

FILE_EXT = ('.jpg', '.tiff', '.png', '.pdf')

def return_opened(tiff_file_dir):
    opened_ = []
    list_files = os.listdir(tiff_file_dir)
    for fl in list_files:
        if fl.lower().endswith(FILE_EXT):
            opened = cv.imread(tiff_file_dir + fl)
            opened_.append(opened)
            if opened is None:
                raise BaseException("The .tiff file could not be opened.")
    return opened_

def stdize_img_size(files_dir, end_size = (300, 300), end_files_dir = None):
    if end_files_dir == None:
        print("WARNING: saving resized files in same directory, overwriting originals.")
    for i in os.listdir(files_dir):
        if i.lower().endswith(FILE_EXT):
            opened = cv.imread(files_dir + i)
            opened = cv.resize(opened, end_size)
            if end_files_dir == None:
                cv.imwrite(files_dir + i, opened)
            else:
                cv.imwrite(end_files_dir + i, opened)
                
def histogram_per_ch(opened_img):
    hist, bin_edges = np.histogram(opened_img, bins=256, range=(0, 256))
    hist1, bin_edges1 = np.histogram(opened_img[:,:,0], bins=256, range=(0, 256))
    hist2, bin_edges2 = np.histogram(opened_img[:,:,1], bins=256, range=(0, 256))
    hist3, bin_edges3 = np.histogram(opened_img[:,:,2], bins=256, range=(0, 256))
    
    plt.title("Blue Channel")
    plt.plot(bin_edges1[0:-1], hist1)
    plt.show()
    
    plt.title("Green Channel")
    plt.plot(bin_edges2[0:-1], hist2)
    plt.show()
    
    plt.title("Red Channel")
    plt.plot(bin_edges3[0:-1], hist3)
    plt.show()
    
    plt.title("ALL CHANNELS")
    plt.plot(bin_edges[0:-1], hist)
    plt.show()
    
            