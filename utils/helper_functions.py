#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:28:00 2022

@author: dirac
"""
import cv2 as cv
#REPLACE OPENCV!
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv

__all__ = ['resize_show', 'return_stats', 'same_size', 'gaussian', 'planck', 'getCSVFloats']

def getCSVFloats(path):
    with open(path, 'r') as file:
        r = csv.reader(file, delimiter = ",")
        data = list(r)
    rows = [[float(c) for c in r] for r in data[1:]]
    return rows

def gaussian(x, s, x0):
    A =  1.0 / (s*np.sqrt(2*np.pi))
    return A * np.exp(-np.square(x-x0) / (2*np.square(s)))

def planck(wv, T):
    h = 6.62607015e-34
    c = 299792458
    k = 1.380649e-23
    res = (2*h*(c**2)) / (wv**5)
    res_ = res*(1. / (np.exp((h*c) / (wv*k*T)) - 1.))
    return res_

def resize_show(opened_img, size = (250,250)):
    img_temp = cv.resize(opened_img, size)
    img_temp = cv.cvtColor(np.float32(img_temp), cv.COLOR_BGR2RGB)
    if np.max(img_temp) > 1.0:    
        plt.imshow(img_temp/255.0)
        plt.show()
    else:
        plt.imshow(img_temp)
        plt.show()
    return img_temp

def return_stats(arr, ax = None):
    if ax == None:
        print("WARNING: NO AXIS SPECIFIED")
        med, mean = np.median(arr), np.mean(arr)
        stdev = np.std(arr)
        max_, min_ = np.max(arr), np.min(arr)
        shape_s, shape_md, shape_m = stdev.shape, med.shape, mean.shape
        
        print("SHAPE OF STDEV, MEDIAN, MEAN ARE {}, {}, {}".format(shape_s, shape_md, shape_m))
        print("MEDIAN OF ARR IS {}, MEAN IS {}".format(med, mean))
        print("MAX OF ARR IS {}, MIN IS {}".format(max_, min_))
    else:  
        print("WARNING: AXIS SPECIFIED")
        med, mean = np.median(arr, axis=ax), np.mean(arr, axis=ax)
        stdev = np.std(arr, axis=ax)
        max_, min_ = np.max(arr, axis=ax), np.min(arr, axis=ax)
        shape_s, shape_md, shape_m = stdev.shape, med.shape, mean.shape
        
        print("SHAPE OF STDEV, MEDIAN, MEAN ARE {}, {}, {}".format(shape_s, shape_md, shape_m))
        print("MEDIAN OF ARR IS {}, MEAN IS {}".format(med, mean))
        print("MAX OF ARR IS {}, MIN IS {}".format(max_, min_))
    

def same_size(opened_img, back):
    if opened_img.shape != back.shape:
        return cv.resize(opened_img, (back.shape[1], back.shape[0]))
    else:
        return opened_img