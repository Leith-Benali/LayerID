#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 08:17:46 2022

@author: dirac
"""
import cv2 as cv
import os
import numpy as np
import sys
sys.path.append("..")

from utils.helper_functions import same_size

__all__ = ['stack_opened', 'residual_img', 'fground']

#ADD ASSERTIONS, EXCEPTIONS, AND DOCUMENTATION STRINGS

def stack_opened(list_npy):
    shape_images = list_npy[0].shape
    stacked_dims = (len(list_npy), shape_images[0], shape_images[1], shape_images[2])
    
    stacked_imgs = np.zeros(stacked_dims)
    for i, img in enumerate(list_npy):
        stacked_imgs[i] = img
    return stacked_imgs

def residual_img(stacked_imgs, indiv_loc, mode = 'abs', exp = 1, thrsh = 4):
    #Residual calculation using median of stacked images on single image
    modes = ['power', 'abs']
    assert mode in modes, "Mode is not valid."
    opened_img = cv.imread(indiv_loc)
    type_ = opened_img.dtype
    back = np.median(stacked_imgs, axis=0)
    
    opened_img = same_size(opened_img, back)
    
    if mode == 'power':
        incr = np.power(opened_img - back, exp)
        abs_ = np.abs(incr)
        resid = (abs_ / np.max(abs_)).astype(type_)
    if mode == 'abs':
        resid = (np.abs(opened_img - back)).astype(type_)
        
    type_ = opened_img.dtype
    return opened_img, back, resid, type_

def fground(resid, type_, stacked_imgs, var_mult = 1.5):
    #Elementary background subtraction with residual and standard deviation thresholding
    stdev = np.std(stacked_imgs, axis=0)
    thrsh = stdev*var_mult
    thrsh = (thrsh * (1/(np.max(thrsh) - np.min(thrsh))) * 255).astype(type_)
    
    foreground = (np.where(resid[:, :, :] > thrsh, 255, 0)).astype(type_)
    return foreground


            
    
