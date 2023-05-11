#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:18:12 2022

@author: dirac
"""
import cv2 as cv
#REPLACE OPENCV!
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv

__all__ = ['getTransmissionAngle', 'partialReflection_p', 'partialReflection_s', 'graphene_thickness']

def getTransmissionAngle(t, n0, n1):
   return np.arcsin((n0 / n1) * np.sin(t))

def partialReflection_p(t0, t1, n0, n1):
    numerator   = n1 * np.cos(t0) - n0 * np.cos(t1)
    denominator = n1 * np.cos(t0) + n0 * np.cos(t1)
    return numerator / denominator

def partialReflection_s(t0, t1, n0, n1):
    numerator   = n0 * np.cos(t0) - n1 * np.cos(t1)
    denominator = n0 * np.cos(t0) + n1 * np.cos(t1)
    return numerator / denominator

def graphene_thickness(n):
    #Returns the thickness of graphene as a function of layer number n
    d = 0.475e-9*n - 0.14e-9
    d*=2
    return d


