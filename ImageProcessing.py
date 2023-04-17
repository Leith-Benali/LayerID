from __future__ import print_function
import os
import cv2 as cv
import numpy as np
import argparse
import random as rng
from layerID_test_new import testing

rng.seed(12345)

def thresh_callback(img, smallest_flake, flake_name, masking, master_cat_file, cluster_count):
    parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
    parser.add_argument('--input', help='Path to input image.', default=img)
    args = parser.parse_args()
    src = cv.imread(cv.samples.findFile(args.input))

    #downscale image
    downscale_factor = 4
    scale_percent = 100/downscale_factor  # percent of original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    src = cv.resize(src, dim, interpolation=cv.INTER_AREA)

    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))
    source_window = 'Source'
    cv.namedWindow(source_window)
    # cv.imshow(source_window, src)


    threshold = 5
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # loop through the contours to get rid of ones that are to small
    result = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(contours[i])
        if w*h > smallest_flake:
            result.append(cnt)
    contours = result

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    # cv.imshow('Contours', drawing)
    cv.imwrite(f"Flakes/boxed_flakes_{flake_name}.jpg", drawing)

    for i, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(contours[i])
        args = {'img_file': img,
                'flake_name': flake_name + str(i),
                'crop': [downscale_factor * y, downscale_factor * (y + h), downscale_factor * x, downscale_factor * (x + w)],
                'masking': masking,
                'master_cat_file': master_cat_file,
                'cluster_count': cluster_count}

        testing(**args)

"""
    Parameters
    ----------
    img_directory : str
        Location of sample image file. (i.e., "...\\RSGR001\\All\\3A1.jpg")
    flake_name : str
        Name of sample image. (i.e., "RSGR001 3A1")
    smallest_flake : int
        Size of the smallest flakes to be detected.
    masking : list of list of ints of form [[miny1,maxy1,minx1,maxx1], ...]
        Regions of image to fit background. Indices relative to cropped
        image. (i.e., [[200,-1, 0,175], [0,200, 500,-1], [700,-1, 500,-1]])
    master_cat_file : str
        Location of master catalog npz file for the same material/substrate as
        sample. (i.e., "...\\Graphene_on_SiO2_master_catalog.npz")
    cluster_count : int
        Number of layers to fit up to. Determine based on how many layers
        "master_cat_file" was trained to.
    """

# Enter the parameters required by the processImage function
img_directory = "Test_Images"
for filename in os.listdir(img_directory):
    img = os.path.join(img_directory, filename)
    thresh_callback(img,
                    smallest_flake=100,
                    flake_name=f"Khang{filename}",
                    masking=[[0, 1, 0, 1]],
                    master_cat_file=".\\Monolayer Search\\Graphene_on_SiO2_master_catalog.npz",
                    cluster_count=5,)

