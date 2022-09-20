# Author:      Adam Robinson
# Description: This program scans a region of the microscope stage in two steps. In the first step,
#              it traverses the specified scan area at a low zoom level and find interesting regions
#              using a simple mechanism that looks for pixels with optical contrast values in a 
#              certain range. During the second step, each of these regions of interest is imaged
#              at a higher zoom. These images are placed in an output directory. The user has the
#              option to have these images processed as they are taken. The processing code will
#              attempt to categorize them by their thickness.

import sys
sys.path.append("hardware_code")

import matplotlib.pyplot as plt
import numpy             as np

import time
import code
import argparse
import os
import cv2
import json
import threading
import math
import sqlite3            as sql
import matplotlib.patches as patches
from typing import List, Tuple

from hardware_code.MicroscopeControl import MicroscopeController

#from MicroscopeControl  import MicroscopeController
from scipy.optimize     import curve_fit
from matplotlib.patches import Rectangle
from Progress           import ProgressBar
from multiprocessing    import Pool
from ProcessScan        import MultiProcessImageProcessor
from datetime           import datetime
from skimage            import data, restoration, util
from shutil             import copyfile

from mpl_toolkits.mplot3d import Axes3D

def preprocess(args_specification):
	parser = argparse.ArgumentParser(description=args_specification['description'])

	types = {'str': str, 'int': int, 'float': float}

	for argument in args_specification['arguments']:
		spec = argument['spec']
		if 'type' in spec:
			spec['type'] = types[spec['type']]
		parser.add_argument(
			*argument['names'], 
			**spec
		)

	args = parser.parse_args()

	return args

def calibrateFocus(microscope, args, gfit=False, debug=False):
	focus_test_points = np.array([
		args.focus_points[i:i + 2] for i in range(0, len(args.focus_points), 2)
	])
	focal_points = []
	for x, y in focus_test_points:
		microscope.stage.moveTo(x, y)
		microscope.autoFocus(
			args.focus_range,
			args.autofocus_parameters[0],
			args.autofocus_parameters[1],
			args.autofocus_parameters[2],
			debug=debug,
			gfit=gfit
		)
		focal_points.append(microscope.focus.getFocus())

	focal_points = np.array(focal_points)

	def fit(X, mx, my, b):
		x, y = X
		return mx*x + my*y + b

	res, cov = curve_fit(fit, focus_test_points.T, focal_points)

	def interp(X):
		x, y = X
		return res[0]*x + res[1]*y + res[2]

	rmse = np.sqrt(np.square(interp(focus_test_points.T) - focal_points).mean())
	print("The focal points have been fit to a function of the form z = ax + by + c.")
	print("RMSE of fit: %f"%rmse)

	if rmse > args.max_focus_error:
		print("Fit too poor to continue.")
		print("Please ensure that the sample is flat and level.")
		exit()

	return interp, res[0], res[1], res[2]

def getFocusInterpolation(mx, my, b):
	def interp(X):
		x, y = X
		return mx*x + my*y + b
	return interp

def getRegionsOfInterest(img, bg, args, x0, y0):
	fine_zoom,   fine_w,   fine_h,   fine_exposure   = args.fine_zoom
	coarse_zoom, coarse_w, coarse_h, coarse_exposure = args.coarse_zoom

	if args.preview:
		pv = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
		cv2.imshow('Scan Preview', pv)
		cv2.waitKey(1)

	# Originally, this code tried to guess a uniform background by finding the most common pixel
	# value. The current version of this code does not subtract a background before processing the
	# image. Instead, it uses the nonuniform background (determined at an earlier step), in the
	# optical contrast calculation.

	img        = img.astype(np.float32)
	contrast   = (bg - img) / bg

	mm_per_pixel_x = coarse_w / contrast.shape[1]
	mm_per_pixel_y = coarse_h / contrast.shape[0]
	mm_per_pixel   = (mm_per_pixel_x + mm_per_pixel_y) / 2

	# Now we iterate over the above mentioned regions of this image and calculate the percentage
	# of each image that has contrast greater than zero.
	x, y    = 0, 0
	regions = []
	pixel_coordinates = [] # DEBUG
	while x < coarse_w - (fine_w / 2):
		while y < coarse_h - (fine_h / 2):
			# We need to convert coordinates into indices into the image array so that we can select
			# the correct subset of pixels for the calculation.
			y_low  = int(round(y  / mm_per_pixel))
			y_high = int(round((y + fine_h) / mm_per_pixel)) + 1

			x_low  = int(round(x  / mm_per_pixel))
			x_high = int(round((x + fine_w) / mm_per_pixel)) + 1

			subimage   = contrast[y_low:y_high, x_low:x_high]
			relevant   = (subimage >= args.contrast_range[0]) & (subimage <= args.contrast_range[1])
			r          = subimage.copy()

			# The halo effect around flakes causes problems when attempting to identify flake 
			# thickness. The halo around a flake tends to smoothly transition between the optical
			# contrast of the flake itself and the substrate. This results in 1-5 pixels around the
			# flake that appear to have lower optical contrast but aren't actually real. The erode
			# and dilate process below tends to eliminate the halo pixels that contain incorrect 
			# contrast values.
			r[relevant]          = 1.0
			r[relevant == False] = 0.0
			ksize   = 4
			kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
			eroded  = cv2.erode(r, kernel)
			dilated = cv2.dilate(eroded, kernel)
			ratio   = dilated.sum() / (subimage.shape[0] * subimage.shape[1])


			if ratio > args.threshold_ratio:
				pixel_coordinates.append([x_low, x_high, y_low, y_high]) # DEBUG
				x_shift = (coarse_w - fine_w) / 2
				y_shift = (coarse_h - fine_h) / 2
				regions.append([x + x0 - x_shift, y + y0 - y_shift])
				# code.interact(local=locals())

			y += fine_h
		y = 0
		x += fine_w

	if args.preview:
		timg = contrast.copy()
		rect_color = contrast.max() * 2
		for rect in pixel_coordinates:
			timg = cv2.rectangle(
				timg, 
				(rect[0], rect[2]), 
				(rect[1], rect[3]), 
				(rect_color, 0, 0), 
				3
			)
		_min = timg.min()
		_max = timg.max()
		timg  = (((timg - _min) / (_max - _min)) * 255).astype(np.uint8)
		pv = cv2.resize(timg, (0, 0), fx=0.3, fy=0.3)
		cv2.imshow('Scan Preview', pv)
		cv2.waitKey(1)

	return regions

def parabolicSubtract(img, n_fit=128):
	def fit(X, mmx, mmy, mx, my, b):
		y, x = X
		return mmx*x**2 + mmy*y**2 + mx*x + my*y + b

	# Select a random subset of the image pixels to perform the fit on. This will be too slow 
	# otherwise.
	fit_x = np.random.randint(0, img.shape[1], (n_fit, ))
	fit_y = np.random.randint(0, img.shape[0], (n_fit, ))
	fit_points = np.stack([fit_y, fit_x], axis=1)

	Z = img[fit_y, fit_x]

	res, cov = curve_fit(fit, fit_points.T, Z)

	def interp(y, x):
		return res[0]*x**2 + res[1]*y**2 + res[2]*x + res[3]*y + res[4]

	Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
	background = interp(Y, X)
	subtracted = img - background.T

	_min = subtracted.min()
	_max = subtracted.max()
	subtracted = (((subtracted - _min) / (_max - _min)) * 255).astype(np.uint8)

	code.interact(local=locals())

	return subtracted

# Used for debugging purposes. Creates a 3d surface plot of an image.
def plotImage(img, ds=20):
	img = cv2.resize(img, (0, 0), fx=(1 / ds), fy=(1 / ds))
	xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

	fig = plt.figure()
	ax  = fig.gca(projection='3d')
	ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
	plt.show()

def calculateBackgroundGreyscale(args, focus_interp, microscope):
	x_min, x_max, y_min, y_max = args.bounds
	# After initial testing I realized that this system produces a background that cannot be 
	# approximated as a simple function when the zoom is not very close to being maxed out. As a 
	# result, it's necessary to determine this background before taking serious images. In order
	# to do this, we take many images at random positions, perform a rolling ball background 
	# subtraction and then take an average of them to get the background. The rolling ball 
	# algorithm tends to produce some irregularities around flakes, so we won't take a regular 
	# average. Instead we will take the standard deviation of each pixel across images and throw
	# out values that are far outside of the mean.
	n_background_images = args.n_background_images
	bg_img_progress = ProgressBar("Background Images", 18, n_background_images, 1, ea=10)
	positions           = np.array([
		np.random.uniform(x_min, x_max, n_background_images), 
		np.random.uniform(y_min, y_max, n_background_images)
	])
	images = []
	for i, (x, y) in enumerate(positions.T):
		microscope.stage.moveTo(x, y)
		microscope.focus.setFocus(
			focus_interp((x, y)),
			corrected=False
		)
		images.append(avgimg(args.coarse_averages, args.coarse_downscale))
		bg_img_progress.update(i + 1)

	imgs  = np.stack(images, axis=2)
	means = imgs.mean(axis=2)
	stds  = imgs.std(axis=2)

	weights = []
	for img in images:
		mask         = np.abs(img - means) < 1.2 * stds
		weight       = np.ones(mask.shape) * 0.001
		weight[mask] = 1.0
		weights.append(weight) 

	weights    = np.stack(weights, axis=2)
	background = np.average(imgs, axis=2, weights=weights)

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.imshow(images[0])
	# ax2.imshow(background)
	# ax3.imshow(images[0] - background)
	# plt.show()
	# plotImage(background)
	# code.interact(local=locals())
	return background

def calculateBackgroundColored(args, focus_interp, microscope):
	x_min, x_max, y_min, y_max = args.bounds
	# After initial testing I realized that this system produces a background that cannot be 
	# approximated as a simple function when the zoom is not very close to being maxed out. As a 
	# result, it's necessary to determine this background before taking serious images. In order
	# to do this, we take many images at random positions, perform a rolling ball background 
	# subtraction and then take an average of them to get the background. The rolling ball 
	# algorithm tends to produce some irregularities around flakes, so we won't take a regular 
	# average. Instead we will take the standard deviation of each pixel across images and throw
	# out values that are far outside of the mean.
	n_background_images = args.n_background_images
	bg_img_progress = ProgressBar("Background Images", 18, n_background_images, 1, ea=10)
	positions           = np.array([
		np.random.uniform(x_min, x_max, n_background_images), 
		np.random.uniform(y_min, y_max, n_background_images)
	])
	images = []
	for i, (x, y) in enumerate(positions.T):
		microscope.stage.moveTo(x, y)
		microscope.focus.setFocus(
			focus_interp((x, y)),
			corrected=False
		)
		images.append(avgimg(args.fine_averages))
		bg_img_progress.update(i + 1)

	imgs_r_arr = [i[:, :, 2] for i in images]
	imgs_g_arr = [i[:, :, 1] for i in images]
	imgs_b_arr = [i[:, :, 0] for i in images]
	imgs_r = np.stack(imgs_r_arr, axis=2)
	imgs_g = np.stack(imgs_g_arr, axis=2)
	imgs_b = np.stack(imgs_b_arr, axis=2)
	means_r = imgs_r.mean(axis=2)
	means_g = imgs_g.mean(axis=2)
	means_b = imgs_b.mean(axis=2)
	stds_r  = imgs_r.std(axis=2)
	stds_g  = imgs_g.std(axis=2)
	stds_b  = imgs_b.std(axis=2)

	weights_r = []
	weights_g = []
	weights_b = []
	for img_r, img_g, img_b in zip(imgs_r_arr, imgs_g_arr, imgs_b_arr):
		mask_r       = np.abs(img_r - means_r) < 1.2 * stds_r
		mask_g       = np.abs(img_g - means_g) < 1.2 * stds_g
		mask_b       = np.abs(img_b - means_b) < 1.2 * stds_b

		weight_r         = np.ones(mask_r.shape) * 0.001
		weight_r[mask_r] = 1.0
		weights_r.append(weight_r) 

		weight_g         = np.ones(mask_g.shape) * 0.001
		weight_g[mask_g] = 1.0
		weights_g.append(weight_g) 

		weight_b         = np.ones(mask_b.shape) * 0.001
		weight_b[mask_b] = 1.0
		weights_b.append(weight_b) 

	weights_r    = np.stack(weights_r, axis=2)
	background_r = np.average(imgs_r, axis=2, weights=weights_r)

	weights_g    = np.stack(weights_g, axis=2)
	background_g = np.average(imgs_g, axis=2, weights=weights_g)

	weights_b    = np.stack(weights_b, axis=2)
	background_b = np.average(imgs_b, axis=2, weights=weights_b)

	background = np.stack([background_b, background_g, background_r], axis=2)

	#code.interact(local=locals())

	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	# ax1.imshow(images[0])
	# ax2.imshow(background)
	# ax3.imshow(images[0] - background)
	# plt.show()
	# code.interact(local=locals())
	return background

# This function will process the args_file (if specified) and ensure that all required arguments
# have been specified.
def validateArgs(args):
	# First we'll take arguments specified at the command line and write them over arguments
	# supplied in the file.

	if args.args_file is not None:
		with open(args.args_file, 'r') as file:
			args_file = json.loads(file.read())
		arg_dict = args.__dict__

		for k, v in arg_dict.items():
			if not k.startswith("_"):
				if k in args_file:
					if v is not None:
						args_file[k] = v
				else:
					args_file[k] = v
	else:
		args_file = args.__dict__		

	# Ensure that all required arguments are specified.
	def isNoneOrMissing(d, key):
		if key not in d:
			return True
		if d[key] is None:
			return True
		return False

	check = ["fine_zoom", "coarse_zoom"]
	for c in check:
		if isNoneOrMissing(args_file, c):
			raise Exception("Argument '%s' is missing."%c)

	if isNoneOrMissing(args_file, "contrast_range"):
		# Layer range must be specified and if it is, material_file must also be specified.
		if isNoneOrMissing(args_file, "layer_range"):
			raise Exception("contrast_range and layer_range parameter missing. " + \
				"You must specify at least one.")
		else:
			if isNoneOrMissing(args_file, "material_file"):
				raise Exception("layer_range was specified but material_file was not.")

	return type("arg_dictionary", (object,), args_file)


def do_something(items: List[str]) -> Tuple[int, int]:
	"""
	This function finds the minimum and maximum length string in the supplied list.
	:param items: A list of strings.
	:return: The minimum and maximum lengths.
	"""
	return min(len(i) for i in items), max(len(i) for i in items)


if __name__ == '__main__':

	#############################################
	# Argument Processing
	#############################################

	with open("_Scan.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)
	# We want to track every part of the scan in a meta data structure so that it can be refered to
	# later if necessary.
	meta_data = {
		'raw_arguments' : args.__dict__,
		'start_time'    : str(datetime.now()),
		'log'           : []
	}

	args = validateArgs(args)
	meta_data['processed_arguments'] = args.__dict__

	if args.layer_range is not None:
		args.n_layers_max = args.layer_range[1]
		with open(args.material_file, 'r') as file:
			material_file = json.loads(file.read())

		# We need to iterate over the "layers" member of this file and find the values corresponding
		# to the layer numbers specified by the user. Next we need to calculate how the RGB contrast
		# values should appear in a greyscale image. This is based on the way openCV converts 
		# RGB images to greyscale.
		layers = material_file['layers']
		temp   = []
		for l in layers:
			temp.append(
				[l[0], l[1], 0.299 * l[2] + 0.587 * l[3] + 0.114 * l[4]]
			)
		layers = temp

		min_val = [line[-1] for line in layers if line[0] == args.layer_range[0]][0]
		max_val = [line[-1] for line in layers if line[0] == args.layer_range[1]][0]

		# We now have the absolute contrast value of the requested layer numbers. We want to 
		# add a reasonable buffer zone around this value though, because the image will never
		# be exact. In order to do this we'll add half the average distance between layers 
		# as a buffer.
		_buffer = np.ediff1d([i[-1] for i in layers]).mean() / 2
		min_val -= _buffer
		max_val += _buffer

		min_val = max(0.07, min_val)

		args.contrast_range = [min_val, max_val]

		meta_data['log'].append(
			"The correct contrast range for layers in [%d, %d] was calculated to be [%f, %f]"%(
				args.layer_range[0], args.layer_range[1], min_val, max_val
			)
		)
		print(args.contrast_range)



	process_images = False
	if args.material_file is not None:
		process_images = True
		imageProcessor = MultiProcessImageProcessor(
			[args.fine_zoom[2], args.fine_zoom[1]], 
			args.n_processes, 
			meta_data
		)
		print("Initialized image processor")



	#############################################
	# Scan Setup
	#############################################

	# Check the output directory.
	if not os.path.exists(args.output_directory):
		os.mkdir(args.output_directory)
	else:
		for f in os.listdir(args.output_directory):
			p = os.path.join(args.output_directory, f)
			if "focus_calibration" not in p:
				if os.path.isfile(p):
					print("The specified output directory is not empty.")
					if input("Proceed (y/n)? ").lower() == 'y':
						break
					else:
						exit()

	skip_focus   = False
	focus_params = None
	if args.saved_focus_timeout > 0:
		# We need to see if there is a previous focus file in the output directory. If there is,
		# then we ask the user if they want to use it.
		fc_name = os.path.join(args.output_directory, "_focus_calibration.json")
		if os.path.isfile(fc_name):
			with open(fc_name, 'r') as file:
				focus_calibration = json.loads(file.read())

			if focus_calibration['args']['bounds'] == args.bounds:
				if input("Previous Focus Calibration Found, Use It? (y/n)? ").lower() == 'y':
					skip_focus   = True
					focus_params = focus_calibration['focus_params']

	# Estimate the size of the scanned area and process the limits arguments.
	x_min, x_max, y_min, y_max = args.bounds

	if args.focus_points is None:
		args.focus_points = [
			x_min, y_min,
			x_min, y_max,
			x_max, y_min,
			x_max, y_max,
			(x_min + x_max) / 2, (y_min + y_max) / 2
		]
		meta_data['log'].append(
			"No focus points were specified, the following focus points were used."
		)
		meta_data['log'].append(args.focus_points)

	scan_area = (x_max - x_min) * (y_max - y_min)
	meta_data['scan_area'] = scan_area

	print("Preparing to scan the region:")
	print("    x = %02.4f to %02.4f mm"%(x_min, x_max))
	print("    y = %02.4f to %02.4f mm"%(y_min, y_max))
	print(" area = %04.4f mm²"%scan_area)

	# Process the fine and coarse zoom setting information.
	fine_zoom,   fine_w,   fine_h,   fine_exposure   = args.fine_zoom
	coarse_zoom, coarse_w, coarse_h, coarse_exposure = args.coarse_zoom

	if input("Proceed (y/n)? ").lower() != 'y':
		exit()

	# Connect to the microscope hardware.
	microscope = MicroscopeController(verbose=True)
	microscope.camera.startCapture()

	# These values were determined to produce the best compromise between scan speed and image
	# quality.
	microscope.stage.setAcceleration(10, 10)
	microscope.stage.setMaxSpeed(7.5, 7.5)

	# Now we perform the focus calibration for the coarse scan. 
	microscope.camera.setExposure(coarse_exposure)
	microscope.focus.setZoom(coarse_zoom)


	#############################################
	# Coarse Scan Initialization
	#############################################

	if not skip_focus:
		interp, mx, my, b = calibrateFocus(microscope, args, debug=False, gfit=False)
		focus_calibration = {
			"args": {
				"bounds" : args.bounds
			},
			"focus_params" : [mx, my, b]
		}
		fc_name = os.path.join(args.output_directory, "_focus_calibration.json")
		with open(fc_name, 'w') as file:
			file.write(json.dumps(focus_calibration))
		print("Focus calibration file written.")
	else:
		interp = getFocusInterpolation(*focus_params)
		mx, my, b = focus_params

	# This is used to average multiple images together during the coarse scan.
	def avgimg(n, ds):
		imgs = []
		for i in range(n):
			img = microscope.camera.getFrame(downscale=ds)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			imgs.append(img)

		base = np.zeros(imgs[0].shape)
		for i in imgs:
			base += i

		return (base / n).astype(np.uint8)

	coarse_background = calculateBackgroundGreyscale(args, interp, microscope)

	# Export the background to an image so the user can reference it later.
	coarse_bg_image_name = os.path.join(args.output_directory, "_coarse_background.png")
	cv2.imwrite(coarse_bg_image_name, coarse_background)

	image_number         = 0
	x_current, y_current = x_min, y_min

	# Here we figure out if the scan area is larger on the x or y axis. We can make the scan faster
	# if the inner most loop goes over the largest axis.
	if x_max - x_min > y_max - y_min:
		swap = True
		outer_current = y_current
		outer_max     = y_max
		outer_width   = coarse_h
		outer_reset   = x_min

		inner_current = x_current
		inner_max     = x_max
		inner_width   = coarse_w

		meta_data['log'].append(
			"The x-axis range is larger, choosing this axis for the inner loop of the scan."
		)

	else:
		swap = False
		outer_current = x_current
		outer_max     = x_max
		outer_width   = coarse_w
		outer_reset   = y_min

		inner_current = y_current
		inner_max     = y_max
		inner_width   = coarse_h

		meta_data['log'].append(
			"The y-axis range is larger, choosing this axis for the inner loop of the scan."
		)

	# The focus motor performs better when it steps down across the sample instead of up. Here
	# we'll determine the slope of the sample in the chosen inner sweep direction and make sure
	# to move in the direction with a downward slope.
	sign = 1.0
	if swap:
		# inner sweep direction is along the x-axis
		if mx < 0:
			print("Sample slopes upwards, reversing direction.")
			meta_data['log'].append(
				"The sample slopes upwards in the positive direction of the inner loop axis."
			)
			meta_data['log'].append(
				"Reversing inner loop direction."
			)
			# The focus moves upwards in the positive x-direction, we need to move in the opposite
			# direction.
			outer_reset   = x_max
			sign          = -1.0
			inner_current = x_max
			def inner_condition():
				return inner_current > x_min
		else:
			def inner_condition():
				return inner_current < inner_max + inner_width
	else:
		if my < 0:
			print("Sample slopes upwards, reversing direction.")
			meta_data['log'].append(
				"The sample slopes upwards in the positive direction of the inner loop axis."
			)
			meta_data['log'].append(
				"Reversing inner loop direction."
			)
			outer_reset   = y_max
			sign          = -1.0
			inner_current = y_max
			def inner_condition():
				return inner_current > y_min
		else:
			def inner_condition():
				return inner_current < inner_max + inner_width
				

	# The initial position of the stage before the loop starts will depend on whether or not we
	# need to swap the direction of the inner loop.
	if swap:
		microscope.stage.moveTo(inner_current, outer_current)
		microscope.focus.setFocus(
			interp((inner_current, outer_current)), 
			corrected=True
		)
	else:
		microscope.stage.moveTo(outer_current, inner_current)
		microscope.focus.setFocus(
			interp((outer_current, inner_current)), 
			corrected=True
		)

	#############################################
	# Coarse Scan Execution
	#############################################

	# Start a coarse scan. We'll process each image as we acquire it and quickly decide whether or
	# not the region warrants further inspection.
	scan_w          = x_max - x_min
	scan_h          = y_max - y_min
	n_coarse_images = int(round((scan_w / coarse_w) * (scan_h / coarse_h)))
	coarse_progress = ProgressBar("Coarse Scan", 18, n_coarse_images, 1, ea=20)

	regions_of_interest  = []

	while outer_current < outer_max + outer_width:
		row_regions_of_interest = []
		while inner_condition():
			x, y = microscope.stage.getPosition()
			img  = avgimg(args.coarse_averages, args.coarse_downscale)
			if args.preview:
				pv   = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
				cv2.imshow('Scan Preview', pv)
				cv2.waitKey(1)
			
			# This function will return the coordinates of all of the regions within this image that
			# contain potentially interesting flakes.
			row_regions_of_interest.extend(getRegionsOfInterest(img, coarse_background, args, x, y))

			inner_current += sign*inner_width
			if swap:
				microscope.stage.moveTo(inner_current, outer_current)
				microscope.focus.setFocus(
					interp((inner_current, outer_current)), 
					corrected=args.quality_focus
				)
			else:
				microscope.stage.moveTo(outer_current, inner_current)
				microscope.focus.setFocus(
					interp((outer_current, inner_current)), 
					corrected=args.quality_focus
				)
			image_number += 1
			coarse_progress.update(image_number)

		regions_of_interest.append(row_regions_of_interest)
		inner_current  = outer_reset
		outer_current += outer_width
		if swap:
			microscope.stage.moveTo(inner_current, outer_current)
			microscope.focus.setFocus(
				interp((inner_current, outer_current)),
				corrected=True
			)
		else:
			microscope.stage.moveTo(outer_current, inner_current)
			microscope.focus.setFocus(
				interp((outer_current, inner_current)),
				corrected=True
			)

	coarse_progress.finish()

	#############################################
	# Fine Scan Initialization
	#############################################

	microscope.focus.setZoom(fine_zoom)
	microscope.camera.setExposure(fine_exposure)
	interp, mx, my, b = calibrateFocus(microscope, args, debug=False)

	total_images = sum([len(i) for i in regions_of_interest])

	fine_progress = ProgressBar("Fine Scan", 18, total_images, 1, ea=20)

	def avgimg(n):
		images = []
		for i in range(n):
			images.append(microscope.camera.getFrame(convert=False))
		r, g, b = [], [] ,[]
		for i in images:
			b.append(i[:, :, 0])
			g.append(i[:, :, 1])
			r.append(i[:, :, 2])

		red   = np.stack(r, axis=2).mean(axis=2)
		green = np.stack(g, axis=2).mean(axis=2)
		blue  = np.stack(b, axis=2).mean(axis=2)
		img   = np.stack([blue, green, red], axis=2).astype(np.uint8)
		return img

	fine_background = calculateBackgroundColored(args, interp, microscope)

	# Export the background to an image so the user can reference it later and so that the image
	# processing code can use it if it is run after this scan is complete.
	fine_bg_image_name = os.path.join(args.output_directory, "_fine_background.png")
	meta_data['fine_background_file'] = fine_bg_image_name
	cv2.imwrite(fine_bg_image_name, fine_background)

	meta_data['image_files'] = []

	#############################################
	# Fine Scan Execution
	#############################################

	image_idx = 0
	# We've finished the coarse scan. Now we'll zoom into the regions of interest that we found. 
	for row in regions_of_interest:
		if len(row) < 1:
			continue
		x0, y0 = row[0]
		microscope.stage.moveTo(x0, y0)
		microscope.focus.setFocus(
			interp((x0, y0)),
			corrected=True
		)
		for x, y in row:
			microscope.stage.moveTo(x, y)
			microscope.focus.setFocus(
				interp((x, y)),
				corrected=args.quality_focus
			)
			img = avgimg(args.fine_averages).astype(np.float32)

			fstr = "%04d_%2.4f_%2.4f.png"%(image_idx, x, y)
			meta_data['image_files'].append({
				'path'     : fstr,
				'position' : [x, y]	
			})

			filename = os.path.join(
				args.output_directory,
				fstr
			)

			cv2.imwrite(filename, img)
			image_idx += 1

			if process_images:
				# The fine background as determined by this program needs to be supplied to the 
				# image processor so it can correctly calculate optical contrast values.
				imageProcessor.addItem(filename, fine_background, args)
				#pass

			fine_progress.update(image_idx + 1)

	fine_progress.finish()

	#############################################
	# Cleanup
	#############################################

	meta_data['end_time'] = str(datetime.now())

	# if process_images:
	# 	pass
	# 	with open(os.path.join(args.output_directory, "_scan.json"), 'w') as file:
	# 		file.write(json.dumps(imageProcessor.getMetaData()))

	meta_data['processed_arguments'] = dict(meta_data['processed_arguments'])

	# We need to strip all of the weird __*__ type stuff from this dictionary so that it can
	# be json serialized.
	processed_args = {}
	for k, v in meta_data['processed_arguments'].items():
		if not k.startswith("__"):
			processed_args[k] = v

	raw_args = {}
	for k, v in meta_data['raw_arguments'].items():
		if not k.startswith("__"):
			raw_args[k] = v

	meta_data['processed_arguments'] = processed_args
	meta_data['raw_arguments']       = raw_args


	with open(os.path.join(args.output_directory, "_scan.json"), 'w') as file:
		file.write(json.dumps(meta_data))

	if process_images:
		try:
			imageProcessor.waitForCompletion()
		except Exception as ex:
			code.interact(local=locals())

	imageProcessor.buildDatabase(args)

	print("Copying the most relevant files into a subdirectory . . . ")
	dbname = os.path.join(args.output_directory, "_database.db")
	con    = sql.connect(dbname)
	cur    = con.cursor()

	subdir = os.path.join(args.output_directory, "filtered_images")
	if not os.path.exists(subdir):
		os.mkdir(subdir)

	filter_stmt = "WHERE L001_area > 25 order by L001_area DESC"
	stmt        = "SELECT file FROM flakes " + filter_stmt
	print("Filtering images with \"%s\""%stmt)
	res = cur.execute(stmt)
	for i, row in enumerate(res):
		outfile = "%06d_%s"%(i, row[0])
		outfile = os.path.join(subdir, outfile)

		infile = os.path.join(args.output_directory, row[0])
		copyfile(infile, outfile)

	print("Copied %d files to %s"%(i, subdir))

	microscope.camera.endCapture()
	cv2.destroyAllWindows()