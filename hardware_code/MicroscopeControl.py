# Author:      Adam Robinson
# Description: This class combines the functionality of the CameraController,
#              FocusController and StageController classes into a cohesive
#              class that performs all of the relevant work related to controlling
#              the microscope. 

from CameraControl  import CameraController
from FocusControl   import FocusController
from StageControl   import StageController
from FocusControl   import autoConnect     as getFocusController
from StageControl   import autoConnect     as getStageController
from scipy.optimize import curve_fit
from Progress       import ProgressBar

import time
import code
import os
import cv2

import matplotlib.pyplot as plt
import numpy             as np


class MicroscopeController:
	def __init__(self, **kwargs):
		"""This will initialize a microscope controller object and connect to all
		of the devices necessary to get things working. The following keyword
		arguments are valid:
		    focus_port     : The COM port to use when connecting to the focus controller
		    focus_baudrate : The baudrate to use when connecting to the focus controller
		    stage_port     : The COM port to use when connecting to the stage controller
		    stage_baudrate : The baudrate to use when connecting to the stage controller
		    verbose        : Print information to the console when an action is being performed

		If the connection parameters for the focus and stage controllers are not
		specified, this class will attempt to connect to them automatically. If they
		cannot be connected to, an exception will be thrown."""
		self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False

		if self.verbose:
			print("Connecting to focus controller . . . ", end='', flush=True)
		if 'focus_port' in kwargs or 'focus_baudrate' in kwargs:
			fp = kwargs['focus_port']
			fb = kwargs['focus_baudrate']
			self.focus = FocusController(fp, fb)
		else:
			self.focus = getFocusController()
			if self.focus is None:
				print("FAIL")
				raise Exception("Failed to connect to focus controller.")

		if self.verbose:
			print("DONE")

		if self.verbose:
			print("Connecting to stage controller . . . ", end='', flush=True)
		if 'stage_port' in kwargs or 'stage_baudrate' in kwargs:
			sp = kwargs['stage_port']
			sb = kwargs['stage_baudrate']
			self.stage = StageController(sp, sb)
		else:
			self.stage = getStageController()
			if self.stage is None:
				print("FAIL")
				raise Exception("Failed to connect to stage controller.")

		if self.verbose:
			print("DONE")

		if 'disable_camera' in kwargs:
			self.disable_camera = kwargs['disable_camera']
		else:
			self.disable_camera = False

		if not self.disable_camera:
			if self.verbose:
				print("Connecting to camera           . . . ", end='', flush=True)
			self.camera = CameraController()
			if self.verbose:
				print("DONE")

	def __del__(self):
		self.cleanup()

	def cleanup(self):
		self.focus.cleanup()
		self.stage.cleanup()

	def autoFocus(self, _range, ndiv=100, passes=1, navg=3, autoExpose=False, gfit=False, debug=False):
		if ndiv < 5:
			raise Exception("You must specify ndiv >= 5.")

		# Save the current focus motor settings so that we can set them
		# when we are done.
		focus_accel     = self.focus.getFocusAcceleration()
		focus_initial_v = self.focus.getFocusInitialVelocity()
		focus_final_v   = self.focus.getFocusMaxVelocity()

		# Save the auto exposure status so that it can be reset.
		exposure_on = self.camera.auto_exposure_enabled
		if exposure_on:
			self.camera.disableAutoExposure()

		# Set the initial and final velocities to be a very small
		# portion of the limit value, so that the motor will move
		# really slowly.
		self.focus.setFocusInitialVelocity(50)
		self.focus.setFocusMaxVelocity(500)

		self.camera.startCapture()

		if autoExpose:
			self.camera.enableAutoExposure()
			print("Adjusting exposure")
			time.sleep(2)
			print("Done")
			self.camera.disableAutoExposure()

		current_range = _range
		for p in range(passes):
			if self.verbose:
				print("Range: [%2.4f, %2.4f]"%(current_range[0], current_range[1]))

			steps = np.linspace(current_range[0], current_range[1], ndiv)

			self.focus.setFocus(0.0)

			# This is used to compute the average of n images in order to 
			# reduce noise.
			def avgimg(n):
				imgs = []
				# TODO: Figure out how to capture the images as greyscale so that
				# we won't have to convert them.
				for i in range(n):
					img = self.camera.getFrame(downscale=3)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					imgs.append(img)

				base = np.zeros(imgs[0].shape)
				for i in imgs:
					base += i

				return (base / n).astype(np.uint8)

			pb1 = ProgressBar("Scanning Focus Range", 30, len(steps), 1, ea=120)
			idx = 0

			images    = []
			positions = []
			for position in steps:
				self.focus.setFocus(position, corrected=False)
				img = avgimg(navg)
				images.append(img)

				# These motors are imperfect, so we can't trust the focus to
				# be exactly what we set it to. We need to read it.
				current_focus = self.focus.getFocus()
				positions.append(current_focus)

				idx += 1
				pb1.update(idx)

			pb1.finish()

			pb2 = ProgressBar("Denoising and Processing", 30, len(images), 1, ea=2)
			idx = 0

			max_laplace  = []
			for img in images:
				# Convert the images to floating point format so that we can
				# get the best possible precision on the Laplacians.
				
				img = cv2.fastNlMeansDenoising(img, 20)
				#lpl = cv2.Laplacian(img, -1, ksize=3)
				#max_laplace.append(lpl.max())
				# lpl = cv2.Laplacian(img, -1, ksize=3)
				# max_laplace.append(np.percentile(lpl.flatten(), 99))
				lpl = cv2.Laplacian(img, -1, ksize=3)
				max_laplace.append(lpl.mean())
				
				idx += 1
				pb2.update(idx)

			pb2.finish()

			if gfit:
				max_laplace = np.array(max_laplace)
				steps       = np.array(steps)
				def fn(x, a, b, c, d):
					return a*np.exp(-np.square(x - b) / np.square(c)) + d

				a0 = max_laplace.max() - max_laplace.min()
				b0 = (steps[-1] + steps[0]) / 2
				c0 = 0.00821672789208
				d0 = max_laplace.min()
				# def test(x):
				# 	return a0*np.exp(-np.square(x - b0) / np.square(c0)) + d0

				# plt.plot(steps, max_laplace)
				# plt.plot(steps, test(steps))
				# plt.show()
				res, cov = curve_fit(fn, steps, max_laplace, p0=[a0, b0, c0, d0])
				focus_position = res[1]
				if debug:
					code.interact(local=locals())
			else:
				# Select the image with the highest maximum value for its Laplacian
				best_idx = np.argmax(max_laplace)
				_max = np.array(max_laplace).max()
				
				focus_position = positions[best_idx]

			# Modify the focus range in case there is another pass to run.
			width          = current_range[1] - current_range[0]
			current_range  = (focus_position - width / 5, focus_position + width / 5)
			

		self.focus.setFocus(focus_position)

		if debug:
			print("DEBUG = True (MicroscopeController.autoFocus)")
			code.interact(local=locals())

		# If auto exposure was on when this was called, turn it back on.
		if exposure_on:
			self.camera.enableAutoExposure()

		# Reset the motion paramters to their original values.
		self.focus.setFocusAcceleration(focus_accel)
		self.focus.setFocusInitialVelocity(focus_initial_v)
		self.focus.setFocusMaxVelocity(focus_final_v)

		return focus_position

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import sys
	def showimg():
		microscope.camera.startCapture()
		img = microscope.camera.getFrame(convert=True)
		plt.imshow(img)
		plt.show()
		microscope.camera.endCapture()

	camera_on = True
	if len(sys.argv) > 1:
		if sys.argv[1] == '--no-camera':
			camera_on = False

	microscope = MicroscopeController(
		verbose=True,
		disable_camera=(not camera_on)
	)

	def exit():
		os._exit(0)

	code.interact(local=locals())
