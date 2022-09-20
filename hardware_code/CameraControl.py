# Author:      Adam Robinson
# Description: This file contains a class that allows a user to programmatically
#              turn the auto-exposure feature of the Pixelink camera on and off.
#              It also contains a function for retrieving a frame from the camera
#              as a NumPy ndarray. 

from pixelinkWrapper import *
from ctypes          import *

import numpy as np
import cv2
import os
import code
import time
import atexit

# Default resolution for our camera is: 2448 x 2048
class CameraController:
	def __init__(self):
		ret = PxLApi.initialize(0)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Unable to initialize a camera.")

		self.camera_handle         = ret[1]
		self.pixel_type            = 0      # PT_COLOR
		self.raw_image_buffer      = create_string_buffer(5000 * 5000 * 2)
		self.auto_exposure_enabled = False
		self.current_res           = (2048, 2448, 3)

	# This needs to be called before capturing images from the camera.
	def startCapture(self):
		ret = PxLApi.setStreamState(
			self.camera_handle, 
			PxLApi.StreamState.START
		)

		if not PxLApi.apiSuccess(ret[0]):
			PxLApi.uninitialize(self.camera_handle)
			raise Exception("Unable to start the stream on the camera.")

	# This should be called when you are done capturing images.
	def endCapture(self):
		ret = PxLApi.setStreamState(
			self.camera_handle, 
			PxLApi.StreamState.STOP
		)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to end capture state.")

	def enableLowRes(self):

		ret = PxLApi.getFeature(self.camera_handle, PxLApi.FeatureId.PIXEL_ADDRESSING)
		if not PxLApi.apiSuccess(ret[0]):
			print("Unable to read current resolution settings.")

		params = ret[2]

		self.endCapture()
		params[PxLApi.PixelAddressingParams.MODE]    = PxLApi.PixelAddressingModes.DECIMATE
		params[PxLApi.PixelAddressingParams.X_VALUE] = 4
		params[PxLApi.PixelAddressingParams.Y_VALUE] = 4

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.PIXEL_ADDRESSING, 
			PxLApi.FeatureFlags.MANUAL, 
			params
		)

		if not PxLApi.apiSuccess(ret[0]):
			print(ret[0])
			raise Exception("Failed to set resolution.")

		self.current_res = (2048 // 4, 2448 // 4, 3)

		self.startCapture()

	def disableLowRes(self):
		ret = PxLApi.getFeature(self.camera_handle, PxLApi.FeatureId.PIXEL_ADDRESSING)
		if not PxLApi.apiSuccess(ret[0]):
			print("Unable to read current resolution settings.")

		params = ret[2]

		code.interact(local=locals())

		self.endCapture()
		params[PxLApi.PixelAddressingParams.MODE]    = PxLApi.PixelAddressingModes.DECIMATE
		params[PxLApi.PixelAddressingParams.X_VALUE] = 1
		params[PxLApi.PixelAddressingParams.Y_VALUE] = 1

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.PIXEL_ADDRESSING, 
			PxLApi.FeatureFlags.MANUAL, 
			params
		)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to set resolution.")

		self.current_res = (2048, 2448, 3)

		self.startCapture()

	# Turns on automatic exposure adjustment.
	def enableAutoExposure(self):
		exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
		params = [exposure]

		ret = PxLApi.getFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not read auto-exposure status.")

		params = ret[2]
		flags  = PxLApi.FeatureFlags.AUTO

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE, 
			flags, 
			params
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not set auto-exposure to on.")
		else:
			self.auto_exposure_enabled = True

	# Disabled automatic exposure adjustment.
	def disableAutoExposure(self):
		exposure = 0 # Intialize exposure to 0, but this value is ignored when initating auto adjustment.
		params = [exposure]

		ret = PxLApi.getFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not read auto-exposure status.")

		params = ret[2]
		flags  = PxLApi.FeatureFlags.MANUAL

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE, 
			flags, 
			params
		)
		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Could not set auto-exposure to off.")
		else:
			self.auto_exposure_enabled = False


	def __del__(self):
		self.cleanup()

	def cleanup(self):
		ret = PxLApi.uninitialize(self.camera_handle)


	def _api_range_error(self, rc):
		a = rc == PxLApi.ReturnCode.ApiInvalidParameterError
		b = rc == PxLApi.ReturnCode.ApiOutOfRangeError
		return a or b

	def getExposure(self):
		ret = PxLApi.getFeature(self.camera_handle, PxLApi.FeatureId.EXPOSURE)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to read exposure.")

		return ret[2][0]

	def setExposure(self, exposure):
		ret = PxLApi.getFeature(self.camera_handle, PxLApi.FeatureId.EXPOSURE)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to read exposure.")

		params    = ret[2]
		params[0] = exposure

		ret = PxLApi.setFeature(
			self.camera_handle, 
			PxLApi.FeatureId.EXPOSURE, 
			PxLApi.FeatureFlags.MANUAL, 
			params
		)

		if not PxLApi.apiSuccess(ret[0]) or self._api_range_error(ret[0]):
			raise Exception("Failed to set exposure.")

	# Returns a frame. By default, this will be in the BGR pixel format.
	# Set convert=True to get an RGB frame.
	def getFrame(self, convert=False, downscale=None):
		ret = PxLApi.getNextFrame(
			self.camera_handle, 
			self.raw_image_buffer
		)
		frameDesc = ret[1]

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to capture frame.")

		ret = PxLApi.formatImage(
			self.raw_image_buffer, 
			frameDesc, 
			PxLApi.ImageFormat.RAW_RGB24
		)

		if not PxLApi.apiSuccess(ret[0]):
			raise Exception("Failed to convert frame to 24 bit RGB.")

		formatedImage = ret[1]

		np_img = np.full_like(
			formatedImage, 
			formatedImage, 
			order="C"
		)
		np_img.dtype = np.uint8

		imageHeight = int(frameDesc.Roi.fHeight)
		imageWidth  = int(frameDesc.Roi.fWidth)
		#newShape    = (imageHeight, imageWidth, 3)
		newShape    = self.current_res

		np_img = np.reshape(np_img, newShape)

		if convert:
			np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

		if downscale is not None:
			np_img = cv2.resize(np_img, (0, 0), fx=(1 / downscale), fy=(1 / downscale))

		return np_img


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	camera = CameraController()
	code.interact(local=locals())




			
			



