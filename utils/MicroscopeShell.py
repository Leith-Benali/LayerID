from MicroscopeControl import MicroscopeController
from numbers           import Number, Real

import matplotlib.pyplot as plt
import numpy             as np

import readline
import cmd
import sys
import cv2
import code
from threading import Thread
import time


class MicroscopeShell(cmd.Cmd):
	intro  = "Microscope Control Shell v0.1\n"
	prompt = "microscope> "

	def __init__(self, *args, **kwargs):
		self.no_camera = kwargs['disable_camera']
		del kwargs['disable_camera']

		super(MicroscopeShell, self).__init__(*args, **kwargs)

		self.microscope =  MicroscopeController(
			verbose=True,
			disable_camera=self.no_camera
		)
		if not no_camera:
			self.microscope.camera.startCapture()

			self.stop = False
			self.preview_thread = Thread(
				target=self._preview_loop
			)

			self.preview_thread.start()

		self.locals = locals()

	def _preview_loop(self):
		while not self.stop:
			time.sleep(1 / 25)
			img = self.microscope.camera.getFrame()
			img = cv2.resize(img, (0, 0), fx=0.28, fy=0.28)
			cv2.imshow('Preview', img)
			cv2.waitKey(1)

		print("Terminating preview thread . . . ")
		cv2.destroyAllWindows()

	def do_autofocus(self, args):
		'Set the focus. Prints the current focus if no argument is given.'
		self.microscope.autoFocus([0.4, 0.6])

	def do_focus(self, args):
		'Set the focus. Prints the current focus if no argument is given.'
		args = self.parse(args)
		if len(args) == 0:
			print("current focus = %f"%self.microscope.focus.getFocus())
		elif len(args) == 1:
			self.microscope.focus.setFocus(args[0])
			print("current focus = %f"%self.microscope.focus.getFocus())
		else:
			print("invalid arguments")

	def do_zoom(self, args):
		'Set the zoom. Prints the current zoom if not argument is given.'
		args = self.parse(args)
		if len(args) == 0:
			print("current zoom = %f"%self.microscope.focus.getZoom())
		elif len(args) == 1:
			self.microscope.focus.setZoom(args[0])
			print("current zoom = %f"%self.microscope.focus.getZoom())
		else:
			print("invalid arguments")

	def do_move(self, args):
		'Move the stage to the specified coordinates.'
		args = self.parse(args)
		if not isinstance(args[0], Real) or not isinstance(args[1], Real):
			print("Arguments must be of type \'float\'.")
			return
		self.microscope.stage.moveTo(args[0], args[1])

		x, y = self.microscope.stage.getPosition()
		print("current position = %f, %f"%(x, y))

	def do_movedelta(self, args):
		'Move the stage position relative to the current position.'
		args = self.parse(args)
		if not isinstance(args[0], Real) or not isinstance(args[1], Real):
			print("Arguments must be of type \'float\'.")
			return
		self.microscope.stage.moveDelta(args[0], args[1])

		x, y = self.microscope.stage.getPosition()
		print("current position = %f, %f"%(x, y))

	def do_position(self, args):
		'Get the current position of the stage.'
		args = self.parse(args)

		x, y = self.microscope.stage.getPosition()
		print("current position = %f, %f"%(x, y))

	def do_joystick(self, args):
		'Set the coarse and fine mode sensitivity of the stage joystick.'
		args = self.parse(args)
		if not isinstance(args[0], Real) or not isinstance(args[1], Real):
			print("Arguments must be of type \'float\'.")
			return

		self.microscope.stage.setJoystickSensitivity(args[0], args[1])

	def do_exposure(self, args):
		"""Set or get the exposure. Specify \'auto\' to turn on continuous autoexposure."""
		args = self.parse(args)
		if len(args) == 0:
			exposure = self.microscope.camera.getExposure() * 1000
			print("exposure = %fms"%exposure)
		elif args[0] == 'auto':
			self.microscope.camera.enableAutoExposure()
			exposure = self.microscope.camera.getExposure() * 1000
			print("exposure = %fms"%exposure)
		else:
			self.microscope.camera.disableAutoExposure()
			self.microscope.camera.setExposure(args[0] / 1000)
			exposure = self.microscope.camera.getExposure() * 1000
			print("exposure = %fms"%exposure)

	def do_capture(self, args):
		"""Capture an image from the camera and save it to the specified location. Image will be displayed on screen if no path is given."""
		if self.no_camera:
			print("Camera is disabled.")
			return

		args = self.parse(args)
		if len(args) == 0:
			plt.imshow(self.microscope.camera.getFrame(convert=True))
			plt.show()
		else:
			img = self.microscope.camera.getFrame()
			try:
				cv2.imwrite(args[0], img)
			except Exception as ex:
				print("Failed to write image.")
				print("Error: %s"%ex)


	def do_py(self, args):
		"""Execute the given python code. Use \'self.microscope\' to reference the connected microscope."""
		exec(args, globals(), self.locals)
		self.locals.update(locals())

	def do_q(self, args):
		'Exit the shell.'
		if not no_camera:
			self.stop = True
		exit()
			

	def parse(self, args):
		arguments = []
		for arg in args.split(' '):
			try:
				arguments.append(int(arg))
				continue
			except:
				pass

			try:
				arguments.append(float(arg))
				continue
			except:
				pass

			arguments.append(arg)

		return [arg for arg in arguments if arg != ""]


if __name__ == '__main__':
	if '--no-camera' in sys.argv:
		no_camera = True
	else:
		no_camera = False

	MicroscopeShell(disable_camera=no_camera).cmdloop()


