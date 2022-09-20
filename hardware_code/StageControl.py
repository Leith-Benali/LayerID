# Authors:     Adam Robinson, Cicely Motamedi
# Description: This class controls motion of the XY stage. It also contains
#              a helper function for automatically recognizing and connecting
#              to the stage.

import serial
import sys
import time
import threading
import code

from queue import SimpleQueue, Empty

# This handles the motion of the motorized stage as well as reading
# data from the stage (in order to determine stage position).
# Long running operations have an optional parameter called "callback".
# When specified, these operations will return immediately and will call
# this callback function when complete. Operations that return a value
# will pass that value to the callback function instead, when specified.
class StageController:
	def __init__(self, port="COM3", baud=9600, timeout=0):
		self.connection = serial.Serial(port, baud, timeout=timeout)

		if not self.connection.is_open:
			raise Exception("Failed to open serial connection.")

		# Read the response buffer and get rid of it. It may be filled
		# with an error response from the system attempting to connect to
		# the focus controller and sending commands to it.
		self.connection.write(b"\r")
		self.readResponse()


		# This will prevent the system from hitting the supports that hold
		# the optics.
		try:
			self.setLimits([-50, 50], [-50, 37])
		except Exception as ex:
			self.cleanup()
			raise Exception("Error setting stage limits.") from ex

		self.task_queue  = SimpleQueue()
		self.task_thread = threading.Thread(
			target=self._do_tasks,
			name="Stage Task Thread",
			daemon=True
		)
		self.exiting     = False
		self.task_thread.start()

	# This function continuously waits for tasks from the user and
	# executes them in first-in-first-out order. It guarentees that
	# tasks will not overlap and will be executed sequentially. 
	def _do_tasks(self):
		while not self.exiting:
			try:
				fn, args, kwargs, cb = self.task_queue.get(block=True, timeout=0.01)
			except Empty as ex:
				# This just means that we couldn't retrieve the task within
				# the alloted timeout. It should work on the next loop 
				# iteration.
				continue

			# If we get to here, there is a task to execute. We will run it
			# synchronously within this thread.
			try:
				result = fn(*args, **kwargs)
				cb(None, result)
			except Exception as ex:
				cb(ex, None)

			time.sleep(0.150)

	def __del__(self):
		self.cleanup()

	def cleanup(self):
		try:
			self.exiting = True
			if self.connection.is_open:
				self.connection.close()
		except:
			pass

	def readResponse(self, timeout=2):
		start           = time.time_ns()
		response_string = ""
		char            = self.connection.read(1).decode("ascii")
		while char != '\r':
			response_string += char
			char             = self.connection.read(1).decode("ascii")
			if (time.time_ns() - start) / 1e9 > timeout:
				raise IOError("Controller took too long to respond.")

		char = self.connection.read(1).decode("ascii")

		while char == '':
			char = self.connection.read(1).decode("ascii")
			if (time.time_ns() - start) / 1e9 > timeout:
				raise IOError("Controller took too long to respond.")

		if char != '\n':
			raise IOError("Incomplete response from controller.")

		return response_string

	# Returns True if the axis is moving and false otherwise.
	def getAxisStatus(self, axis):
		if axis == 'x':
			command_string = b"RB X \r"
		elif axis == 'y':
			command_string = b"RB Y \r"
		else:
			raise Exception("Unrecognized axis specified.")

		self.connection.write(command_string)

		response = self.readResponse()
		status   = ord(response[1])

		moving = (status & 0b00000001)

		return moving == 1

	# This function will block until both axes are in the idle state.
	# It will user the serial connection to check the axis status every
	# "interval" seconds and will throw an exception after "timeout" seconds,
	# if both axes have not reached the idle state.
	def _wait_for_idle(self, interval=0.15, timeout=30):
		start = time.time_ns()

		while True:
			if self.getAxisStatus('x') or self.getAxisStatus('y'):
				time.sleep(interval)
			else:
				break

			if (time.time_ns() - start) / 1e9 >= timeout:
				msg = """Timeout of %f seconds exceeded while waiting for motor 
				to reach idle state."""
				raise Exception(msg%timeout)

	# Move the stage position to the given value in millimeters.
	def moveTo(self, x, y, cb=None, return_immediately=False):
		if cb is None:
			return self._moveTo(x, y, return_immediately=return_immediately)
		else:
			self.task_queue.put((
				self._moveTo, [x, y], {'return_immediately': return_immediately}, cb
			))

	def _moveTo(self, x, y, return_immediately=False):
		x = int(round(x * 10000))
		y = int(round(y * 10000))

		command_string = b"MOVE X=%d Y=%d \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

		if return_immediately:
			return
		self._wait_for_idle()
		

	def moveDelta(self, x, y, cb=None):
		if cb is None:
			return self._moveDelta(x, y)
		else:
			self.task_queue.put((
				self._moveDelta, [x, y], {}, cb
			))

	def _moveDelta(self, x, y):
		command_string = b"R X=%d Y=%d Z \r"%(
			int(round(x * 10000)), 
			int(round(y * 10000))
		)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

		self._wait_for_idle()
		

	# Set the amount of time in milliseconds that it should take for the
	# axis to go from the start velocity to the maximum velocity.
	def setAcceleration(self, x, y, cb=None):
		if cb is None:
			return self._setAcceleration(x, y)
		else:
			self.task_queue.put((
				self._setAcceleration, [x, y], {}, cb
			))

	def _setAcceleration(self, x, y):
		if not isinstance(x, int) or not isinstance(y, int):
			raise Exception("This function takes integers (milliseconds) as arguments.")

		command_string = b"AC X=%d Y=%d \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Sets the sensitivity of the hardware joystick attached to the controller.
	# The coarse argument applies when fine control has been toggled off with the
	# button in the center of the joystick. The fine argument applies when it has
	# been toggled on. Arguments are in percent of the normal sensitivity.
	def setJoystickSensitivity(self, coarse, fine, cb=None):
		if cb is None:
			return self._setJoystickSensitivity(coarse, fine)
		else:
			self.task_queue.put((
				self._setJoystickSensitivity, [coarse, fine], {}, cb
			))

	def _setJoystickSensitivity(self, coarse, fine):
		if coarse < 0.1 or coarse > 100 or fine < 0.1 or fine > 100:
			raise Exception("Sensitivity values must be in the range [0.1, 100]")


		command_string = b"JS X=%2.2f Y=%2.2f \r"%(coarse, fine)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Sets the maximum speed of the stage on each axis.
	def setMaxSpeed(self, x, y, cb=None):
		if cb is None:
			return self._setMaxSpeed(x, y)
		else:
			self.task_queue.put((
				self._setMaxSpeed, [x, y], {}, cb
			))

	def _setMaxSpeed(self, x, y):
		if x > 7.5 or x < 0.0 or y > 7.5 or y < 0.0:
			raise Exception("Values must be in the range [0, 7.5] mm/s")

		command_string = b"S X=%1.6f Y=%1.6f \r"%(x, y)

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Given a range of valid values on the x and y axes,
	# this function will set limits on the motion of the stage.
	# These limits will be enforced by the controller.
	def setLimits(self, x, y):
		x = [int(round(x[0])), int(round(x[1]))]
		y = [int(round(y[0])), int(round(y[1]))]

		command_string = b"SETLOW X=%d Y=%d \r"%(x[0], y[0])

		self.connection.write(command_string)
		response      = self.readResponse()
		response_data = self._parse_response(response)

		command_string = b"SETUP X=%d Y=%d \r"%(x[1], y[1])

		self.connection.write(command_string)
		response      = self.readResponse()
		response_data = self._parse_response(response)

	# Resets the position of the stage.
	def home(self, cb=None):
		if cb is None:
			return self._home()
		else:
			self.task_queue.put((
				self._home, [], {}, cb
			))

	def _home(self):
		self.connection.write(b'MOVE X=0 Y=0 \r')
		response      = self.readResponse()
		response_data = self._parse_response(response)

		self._wait_for_idle()
		


	# Returns the X and Y positions in millimeters.
	def getPosition(self, cb=None):
		if cb is None:
			return self._getPosition()
		else:
			self.task_queue.put((
				self._getPosition, [], {}, cb
			))
			
	def _getPosition(self):
		command_string = b"WHERE X Y Z \r"

		self.connection.write(command_string)

		response      = self.readResponse()
		response_data = self._parse_response(response)

		X_str, Y_str = response_data.split(" ")

		return float(X_str) / 10000, float(Y_str) / 10000

	def _parse_response(self, response):
		if response[0] != ':':
			raise IOError("Malformed response from controller. (%s)"%(response))

		if response[1] == 'A':
			response_data = ""
			idx           = 2
			while idx < len(response):
				response_data += response[idx]
				idx           += 1

			return response_data.strip()
		elif response[1] == 'N':
			valid_numbers       = '0123456789'
			response_error_code = ""
			if response[3] in valid_numbers:
				response_error_code += response[3]
				if response[4] in valid_numbers:
					response_error_code += response[4]
			else:
				raise IOError("Controller responded with error but did not specify error code. (%s)"%(response))

			response_error_code = int(response_error_code)
			raise Exception("Controller responded with error code %d (%s)"%(response_error_code, response))
		else:
			raise IOError("Malformed response from controller. (%s)"%response)

# This class will attempt to connect to the stage controller, assuming
# that it is plugged in. It does this by querying every serial controller
# it can find and attempting to issue an innocuous command to it. If the
# controller responds appropriately, it is assumed to be the stage controller.
# This function will return a connected and ready to go StageController object
# if it succeeds and None otherwise.
def autoConnect():
	# Build a list of ports to try.
	ports = ["COM4", "COM1", "COM2", "COM3", "COM0"]

	for i in range(5, 10):
		ports.append("COM%d"%i)

	for p in ports:
		try:
			# This class always set motion limits on the stage upon
			# initialization, so it will throw an exception if we have 
			# connected to something that isn't the controller.
			controller = StageController(p, 9600)
			return controller
		except Exception as ex:
			raise ex

	return None

if __name__ == '__main__':
	port       = sys.argv[1]
	baud       = int(sys.argv[2])
	controller = StageController(port, baud)

	code.interact(local=locals())