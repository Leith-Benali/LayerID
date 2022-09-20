# Author:      Adam Robinson
# Description: This is a simple console progress bar that attempts to
#              estimate the amount of time remaining in an operation by
#              recording the time between steps.

import os
import sys
import numpy as np
import time
import shutil
from datetime import datetime

# Returns the dimensions of the current terminal window or a good guess if 
# something goes wrong.
def terminal_dims():
	s    = shutil.get_terminal_size()
	cols = s.columns
	rows = s.lines

	return rows, cols

class ProgressBar:
	def __init__(self, prefix, prefix_width, total, update_every=5, ea=15, skip=5):
		self.prefix         = prefix
		self.prefix_width   = prefix_width
		self.update_every   = update_every
		self.estimate_after = ea
		self.total          = total
		self.skip           = skip # Wait this many seconds to start collecting timing data
		                           # This is useful when a processor is auto-tuning its clock
		                           # rate, because the first few iterations may be slower and
		                           # not representative.
		self.current        = 0.0
		self.last           = 0.0
		self.remaining      = 0
		self.width          = terminal_dims()[1] - 62
		self.estimate       = True
		self.start_time     = datetime.now()
		self.update_count   = 0
		self.times          = [] # Timing of chunks of work.
		self.sizes          = [] # Amount of work in each chunk.
		self.display()

		self.last_time = time.time()


	def update(self, value):
		self.current      =  value
		self.update_count += 1

		if self.update_count % self.update_every == 0 or self.update_count == 1:
			work = int(self.current - self.last)
			t = (datetime.now() - self.start_time).seconds
			should_estimate = t > self.estimate_after
			if work != 0 and t > self.skip:
				self.last      = self.current
				timenow        = time.time()
				timing         = timenow - self.last_time
				self.last_time = timenow

				self.times.append(timing)
				self.sizes.append(work)

				# Convert to numpy arrays and calculate the average
				# time per unit work.
				times = np.array(self.times)
				works = np.array(self.sizes)

				avg = (times / works).mean()

				# Figure out how much work is left.
				self.remaining = self.total - self.current
				self.remaining = self.remaining * avg

			self.display(est=should_estimate)

	def finish(self):
		self.current = self.total
		total_time   = (datetime.now() - self.start_time).seconds
		seconds      = int(total_time % 60)
		minutes      = int(np.floor(total_time / 60))
		time         = ' (%02i:%02i elapsed)'%(minutes, seconds)
		self.display(_end='')
		print(time, end='')
		print('\n', end='')
		self.ttc = total_time

	# This function returns a tuple with the first member being the
	# percentage to display and the second number being the number
	# of ticks to draw in the progress bar.
	def get_display(self):
		percentage = (self.current / self.total) * 100
		ticks      = int(np.floor((self.current / self.total) * self.width))
		return (ticks, percentage)

	def display(self, est=False, _end='\r'):
		ticks, percentage = self.get_display()
		fill   = '='  * ticks
		space  = ' ' * (self.width - ticks)
		disp   = '%' + '%05.2f'%(percentage)

		rem_seconds = int(self.remaining)
		rem_minutes = rem_seconds // 60
		rem_seconds = rem_seconds % 60
		rem_hours   = rem_minutes // 60
		rem_minutes = rem_minutes % 60
		rem         = '%02i:%02i:%02i rem.'%(
			rem_hours, 
			rem_minutes, 
			rem_seconds
		)

		if self.current == self.total or not self.estimate or not est:
			rem = ''

		prefix = self.prefix + (' ' * (self.prefix_width - len(self.prefix)))


		# This is the only consistent way to clear the current line.
		# Using a \r character at the end of the line only works for
		# some environments. Odds are that this will not work on windows
		# but who cares.
		if os.name == 'nt':
			print("", end='\r')
		else:
			sys.stdout.write("\033[K")
		print(prefix + '[' + fill + space + ']' + ' ' + disp + ' ' + rem, end=_end)