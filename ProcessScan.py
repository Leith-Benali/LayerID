# Author:      Adam Robinson
# Description: This file contains the logic used by Scan.py to process images
#              as they are acquired. It can also be run as a standalone command
#              line program to process a folder of images that has not yet been
#              processed. This program will move images to a folder called 
#              "deleted", if they don't appear to conatain any flakes whose 
#              largest bounding box dimension is greater than or equal to 
#              the --rejection-threshold parameter (relative to image dimensions).
#
#              This program runs efficiently using multiple subprocesses. The speedup
#              is linearly related to the number of subprocesses up to the number of
#              cores on your systems CPU. 

import sys

from pyparsing import fname

sys.path.append("hardware_code")

import code
import sys
import os
import time
import cv2
import argparse
import json
import shutil
import numpy             as np
import matplotlib.pyplot as plt
import sqlite3           as sql

from datetime        import datetime
from multiprocessing import Pool
from ImageProcessing import processFile
from Progress        import ProgressBar

# Process the command line arguments supplied to the program.
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


# I know, weird name, but it does accurately and somewhat concisely describe what this code
# does.
class MultiProcessImageProcessor:
	def __init__(self, image_dims, n_processes=8, metadata=None, max_layers=3):
		self.n_processes  = n_processes

		if self.n_processes == 1:
			self.sync = True
		else:
			self.sync = False
			self.pool = Pool(n_processes)
		self.metadata     = metadata

		self.image_dims         = image_dims
		self.current_in_process = 0
		self.total_processed    = 0
		self.results            = []
		self.failures           = []

		self.metadata['image_processing'] = {
			'initiated' : str(datetime.now()),
			'files'     : {}
		}

	def addItem(self, item, bg, args):
		if self.sync:
			processFile(item, bg, self.image_dims, args)
		else:
			res = self.pool.apply_async(processFile, (item, bg, self.image_dims, args))
			self.results.append(res)
			self.current_in_process += 1


	def waitForCompletion(self):
		if self.sync:
			return
		pb = ProgressBar("Processing Images", 18, self.current_in_process, 1, ea=25)
		processed = 0
		while self.current_in_process > 0:
			done          = []
			for r in self.results:
				if r.ready():
					status, files            = r.get(0.01)
					self.total_processed    += 1
					self.current_in_process -= 1
					
					processed += 1
					pb.update(processed)
					done.append(r)

					if not status:
						self.failures.append(fname)

			for d in done:
				self.results.remove(d)

			time.sleep(0.001)
		pb.finish()

	def getMetaData(self):
		return self.metadata

	def buildDatabase(self, args):
		print("Building database ... ", end='')
		dbname = os.path.join(args.output_directory, "_database.db")
		con    = sql.connect(dbname)
		cur    = con.cursor()
		path   = args.output_directory

		layer_columns = ['L%03d_area REAL'%(i + 1) for i in range(args.n_layers_max)]
		column_spec   = "(id INT PRIMARY KEY, file TEXT, geom_idx INT, area REAL, %s)"%(
			", ".join(layer_columns)
		)

		cur.execute("CREATE TABLE flakes %s"%column_spec)

		files = []
		for entry in os.listdir(path):
			file = entry.replace("\\", "/").split("/")[-1]
			ext  = ".".join(file.split(".")[-2:])
			if ext == "stats.json":
				files.append(os.path.join(path, entry))

		current_id = 0
		for i, file in enumerate(files):
			with open(file, 'r') as f:
				data = json.loads(f.read())

			for flake in data['flakes']:
				columns       = "id, %s"%(", ".join(flake.keys()))
				value_strings = []
				for v in flake.values():
					if type(v) is str:
						value_strings.append("'%s'"%v)
					else:
						value_strings.append("%.2f"%float(v))

				values    = "%d, %s"%(current_id, ", ".join(value_strings))
				statement = "INSERT INTO flakes (%s) VALUES (%s)"%(columns, values)
				cur.execute(statement)
				current_id += 1

			print("file %d / %d"%(i + 1, len(files)))

		con.commit()
		con.close()

		print("Done")


def main():
	# Load the arguments file.
	with open("_ProcessScan.json", 'r') as file:
		args_specification = json.loads(file.read())
	args = preprocess(args_specification)

	# We need to load the background file for these images.
	metadata_fname = os.path.join(args.output_directory, "_scan.json")
	with open(metadata_fname, 'r') as file:
		metadata = json.loads(file.read())

	fine_bg_name = os.path.join(args.output_directory, metadata['fine_background_file'])
	bg_image = cv2.imread(fine_bg_name)

	files = [os.path.join(args.output_directory, entry['path']) for entry in metadata['image_files']]

	print("Processing %d files using %d processes." % (
		len(files),
		args.n_processes
	))

	imageProcessor = MultiProcessImageProcessor(args.image_dims, args.n_processes, metadata, args.n_layers_max)

	for file in files:
		imageProcessor.addItem(file, bg_image, args)

	imageProcessor.waitForCompletion()

	metadata = imageProcessor.getMetaData()

	with open(os.path.join(args.output_directory, "_scan.json"), 'w') as file:
		file.write(json.dumps(metadata))

	imageProcessor.buildDatabase(args)


# When this program is run standalone we just want it to load all image files in the output
# directory and process them.
if __name__ == '__main__':
	main()
