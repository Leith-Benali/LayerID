import code
import sys
import json
import cv2
import argparse
import os

import sqlite3           as sql
import numpy             as np
import matplotlib.pyplot as plt

from shutil import copyfile

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


if __name__ == '__main__':
	with open("_FilterDatabase.json", 'r') as file:
		args_specification = json.loads(file.read())
	args  = preprocess(args_specification)

	dbname = os.path.join(args.directory, "_database.db")
	con    = sql.connect(dbname)
	cur    = con.cursor()

	if not os.path.exists(args.output):
		os.mkdir(args.output)

	res = cur.execute("SELECT file FROM flakes " + args.filter)
	for i, row in enumerate(res):
		outfile = "%06d_%s"%(i, row[0])
		outfile = os.path.join(args.output, outfile)

		infile = os.path.join(args.directory, row[0])
		copyfile(infile, outfile)

	print("Copied %d files"%i)