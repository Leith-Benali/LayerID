import matplotlib.pyplot as plt
import numpy             as np
import code

from scipy.optimize import curve_fit

with open("source_spectrum.txt", 'r') as file:
	raw = file.read()

lines = raw.split("\n")
cells = [i.split("\t")[:2] for i in lines]
data  = np.array([list(map(float, i)) for i in cells])

code.interact(local=locals())