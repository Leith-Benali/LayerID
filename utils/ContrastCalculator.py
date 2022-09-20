# Name:        FPContrastCalculator.py
# Author:      Adam Robinson (magnanimousllamacopter@gmail.com)
# Description: FPContrastCalculator - short for First Principles Contrast Calculator. Contains code
#              for calculating the optical contrast of stacks of arbitrary materials due to thin
#              film interference. This calculation is based heavily on work by Menon et. al. 
#              (Thanmay S Menon et al 2019 Nanotechnology 30 395704)
#              This file also contains helper functions for loading files containing the information
#              necessary to perform calculations.

import code
import json
import argparse
import sys 
import os
import time
import csv


import matplotlib.pyplot as plt
import numpy             as np

from scipy.integrate   import simps as simpson
from scipy.interpolate import interp1d

class ContrastCalculator:
	def __init__(self, **kwargs):
		self._validateArgs(**kwargs)

		# Calculate the angle bounds for the integration.
		NA = self.lens['NA']
		# self.angle_domain = np.linspace(
		# 	0.0, np.arcsin(NA / self.medium.real), self.wavelength_resolution
		# )
		self.angle_domain = np.linspace(
			0.0, np.arcsin(NA / self.medium.real), self.wavelength_resolution
		)

		# Calculate the wavelength bounds for integration. This is more involved.
		_min, _max = self._findWavelengthDomain()
		self.wavelength_domain = np.linspace(
			_min, _max, self.wavelength_resolution
		)

		# Now we need to interpolate all of the wavelength dependent data and resample it so that
		# the wavelength values are exactly the same for everything. This will dramatically speed
		# the calculation.
		self._resampleWavelengthData()

	def getContrast(self, substrate_idx):
		r_bg, g_bg, b_bg = self._getIntensity(substrate_idx)
		r_s,  g_s,  b_s  = self._getIntensity(0)

		r = -(r_bg - r_s) / r_bg
		g = -(g_bg - g_s) / g_bg
		b = -(b_bg - b_s) / b_bg

		return r, g, b

	def _getIntensity(self, idx):
		def getTransmissionAngle(t, n0, n1):
			return np.arcsin((n0 / n1) * np.sin(t))

		def partialReflection_p(t0, t1, n0, n1):
			numerator   = n1 * np.cos(t0) - n0 * np.cos(t1)
			denominator = n1 * np.cos(t0) + n0 * np.cos(t1)
			return numerator / denominator

		def partialReflection_s(t0, t1, n0, n1):
			numerator   = n0 * np.cos(t0) - n1 * np.cos(t1)
			denominator = n0 * np.cos(t0) + n1 * np.cos(t1)
			return numerator / denominator

		def reflectionCoefficient_p(t0, indices, w):
			n0  = self.medium
			n1  = indices[idx]
			t1  = getTransmissionAngle(t0, n0, n1)
			r0  = partialReflection_p(t0, t1, n0, n1)

			if idx == len(self.refractiveData) - 1:
				return r0

			phi = 4 * np.pi * n1 * np.cos(t1) * self.heights[idx] / w

			inner = _reflectionCoefficient_p(indices, t1, w, n1, idx + 1)

			ex = np.exp(-1j * phi)
			return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def _reflectionCoefficient_p(indices, t0, w, n0, i):
			N = len(self.refractiveData)

			n1 = indices[i]
			t1 = getTransmissionAngle(t0, n0, n1)
			r0 = partialReflection_p(t0, t1, n0, n1)

			if i == N - 1:
				return r0
			else:
				phi   = 4 * np.pi * n1 * np.cos(t1) * self.heights[i] / w
				inner = _reflectionCoefficient_p(indices, t1, w, n1, i + 1)
				ex    = np.exp(-1j * phi)

				return (r0 + inner * ex) / (1 + r0 * inner * ex)

		def reflectionCoefficient_s(t0, indices, w):
			n0  = self.medium
			n1  = indices[idx]
			t1  = getTransmissionAngle(t0, n0, n1)
			r0  = partialReflection_s(t0, t1, n0, n1)

			if idx == len(self.refractiveData) - 1:
				return r0

			phi = 4 * np.pi * n1 * np.cos(t1) * self.heights[idx] / w

			inner = _reflectionCoefficient_s(indices, t1, w, n1, idx + 1)

			ex = np.exp(-1j * phi)
			return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def _reflectionCoefficient_s(indices, t0, w, n0, i):
			N = len(self.refractiveData)

			n1 = indices[i]
			t1 = getTransmissionAngle(t0, n0, n1)
			r0 = partialReflection_s(t0, t1, n0, n1)

			if i == N - 1:
				return r0
			else:
				phi   = 4 * np.pi * n1 * np.cos(t1) * self.heights[i] / w
				inner = _reflectionCoefficient_s(indices, t1, w, n1, i + 1)
				ex    = np.exp(-1j * phi)

				return (r0 + inner * ex) / (1 + r0 * inner * ex)


		def innerIntegrand(t, indices, w):
			Rp = reflectionCoefficient_p(t, indices, w)
			Rs = reflectionCoefficient_s(t, indices, w)
			I = Rp.real**2 + Rp.imag**2 + Rs.real**2 + Rs.imag**2

			return I * self.sourceIntensity * np.sin(t)

		def angleIntegral(w):
			index   = np.where(self.wavelength_domain == w)[0][0]
			indices = np.array([layer[index] for layer in self.refractiveData])
			# Get the source intensity and the refractive index of each layer 
			# at this wavelength.
			y = innerIntegrand(self.angle_domain, indices, w)

			return simpson(y, self.angle_domain)

		rawIntensity = []
		for w in self.wavelength_domain:
			rawIntensity.append(angleIntegral(w))

		rawIntensity = np.array(rawIntensity)

		def getChannel(channel):
			channelIntensity = None
			if channel == 'r':
				channelIntensity = self.redResponse
			elif channel == 'g':
				channelIntensity = self.greenResponse
			elif channel == 'b':
				channelIntensity = self.blueResponse

			integrand = rawIntensity * channelIntensity * self.sourceSpectrum
			return simpson(integrand, self.wavelength_domain)

		return getChannel('r'), getChannel('g'), getChannel('b')

	# This checks the domain of all wavelength dependent data supplied by the user and finds the 
	# largest range of values that falls inside of the domain for all supplied data.
	def _findWavelengthDomain(self):
		wvMin = 0.0
		wvMax = 1.0

		for ni in self.materials:
			if not type(ni) is complex:
				# This is an array of per-wavelength values.
				lower = np.array(ni['lambda']).min()
				upper = np.array(ni['lambda']).max()
				wvMin = max(wvMin, lower)
				wvMax = min(wvMax, upper)


		# Check the wavelength domain of the camera data.
		R_min = np.array(self.camera['r']['lambda']).min()
		R_max = np.array(self.camera['r']['lambda']).max()
		G_min = np.array(self.camera['g']['lambda']).min()
		G_max = np.array(self.camera['g']['lambda']).max()
		B_min = np.array(self.camera['b']['lambda']).min()
		B_max = np.array(self.camera['b']['lambda']).max()

		wvMin = max(wvMin, R_min)
		wvMin = max(wvMin, G_min)
		wvMin = max(wvMin, B_min)

		wvMax = min(wvMax, R_max)
		wvMax = min(wvMax, G_max)
		wvMax = min(wvMax, B_max)

		# Check the wavelength domain of the source spectrum, if it is provided as wavelength 
		# dependent values. Otherwise it is a color temperature and the domain is (0, inf)
		if type(self.source['spectrum']) is tuple:
			# Check the source spectrum.
			sourceMin = np.array(self.source['spectrum']['lambda']).min()
			sourceMax = np.array(self.source['spectrum']['lambda']).max()

			wvMin = max(wvMin, sourceMin)
			wvMax = min(wvMax, sourceMax)

		# Constrain the bounds based on the range of wavelengths that the objective lens is 
		# transparent to.
		wvMin = max(wvMin, self.lens['spectral_domain'][0])
		wvMax = min(wvMax, self.lens['spectral_domain'][1])

		return [wvMin, wvMax]

	def _resampleWavelengthData(self):
		refractiveIndices = []
		for ni in self.materials:
			if type(ni) is complex:
				y = np.ones(self.wavelength_resolution) * ni
				refractiveIndices.append({'lambda' : x, 'n' : y})
			else:
				refractiveIndices.append(ni)

		sourceSpectrum = None
		# Now convert the source spectrum if only a color temperature was 
		# specified.
		if type(self.source['spectrum']) is not tuple:
			# If a string was specified for the spectrum, then we interpret it as 
			# <center>,<fwhm> for a gaussian spectrum.
			if type(self.source['spectrum']) is str:
				# Treat this as a FWHM and center wavelength for a "monochromatic" source.
				center, fwhm    = [float(i) for i in self.source['spectrum'].split(',')]
				s               = fwhm / np.sqrt(2 * np.log(2))
				def gaussian(x, s, x0):
					A = (1 / (s * np.sqrt(2*np.pi)))
					return A * np.exp(-np.square(x - x0) / (2 * np.square(s)))
				sourceSpectrum = (
					self.wavelength_domain, 
					gaussian(self.wavelength_domain, s, center)
				)
			else:
				h = 6.62607015e-34
				c = 299792458
				k = 1.380649e-23
				def Planck(wv, T):
					res = (2 * h * (c**2) / (wv**5))
					res = res * (1 / (np.exp((h * c) / (wv * k * T)) - 1))
					return res

				sourceSpectrum = (
					self.wavelength_domain, 
					Planck(self.wavelength_domain, self.source['spectrum'])
				)
		else:
			sourceSpectrum = self.source['spectrum']

		# We should now have all of functions with wavelength for an independent
		# variable in the form of arrays. Now we interpolate each of them and
		# use the interpolation to resample them so they all have the same 
		# wavelength values.

		self.refractiveData = []
		for ni in refractiveIndices:
			interp = interp1d(ni['lambda'], ni['n'], kind="cubic")
			self.refractiveData.append(interp(self.wavelength_domain))


		# Refractive index data is now in the proper form for efficient numeric
		# integration. Now we do the same for the color response curve of the
		# camera.

		Rinterp = interp1d(self.camera['r']['lambda'], self.camera['r']['I'], kind='cubic')
		Ginterp = interp1d(self.camera['g']['lambda'], self.camera['g']['I'], kind='cubic')
		Binterp = interp1d(self.camera['b']['lambda'], self.camera['b']['I'], kind='cubic')

		self.redResponse   = Rinterp(self.wavelength_domain)
		self.greenResponse = Ginterp(self.wavelength_domain)
		self.blueResponse  = Binterp(self.wavelength_domain)


		# We've now resampled the color responses as well. Next we handle the
		# spectrum of the source.
		spectrumInterp = interp1d(
			sourceSpectrum[0], 
			sourceSpectrum[1], 
			kind='cubic'
		)
		self.sourceSpectrum = spectrumInterp(self.wavelength_domain)

		# Calculate the source angle dependence from the parameter given.
		# self.sourceIntensity = np.exp(
		# 	-np.square(self.angle_domain) / (2 * np.square(self.source['angle_dependence']))
		# )
		self.sourceIntensity = np.exp(
			-2 * np.square(np.sin(self.angle_domain)) / (np.square(np.sin(self.angle_domain[-1])))
		)

		self._normalize()
		
	def _normalize(self):
		redConstant   = simpson(self.redResponse,   self.wavelength_domain)
		greenConstant = simpson(self.greenResponse, self.wavelength_domain)
		blueConstant  = simpson(self.blueResponse,  self.wavelength_domain)

		self.redResponse   = self.redResponse   / redConstant
		self.greenResponse = self.greenResponse / greenConstant
		self.blueResponse  = self.blueResponse  / blueConstant

		spectrumConstant = simpson(self.sourceSpectrum, self.wavelength_domain)
		self.sourceSpectrum = self.sourceSpectrum / spectrumConstant

		intensityConstant = simpson(self.sourceIntensity, self.angle_domain)
		self.sourceIntensity = self.sourceIntensity / intensityConstant

	def setSourceSpectrumTemperature(self, temperature):
		h = 6.62607015e-34
		c = 299792458
		k = 1.380649e-23
		def Planck(wv, T):
			res = (2 * h * (c**2) / (wv**5))
			res = res * (1 / (np.exp((h * c) / (wv * k * T)) - 1))
			return res

		self.sourceSpectrum = Planck(self.wavelength_domain, temperature)

	def setSourceSpectrumMonochrome(self, center, fwhm):
		s = fwhm / np.sqrt(2 * np.log(2))
		def gaussian(x, s, x0):
			A = (1 / (s * np.sqrt(2*np.pi)))
			return A * np.exp(-np.square(x - x0) / (2 * np.square(s)))

		self.sourceSpectrum = gaussian(self.wavelength_domain, s, center)

	def setImmersionMedium(self, medium):
		self.medium = medium

	def setHeights(self, heights):
		self.heights = heights

	# Performs basic validation, ensuring that all required arguments are supplied and providing
	# defaults for optional arguments that are not specified. This function will set all of the
	# necessary member variables for the rest of the initialization process.
	def _validateArgs(self, **kwargs):
		if 'materials' not in kwargs:
			raise Exception("'materials' argument is missing")
		else:
			self.materials = kwargs['materials']

		if 'heights' not in kwargs:
			raise Exception("'heights' argument is missing")
		else:
			self.heights = kwargs['heights']

		if 'camera' not in kwargs:
			raise Exception("'camera' argument is missing")
		else:
			self.camera = kwargs['camera']

		if 'source' not in kwargs:
			# 3200K color temperature and angle dependence parameter 0.1.
			source = {'spectrum': 3200, 'angle_dependence': 0.1}
		else:
			self.source = kwargs['source']

		if 'lens' not in kwargs:
			raise Exception("'lens' argument is missing")
		else:
			self.lens = kwargs['lens']

		if 'medium' not in kwargs:
			self.medium = complex(1.0003, 0.0) # Air
		else:
			self.medium = kwargs['medium']

		if 'wavelength_resolution' not in kwargs:
			self.wavelength_resolution = 256
		else:
			self.wavelength_resolution = kwargs['wavelength_resolution']

		if 'angle_resolution' not in kwargs:
			self.angle_resolution = 256
		else:
			self.angle_resolution = kwargs['angle_resolution']

class ContrastCalculatorLoader:
	def __init__(self, refraction, camera_file, source_spectrum, source_angle, **kwargs):
		def getCSVFloats(path):
			with open(path, 'r') as file:
				r = csv.reader(file, delimiter=",")
				data = list(r)

			rows = [[float(c) for c in r] for r in data[1:]]
			return rows


		refractive_data = []
		for n in refraction:
			try:
				c = complex(n)
				refractive_data.append(c)
			except:
				rows = getCSVFloats(n)
				wv   = np.array([i[0] for i in rows]) * 1e-6
				n0   = np.array([i[1] for i in rows])
				try:
					k0   = np.array([i[2] for i in rows])
				except:
					# No k data.
					k0 = np.ones(len(n0)) * 0.0

				n0 = [complex(n0[i], k0[i]) for i in range(len(n0))]
				refractive_data.append({'lambda': wv, 'n': n0})

		color_data = getCSVFloats(camera_file)

		wv = [i[0] for i in color_data]
		r  = [i[1] for i in color_data]
		g  = [i[2] for i in color_data]
		b  = [i[3] for i in color_data]
		camera = {
			'r': {'lambda': wv, 'I': r}, 
			'g': {'lambda': wv, 'I': g}, 
			'b': {'lambda': wv, 'I': b}
		}

		try:
			s = float(source_spectrum)
			source_spectrum = s
		except:
			if ',' not in source_spectrum:
				data = getCSVFloats(source_spectrum)
				wv   = [i[0] for i in data]
				I    = [i[1] for i in data]
				source_spectrum = {'lambda': wv, 'I': I}


		try:
			g = float(source_angle)
			source_angle = g
		except:
			data = getCSVFloats(source_angle)
			angle = [i[0] for i in data]
			I     = [i[1] for i in data]
			source_angle = (wv, I)

		source = {
			'spectrum'         : source_spectrum,
			'angle_dependence' : source_angle
		}

		self.args = {
			'materials' : refractive_data,
			'camera'    : camera,
			'source'    : source,
			**kwargs
		}

	def getCalculator(self):
		return ContrastCalculator(**self.args)


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
	# https://www.hindawi.com/journals/jnm/2014/989672/
	# The thickness of graphene as a function of layer number appears to be
	# d = 0.475*n -0.14 with an error that is well under a femto-meter.
	n = np.arange(10) + 1
	d = 0.475e-9*n - 0.14e-9

	# https://refractiveindex.info/ has entries for c-Plane HOPG and for actual graphene. In the
	# interest of making this calculation as accurate as possible. This code will use refractive
	# index data for the former when calculating for one layer and data from the latter for all 
	# subsequent layer calculations.
	calculator = ContrastCalculatorLoader(
		["materials/graphene.csv", "materials/quartz_thin_film.csv", "materials/silicon.csv"],
		"cameras/IMX264.csv",
		4000,
		0.1,
		heights=[3.34e-10, 90e-9],
		lens={
			'NA' : 0.42,
			'spectral_domain' : [
				435e-9,
				655e-9
			]
		},
		wavelength_resolution=256,
		angle_resolution=256
	).getCalculator()


	# Here we run the calculation for a monolayer. The '1' parameter passed to the calculator means
	# that material with index '1' should be considered the start of the substrate material (SiO2).
	r0, g0, b0 = calculator.getContrast(1)
	r,  g,  b  = [r0], [g0], [b0]

	# Here we initialize another calculator with c-Plane HOPG.
	calculator = ContrastCalculatorLoader(
		["materials/HOPG_c_plane.csv", "materials/quartz_thin_film.csv", "materials/silicon.csv"],
		"cameras/IMX264.csv",
		4000,
		0.1,
		heights=[3.34e-10, 90e-9],
		lens={
			'NA' : 0.42,
			'spectral_domain' : [
				435e-9,
				655e-9
			]
		},
		wavelength_resolution=256,
		angle_resolution=256
	).getCalculator()


	for i, d0 in enumerate(d[1:]):
		print("%d / %d"%(i + 1, len(d)))
		heights = calculator.heights
		heights[0] = d0
		calculator.setHeights(heights)
		r0, g0, b0 = calculator.getContrast(1)
		r.append(r0)
		g.append(g0)
		b.append(b0)

	layer_data = []
	for n0, d0, r0, g0, b0 in zip(n, d, r, g, b):
		layer_data.append("[%d, %E, %E, %E, %E]"%(n0, d0, r0, g0, b0))
	layer_data_string = ",\n".join(layer_data)

	print(layer_data_string)

	plt.scatter(n, r, color='red',   s=3)
	plt.scatter(n, g, color='green', s=3)
	plt.scatter(n, b, color='blue',  s=3)
	plt.xlabel("Number of Layers")
	plt.xticks(np.arange(10) + 1)
	plt.ylabel("Optical Contrast")
	plt.title(r"Optical Contrast vs. Number of Layers for Graphene/Graphite on 90nm $SiO_2$")
	plt.show()
