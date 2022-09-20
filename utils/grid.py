import matplotlib.pyplot as plt
import numpy             as np
import code
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid

def nBilateral(img, n, params):
	for i in range(n):
		img = cv2.bilateralFilter(img, 5, *params)

	return img

def nSharpen(img, n):
	sharpen_kernel = np.array([
		[-1, -1, -1], 
		[-1,  9, -1], 
		[-1, -1, -1]
	])
	for i in range(n):
		img = cv2.filter2D(img, -1, sharpen_kernel)

	return img

test_image = cv2.cvtColor(cv2.imread("0123_sub.png"), cv2.COLOR_BGR2RGB)
iterations = np.arange(10) * 4
sigmas     = np.arange(10) * 4
images     = []

fig  = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0)

for n in iterations:
	for s in sigmas:
		images.append(nBilateral(test_image, n, (s, 50)))

for axis, img in zip(grid, images):
	axis.imshow(img)


plt.savefig("test.png", dpi=400)