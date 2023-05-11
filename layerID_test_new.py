"""Process optical images of thin flakes to distinguish layer thicknesses.
    Usage:
        - Change img_file (line 18) to the file location of image.
        - Adjust comp (line 64): higher value runs faster with less resolution."""

from read_npz import npz2dict
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture as GMM
from background import stack_opened, residual_img
from utils import return_opened


def line2d(x, y, coeffs=[1]*3, return_coeff=False):
    """Returns the result of a plane, or returns the coefficients"""
    a0 = (x*0+1)*coeffs[0]
    a1 = x*coeffs[1]
    a2 = y*coeffs[2]
    if return_coeff:
        return a0, a1, a2
    else:
        return a0+a1+a2

def make_ket(lst):
    """Turn a list of values into a ket (Dirac notation)"""
    return (np.array(lst)).reshape(-1,1)

def make_bra(lst):
    """Turn a list of values into a bra (Dirac notation)"""
    return (np.array(lst)).reshape(1,-1)

def multivar_gauss(data, mean, cov, cluster_count):
    """Multivariate Gaussian distribution function for GMM calculation.
        data: [blue value, green value, red value]
        mean: mean vector of one cluster
        cov: covariance matrix of one cluster
        cluster_count: number of clusters"""
    data = np.array(data)
    mean = np.array(mean)
    return (((2*np.pi)**cluster_count) * np.linalg.det(cov) * np.exp(make_bra(data-mean).dot((np.linalg.inv(cov)).dot(make_ket(data-mean)))))**(-0.5)

def discreteCB(fig, ax, img, flake_name):
    """Gives a discrete colorbar to a color map"""
    cluster_count = np.nanmax(img).astype(np.uint8) + 1
    cmp = mpl.cm.get_cmap('viridis')
    cmap = mpl.colors.ListedColormap([cmp(ii/cluster_count) for ii in range(cluster_count)])
    bounds = np.linspace(0,cluster_count,cluster_count+1)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=256)
    pimg = ax.imshow(img, norm=norm)
    cbar = fig.colorbar(pimg, ax=ax, ticks=[ii+0.5 for ii in range(cluster_count)])
    cbar.ax.set_yticklabels([ii for ii in range(cluster_count)])
    plt.savefig(f'.\\Flakes\\accuracy_test_{flake_name}.png')

def testing(img_file, flake_name, file_dir, crop, masking,
            master_cat_file, cluster_count, comp_rate=300):
    """
    Identify thickness of flake "img_file" using "master_cat_file"
    Parameters
    ----------
    img_file : str
        Location of sample image file. (i.e., "...\\RSGR003\\All\\4A1.jpg")
    flake_name : str
        Name of sample image. (i.e., "RSGR003_4A1")
    crop : list of ints of form [miny, maxy, minx, maxx]
        Region to crop sample image to. (i.e., [500,1300, 1000,1600])
    masking : list of list of ints of form [[miny1,maxy1,minx1,maxx1], ...]
        Regions of substrate to fit background. Indices relative to cropped
        image. (i.e., [[500,-1, 0,100], [700,-1, 450,-1],[0,300, 320,-1]])
    master_cat_file : str
        Location of master catalog npz file for the same material/substrate as
        sample. (i.e., "...\\Graphene_on_SiO2_master_catalog.npz")
    cluster_count : int
        Number of layers to fit up to. Determine based on how many layers
        "master_cat_file" was trained to.
    comp_rate : int [default: 300]
        Factor in calculating the compression of the normalized image.
        Compression is comp = sqrt(pixels in cropped image)/comp_rate
    """

    tic = time.perf_counter()

        ## Import and pre-processing
    ## Image import
    individual_loc = img_file

    # Median filter background subtraction
    all_opened = return_opened(file_dir)
    print(type(all_opened[0]))
    stacked = stack_opened(all_opened)
    opened_img, residual, type_ = residual_img(stacked, individual_loc)
    width = int(cv2.imread(img_file).shape[1])
    height = int(cv2.imread(img_file).shape[0])
    dim = (width, height)
    resized = cv2.resize(residual, dim, interpolation = cv2.INTER_AREA)

    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    ## Bilateral filtering
    imgc = img[crop[0]:crop[1], crop[2]:crop[3]]
    img_bl = (imgc).astype(np.float32) / 256
    for ii in range(1):
        img_bl = cv2.bilateralFilter(img_bl, 1, 1, 1)


    mask = np.ones(img_bl[:, :, 0].shape)
    for reg in masking:
        mask[reg[0]:reg[1], reg[2]:reg[3]] = 0


    # plt.imshow(img_bl)
    # plt.show()


    ## Fit to background based on pixels outside the flake.
    y_dim, x_dim, _ = img_bl.shape
    R = img_bl[:,:,0].flatten()
    G = img_bl[:,:,1].flatten()
    B = img_bl[:,:,2].flatten()
    X_, Y_ = np.meshgrid(np.arange(x_dim),np.arange(y_dim))
    X = X_.flatten()
    Y = Y_.flatten()
    sub_loc = ((mask.flatten())==0).nonzero()[0]
    Rsub = R[sub_loc]
    Gsub = G[sub_loc]
    Bsub = B[sub_loc]
    Xsub = X[sub_loc]
    Ysub = Y[sub_loc]

    Asub = np.array([*line2d(Xsub, Ysub, return_coeff=True)]).T

    Rcop,_,_,_ = np.linalg.lstsq(Asub, Rsub, rcond=None)
    Gcop,_,_,_ = np.linalg.lstsq(Asub, Gsub, rcond=None)
    Bcop,_,_,_ = np.linalg.lstsq(Asub, Bsub, rcond=None)

    Rfitp = line2d(X, Y, coeffs=[*Rcop])
    Gfitp = line2d(X, Y, coeffs=[*Gcop])
    Bfitp = line2d(X, Y, coeffs=[*Bcop])

    img_poly = np.dstack([(R-Rfitp+1).reshape(y_dim,x_dim)/2,
                          (G-Gfitp+1).reshape(y_dim,x_dim)/2,
                          (B-Bfitp+1).reshape(y_dim,x_dim)/2])

    img_bl2 = img_poly.astype(np.float32)
    for ii in range(1):
        img_bl2 = cv2.bilateralFilter(img_bl2,1,0.5,1)


    # print('Manually inspect background reduction, then close figures.')
    # plt.figure()
    # plt.imshow(img_bl2)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    img_size = img_bl2.shape
    comp = int(((img_size[0]*img_size[1])**(0.5))/comp_rate)
    if comp == 0:
        comp = 1
    img_proc = img_bl2[::comp,::comp]
    imgc_proc = imgc[::comp,::comp]
    # cv2.imwrite(f"Flakes/bilateralFiltered_{flake_name}.jpg", imgc_proc)
    y_size, x_size, _ = img_proc.shape

    R = img_proc[:,:,0].flatten()
    G = img_proc[:,:,1].flatten()
    B = img_proc[:,:,2].flatten()


    ## Import master catalog values
    in_file_dict = npz2dict(master_cat_file)
    values_dict = {}
    for key in in_file_dict:
        value_name = key[:(np.array([c for c in key])=='-').nonzero()[0][0]] ## c is character in a string, returns everything before the hyphen
        if not value_name in values_dict:
            values_dict[value_name] = []
        values_dict[value_name].append(key)

    master_weights = {}
    master_red_mean = {}
    master_green_mean = {}
    master_blue_mean = {}
    master_cov = {}

    clusters = []
    for tt in range(cluster_count):
        try:
            master_weights[tt] = in_file_dict[values_dict['weights'][tt]]
            master_red_mean[tt] = in_file_dict[values_dict['red mean'][tt]]
            master_green_mean[tt] = in_file_dict[values_dict['green mean'][tt]]
            master_blue_mean[tt] = in_file_dict[values_dict['blue mean'][tt]]
            master_cov[tt] = in_file_dict[values_dict['covariance'][tt]]
            clusters.append(tt)
        except IndexError:
            print(f'Layer {tt} not trained.')

    pixel_count = len(R) ## Number of pixels

    data0 = master_weights.items()
    data1 = master_red_mean.items()
    data2 = master_green_mean.items()
    data3 = master_blue_mean.items()
    data4 = master_cov.items()
    list0 = list(data0)
    list1 = list(data1)
    list2 = list(data2)
    list3 = list(data3)
    list4 = list(data4)
    arr0 = np.array(list0)
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    arr3 = np.array(list3)
    arr4 = np.array(list4, dtype=object)
    master_weights_arr = arr0[:,1] / np.sum(arr0[:,1])
    master_red_mean_arr = arr1[:, 1]
    master_green_mean_arr = arr2[:, 1]
    master_blue_mean_arr = arr3[:, 1]
    master_cov_arr = np.stack(arr4[:,1])
    master_means_arr = np.column_stack([master_blue_mean_arr,master_green_mean_arr,master_red_mean_arr])

    model = GMM(n_components=cluster_count, weights_init=master_weights_arr, means_init=master_means_arr, precisions_init=np.linalg.inv(master_cov_arr))

    # Reshape the image into a dataset for the gmm.
    data = img_proc.reshape(img_proc.shape[0] * img_proc.shape[1], 3)
    res = model.fit_predict(data)
    res = res.reshape(img_proc.shape[0], img_proc.shape[1])

    cluster_prob = model.predict_proba(data)
    ## Calculate probability for each pixel to belong to each thickness.

    ## Give each pixel a home (most probable cluster)
    nearest_cluster = np.empty(pixel_count)
    for nn in range(pixel_count):
        if np.all(np.isnan(cluster_prob[nn])):
            nearest_cluster[nn] = np.nan
        else:
            nearest_cluster[nn] = (cluster_prob[nn]==np.amax(cluster_prob[nn])).nonzero()[0][0]

    layer_image = nearest_cluster.reshape(y_size, x_size)

    toc = time.perf_counter()
    print(f'Time elapsed: {toc-tic} seconds')

    fig, ax = plt.subplots()
    discreteCB(fig, ax, layer_image, flake_name)
    # fig = plt.figure()
    # plt.imshow(imgc_proc)
    # plt.show()

    # cv2.imwrite(f'.\\Flakes\\accuracy_test_{flake_name}.png', layer_image*51)
    cv2.imwrite(f'.\\Flakes\\accuracy_original_{flake_name}.png', cv2.cvtColor(imgc_proc, cv2.COLOR_RGB2BGR))


    print("Images saved.")



# args = {'img_file': ".\\Raw_Images\\Fig_1g.jpg",
#         'flake_name': "KHGR002 3D1",
#         'crop': [0,-1,0,-1],
#         'background': ".\\Raw_Images\\Background.jpg",
#         'masking': [[0,173,531,-1],[280,480,0,140]],
#         'master_cat_file': ".\\Monolayer Search\\Graphene_on_SiO2_master_catalog.npz",
#         'cluster_count':5}

# args = {'img_file': ".\\Raw_Images\\Fig_4i.jpg",
#         'flake_name': "KHGR002 3D1",
#         'crop': [0,2020,1470,2970],
#         'masking': [[0,1260,0,640]],
#         'master_cat_file': ".\\Monolayer Search\\Test2_on_SiO2_master_catalog.npz",
#         'cluster_count':4}

# testing(**args)