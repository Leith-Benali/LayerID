"""Process optical images of thin flakes to distinguish layer thicknesses."""

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from read_npz import npz2dict
import os
import time
import cv2
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import MeanShift, estimate_bandwidth
from background import stack_opened, residual_img
from utils import return_opened

def line2d(x, y, coeffs=[1]*3, return_coeff=False):
    """Returns the result of a 2D quadratic, or returns the coefficients"""
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
    return (((2*np.pi)**3) * np.linalg.det(cov) * np.exp(make_bra(data-mean).dot((np.linalg.inv(cov)).dot(make_ket(data-mean)))))**(-0.5)

def confellipsoid(rad, cent_x, cent_y, cent_z, covariance, res=20, conf=7.815):
    """Creates a confidence ellipsoid based on conf parameter."""

    ## Initial sphere
    u = np.linspace(0, 2*np.pi, res) # Azimuthal angle phi
    v = np.linspace(0, np.pi, res) # Polar angle theta
    x_ = np.outer(np.cos(u), np.sin(v))
    y_ = np.outer(np.sin(u), np.sin(v))
    z_ = np.outer(np.ones(np.size(u)), np.cos(v))
    p = np.stack((x_,y_,z_), axis=2)
    ## Determine confidence matrix transformation
    eigval, eigvec = la.eig(covariance)
    conftrans = np.matmul(np.matmul(eigvec, np.diag(np.sqrt(conf*eigval))), la.inv(eigvec))
    ## Make ellipsoids
    rad_size = np.ones(rad.shape)
    offset = np.array([cent_x, cent_y, cent_z])
    cov_rot = np.tile(conftrans, (res,res,1,1))
    ell_inter = rad_size*np.matmul(cov_rot, p[:,:,:,np.newaxis],
                axes=[(-2,-1),(-2,-1),(-2,-1)]).reshape(res,res,3)
    elli = ell_inter + np.tile(offset, (res,res,1))
    return elli


def training(img_file, flake_name, file_dir, crop, masking, out_file,
             comp_rate=100, density=8, min_clusters=0,
             max_clusters=100, convergence_param=0.0001):
    """
    Clusters pixels of like-color in flake image "img_file" and saves the
    clusters to the catalog "out_file".

    Parameters
    ----------
    img_file : str
        Location of sample image file. (i.e., "...\\RSGR001\\All\\3A1.jpg")
    flake_name : str
        Name of sample image. (i.e., "RSGR001 3A1")
    file_dir : str
        Location of images to use for median background subtraction
    crop : list of ints of form [miny, maxy, minx, maxx]
        Region to crop sample image to. (i.e., [1750,2550, 2100,2850])
    masking : list of list of ints of form [[miny1,maxy1,minx1,maxx1], ...]
        Regions of substrate to fit background. Indices relative to cropped
        image. (i.e., [[200,-1, 0,175], [0,200, 500,-1], [700,-1, 500,-1]])
    out_file : str
        Location of catalog npz file for the same material/substrate as sample.
        (i.e., "...\\Graphene_on_SiO2_catalog.npz")
    comp_rate : int [default: 100]
        Factor in calculating the compression of the normalized image.
        Compression is comp = sqrt(pixels in cropped image)/comp_rate
    density : int [default: 8]
        density^3 gives the number of initial mean points.
    min_clusters : int [default: 0]
        Set minimum number of GMM clusters to fit to.
    max_clusters : int [default: 100]
        Set maximum number of GMM clusters to fit to.
    convergence_param : float [default: 0.0001]
        Parameter to determine how precisely to fit the GMM clusters.
    """

    tic = time.perf_counter()

        ## Import and pre-processing
    ## Image import
    individual_loc = img_file

    # Median filter background subtraction
    all_opened = return_opened(file_dir)
    print(type(all_opened[0]))
    stacked = stack_opened(all_opened)
    opened_img, background, residual, type_ = residual_img(stacked, individual_loc)
    width = int(cv2.imread(img_file).shape[1])
    height = int(cv2.imread(img_file).shape[0])
    dim = (width, height)
    resized = cv2.resize(residual, dim, interpolation=cv2.INTER_AREA)

    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # To disable background subtraction, uncomment the line below, and comment the lines above under # Median filter background subtraction
    # img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    ## Bilateral filtering
    #img_bl = (img).astype(np.float32)/256
    img_bl = (img[crop[0]:crop[1], crop[2]:crop[3]]).astype(np.float32) / 256
    for ii in range(1):
        img_bl = cv2.bilateralFilter(img_bl,2,1,1)

    mask = np.ones(img_bl[:,:,0].shape)
    for reg in masking:
        mask[reg[0]:reg[1], reg[2]:reg[3]] = 0

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

    print('Manually inspect background reduction, then close figures.')
    plt.figure()
    plt.imshow(img_poly)
    plt.figure()
    plt.imshow(mask)
    plt.show() ## Displays for manual inspection for continuing.

    ## Another round of bilateral filtering
    img_bl2 = img_poly.astype(np.float32)
    for ii in range(3):
        img_bl2 = cv2.bilateralFilter(img_bl2,2,0.5,1)

    ## Cropping and compression
    img_size = img_bl2.shape
    comp =  int(((img_size[0]*img_size[1])**(0.5))/comp_rate) ## Compress image by 'comp'
    img_proc = (img_bl2[0::comp, 0::comp])
    color = img_proc.flatten().reshape(-1,3)

    print(f"Image for {flake_name} loaded and pre-processed.")

    image_size = img_proc.shape

    R = img_proc[:,:,0].flatten()
    G = img_proc[:,:,1].flatten()
    B = img_proc[:,:,2].flatten()

    BGRmin = np.amin([B,G,R], axis=1).astype(float)
    BGRmax = np.amax([B,G,R], axis=1).astype(float)


         ## Meanshift
    # Reshape the image into a dataset for the ms.
    data = img_proc.reshape(img_proc.shape[0] * img_proc.shape[1], 3)
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=img_proc.shape[0] * img_proc.shape[1])

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    ms.fit_predict(data)
    cluster_centers = ms.cluster_centers_

    meanB_no = cluster_centers[:,0]
    meanG_no = cluster_centers[:,1]
    meanR_no = cluster_centers[:,2]
    k_mp = len(meanB_no)  ## Number of mean points

    diag = np.sqrt(np.sum((BGRmax - BGRmin)**2))/(density - 1)
    epsilon = diag/3 ## Radius of each point's sphere of posession (OPT)


        ## DBSCAN to consolidate further
    mp_visit = np.full(k_mp, False) ## Tracks whether a mean point has been visited.
    eps2 = epsilon/2 ## Some radius to reach the next point (OPT)
    mp_grouping = np.full(k_mp, -1) ## Keeps track of what group each mp belongs to.
    group_number = 0

    print("DBSCAN algorithm start.")
    cycle = 0
    while not np.all(mp_visit): ## Until all mean points are visited,...
        if not group_number in mp_grouping: ## If group_number has no group members,...
            kk = (mp_visit==False).nonzero()[0][0] ## make the first non-visited mp a member.
            mp_visit[kk] = True
            mp_grouping[kk] = group_number
        for nn in range(k_mp): ## (Could possibly speed up by skipping the mp's that have been assigned to previous groups.)
            if nn != kk:
                if eps2**2 >= (meanB_no[nn]-meanB_no[kk])**2 + (meanG_no[nn]-meanG_no[kk])**2 + (meanR_no[nn]-meanR_no[kk])**2:
                    mp_grouping[nn] = group_number ## Add nn to group_number.
        group_indices = (mp_grouping==group_number).nonzero()[0] ## Indices of group_number members
        if not np.all(mp_visit[group_indices]): ## Check if an mp in group_number hasn't been visited
            kk = group_indices[(mp_visit[group_indices]==False).nonzero()[0][0]] ## First non-visited mp in group_visit gets visited.
            mp_visit[kk] = True
        else:
            group_number += 1
        cycle += 1

    print(f"DBSCAN completed after {cycle} cycles.")
    print(f"Mean Point Grouping: {mp_grouping}")


    ## Once the mean points have been grouped, average the mean points to get
    ## the k points about which GMM will be applied.


    print("Initializing GMM algorithm.")
    ## Initialize means
    GMMB = []
    GMMG = []
    GMMR = []
    for nn in range(group_number+1):
        if nn in mp_grouping:
            GMMB.append(np.mean(meanB_no[(mp_grouping==nn).nonzero()[0]]))
            GMMG.append(np.mean(meanG_no[(mp_grouping==nn).nonzero()[0]]))
            GMMR.append(np.mean(meanR_no[(mp_grouping==nn).nonzero()[0]]))

    cluster_count = len(GMMB) ## Number of data clusters
    pixel_count = len(B) ## Number of pixels

    if cluster_count < min_clusters:
        cluster_count = min_clusters
    elif cluster_count > max_clusters:
        cluster_count = max_clusters

    print(f"Fitting to {cluster_count} clusters")

    def processWithClustering(img: np.ndarray):

        model = GMM(n_components=cluster_count)

        # Reshape the image into a dataset for the gmm.
        data = img.reshape(img.shape[0] * img.shape[1], 3)

        res = model.fit_predict(data)
        res = res.reshape(img.shape[0], img.shape[1])

        cluster_prob = model.predict_proba(data)

        # plt.imshow(img)
        # plt.show()

        weight = model.weights_
        GMMB = model.means_[:,0]
        GMMG = model.means_[:,1]
        GMMR = model.means_[:,2]
        covariance = model.covariances_

        nearest_cluster = np.empty(pixel_count)
        for nn in range(pixel_count):
            nearest_cluster[nn] = (cluster_prob[nn]==np.amax(cluster_prob[nn])).nonzero()[0]

        layer_image = nearest_cluster.reshape(image_size[0], image_size[1])

        ## Make ellipsoids for each GMM cycle of interest
        ellipsoids = {}
        for kk in range(cluster_count):
            ellipsoids[f'{kk}'] = confellipsoid(weight[kk],
                                                GMMB[kk],
                                                GMMG[kk],
                                                GMMR[kk],
                                                covariance[kk])

        cmap = mpl.cm.get_cmap('viridis')
        elli_color = [cmap(ii/cluster_count, alpha=0.25) for ii in range(cluster_count)]


            ## Export and Display
        try:
            in_file_dict = npz2dict(out_file)
        except FileNotFoundError:
            print(f'Created file {out_file}') ## Will be created during np.savez step.
            in_file_dict = {}
            pass
        in_file_dict[flake_name + ' weights'] = weight
        in_file_dict[flake_name + ' blue means'] = GMMB
        in_file_dict[flake_name + ' green means'] = GMMG
        in_file_dict[flake_name + ' red means'] = GMMR
        in_file_dict[flake_name + ' covariance'] = covariance


        toc = time.perf_counter()


        fig1 = plt.figure() ## Scatter plot of data points, colored according to nearest GMM fit.
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(B, G, R, c=nearest_cluster, s=0.5, marker=',')
        ax1.set_xlabel('Blue')
        ax1.set_ylabel('Green')
        ax1.set_zlabel('Red')

        fig2 = plt.figure() ## 95% confidence ellipsoids of GMM fits.
        ax2 = fig2.add_subplot(111, projection='3d')
        for kk in ellipsoids:
            ax2.plot_surface(ellipsoids[kk][:,:,2],
                             ellipsoids[kk][:,:,1],
                             ellipsoids[kk][:,:,0],
                             color=elli_color[int(kk)])
        ax2.set_xlabel('Blue')
        ax2.set_ylabel('Green')
        ax2.set_zlabel('Red')

        fig3, ax3 = plt.subplots()  ## False image of sample, colored based on GMM fit.
        ax3.imshow(layer_image)

        print("{} seconds to run.".format(toc - tic))

        plt.show(block=True)
        thickness = []
        for ii in range(cluster_count):
            thickness.append(int(input(f'How thick is cluster {ii}?: ')))
        in_file_dict[flake_name + ' cluster thicknesses'] = np.array(thickness)

        keep = input('Keep this clustering? (True/False): ')
        if keep=='True':
            try:
                os.remove(out_file)
            except FileNotFoundError:
                pass
            with open(out_file, 'wb') as f:
                np.savez(f, **in_file_dict)
            print(f'Clustering data saved to {out_file}.')
            print('Training complete. Close pyplot figures to finish.')
        else:
            print('Clustering rejected. Check parameters and run script again.')

    processWithClustering(img_proc)

args = {'img_file': ".\\Test_Images\\trainingflake2.jpg",
        'flake_name': "Khang",
        'crop': [0,-1,0,-1],
        'file_dir' : "Graphene_Scan",
        'masking': [[0,50,0,115], [255,-1,240,-1], [280, -1, 0, 70]],
        'out_file': ".\\Monolayer Search\\Graphene_on_SiO2_catalog.npz",
        'min_clusters': 6, 'max_clusters': 12}

training(**args)