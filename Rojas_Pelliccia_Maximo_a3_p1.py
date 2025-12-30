# imports
import skimage
import cv2
from scipy.spatial import distance
import random as rd
import json
from pylab import *
from scipy import signal
from scipy import *
from PIL import Image

##############################################
### Provided code - nothing to change here ###
##############################################

"""
Harris Corner Detector
Usage: Call the function harris(filename) for corner detection
Reference   (Code adapted from):
             http://www.kaij.org/blog/?p=89
             Kai Jiang - Harris Corner Detector in Python

"""



def harris(filename, min_distance=10, threshold=0.1):
    """
    filename: Path of image file
    threshold: (optional)Threshold for corner detection
    min_distance : (optional)Minimum number of pixels separating
     corners and image boundary
    """
    im = np.array(Image.open(filename).convert("L"))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim, min_distance, threshold)
    plot_harris_points(im, filtered_coords)


def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size + 1, -sizey:sizey + 1]
    # x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x ** 2 / float((0.5 * size) ** 2) + y ** 2 / float((0.5 * sizey) ** 2)))
    gy = - y * exp(-(x ** 2 / float((0.5 * size) ** 2) + y ** 2 / float((0.5 * sizey) ** 2)))
    return gx, gy


def gauss_kernel(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size + 1, -sizey:sizey + 1]
    g = exp(-(x ** 2 / float(size) + y ** 2 / float(sizey)))
    return g / g.sum()


def compute_harris_response(im):
    """ compute the Harris corner detector response function
        for each pixel in the image"""
    # derivatives
    gx, gy = gauss_derivative_kernels(3)
    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')
    # kernel for blurring
    gauss = gauss_kernel(3)
    # compute components of the structure tensor
    Wxx = signal.convolve(imx * imx, gauss, mode='same')
    Wxy = signal.convolve(imx * imy, gauss, mode='same')
    Wyy = signal.convolve(imy * imy, gauss, mode='same')
    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    return Wdet / Wtr


def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating
        corners and image boundary"""
    # find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [(candidates[0][c], candidates[1][c]) for c in range(len(candidates[0]))]
    # ...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]
    # sort candidates
    index = argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance, min_distance:-min_distance] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0] - min_distance):(coords[i][0] + min_distance),
            (coords[i][1] - min_distance):(coords[i][1] + min_distance)] = 0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'r*')
    axis('off')
    show()


# Usage:
# harris('./path/to/image.jpg')


# Provided code for plotting inlier matches between two images

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')

    ax.plot(inliers[:, 0], inliers[:, 1], '+r')
    ax.plot(inliers[:, 2] + img1.shape[1], inliers[:, 3], '+r')
    ax.plot([inliers[:, 0], inliers[:, 2] + img1.shape[1]],
            [inliers[:, 1], inliers[:, 3]], 'r', linewidth=0.4)
    ax.axis('off')

# Usage:
# fig, ax = plt.subplots(figsize=(20,10))
# plot_inlier_matches(ax, img1, img2, computed_inliers)


#######################################
### Your implementation starts here ###
#######################################

def sift_descriptors(greyimg1, greyimg2):
    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(greyimg1, None)
    kpts2, desc2 = sift.detectAndCompute(greyimg2, None)

    return kpts1, desc1, kpts2, desc2

def find_matches(dist, kp1, kp2, val):
    to_ret = []
    h, w = dist.shape
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    for i in range(h):
        for j in range(w):
            if dist[i][j] <= val:
                aux = list(kp1[i].pt + kp2[j].pt)
                to_ret.append(aux)
    to_ret = np.array(to_ret)
    return to_ret

def calc_homography(points):
    A = []
    for i in range(4):
        p1 = np.append(points[i][:2], 1)
        p2 = np.append(points[i][2:], 1)
        A.append([0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]])
        A.append([p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H / H[2, 2]

def calc_ransac(points, iters, threshold):
    inlier_count = 0
    best_homog = None
    best_inliers = None

    for i in range(iters):
        sample = rd.sample(range(points.shape[0]), 4)
        homog = calc_homography(points[sample])
        if np.linalg.matrix_rank(homog) < 3:
            continue
        errors = np.zeros(points.shape[0])
        for j in range(points.shape[0]):
            p1 = np.append(points[j][:2], 1)
            p2 = np.append(points[j][2:], 1)
            aux = np.dot(homog, p1) / np.dot(homog, p1)[-1]
            errors[j] = np.linalg.norm(p2[:2] - aux[:2])
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > inlier_count:
            inlier_count = len(inliers)
            best_homog = homog
            best_inliers = points[inliers]

    homog = np.zeros((2*inlier_count, 9))
    for i in range(inlier_count):
        p1 = np.append(best_inliers[i][:2], 1)
        p2 = np.append(best_inliers[i][2:], 1)
        homog[2*i] = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
        homog[2*i + 1] = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
    U, S, V = np.linalg.svd(homog)
    best_homog = V[-1, :].reshape((3, 3))

    inlier_residual = 0
    for i in range(inlier_count):
        p1 = np.append(best_inliers[i][:2], 1)
        p2 = np.append(best_inliers[i][2:], 1)
        aux = np.dot(best_homog, p1) / np.dot(best_homog, p1)[-1]
        inlier_residual += np.linalg.norm(p2[:2] - aux[:2])
    inlier_residual /= inlier_count
    print("Average inlier residual: ", inlier_residual)
    print("Number of inliers: ", inlier_count)

    return best_homog, best_inliers, errors, inlier_count

def warp_and_stitch(grey1, grey2, homog):
    h1, w1, _ = grey1.shape
    homog_transform = skimage.transform.ProjectiveTransform(homog)
    mat = np.array([[0,0], [0, h1], [w1, h1], [w1, 0]])
    new_homog = homog_transform(mat)

    space = np.vstack((mat, new_homog))
    min_t = np.int32(space.min(axis=0))
    max_t = np.int32(space.max(axis=0))
    x_offset = -min_t[0]
    y_offset = -min_t[1]
    offset_mat = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])

    img_transform = cv2.warpPerspective(grey2, offset_mat.dot(homog), tuple(max_t - min_t))
    ht, wt, _ = img_transform.shape
    for i in range(h1):
        for j in range(w1):
            if x_offset + j < wt and y_offset + i < ht:
                if np.count_nonzero(img_transform[y_offset + i, x_offset + j]) == 0:
                    img_transform[y_offset + i, x_offset + j, :] = grey1[i, j, :]
                else:
                    img_transform[y_offset + i, x_offset + j, :] = img_transform[y_offset + i, x_offset + j, :] /2 + grey1[i, j, :] /2

    return img_transform


if __name__ == "__main__":
    with open("Rojas_Pelliccia_Maximo_a3_config.json", "r") as f:
        config = json.load(f)

    img1_path = config["img1"]
    img2_path = config["img2"]
    match_threshold = config["match_threshold"]
    ransac_iterations = config["ransac_iterations"]
    ransac_threshold = config["ransac_threshold"]

    img1 = skimage.io.imread(img1_path)
    img2 = skimage.io.imread(img2_path)
    greyimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    greyimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    keypts1, des1, keypts2, des2 = sift_descriptors(greyimg1, greyimg2)

    des_dist = distance.cdist(des1, des2, 'sqeuclidean')

    matches = find_matches(des_dist, keypts1, keypts2, match_threshold)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(ax, img1, img2, matches)
    plt.show()

    homography, ransac_matches, error, inliers = calc_ransac(matches, ransac_iterations, ransac_threshold)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(ax, greyimg1, greyimg2, ransac_matches)
    plt.show()

    new_img = warp_and_stitch(img2, img1, homography)

    plt.figure(figsize=(12, 8))
    plt.imshow(new_img)
    plt.show()