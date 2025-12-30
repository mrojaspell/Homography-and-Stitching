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

    img3_path = config["img3"]
    img4_path = config["img4"]
    img5_path = config["img5"]
    match_threshold = config["match_threshold"]
    ransac_iterations = config["ransac_iterations"]
    ransac_threshold = config["ransac_threshold"]

    img3 = skimage.io.imread(img3_path)
    img4 = skimage.io.imread(img4_path)
    img5 = skimage.io.imread(img5_path)
    greyimg3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    greyimg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    greyimg5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

    keypts3, des3, keypts4, des4 = sift_descriptors(greyimg3, greyimg4)
    des_dist34 = distance.cdist(des3, des4, 'sqeuclidean')
    keypts4, des4, keypts5, des5 = sift_descriptors(greyimg4, greyimg5)
    des_dist45 = distance.cdist(des4, des5, 'sqeuclidean')

    matches34 = find_matches(des_dist34, keypts3, keypts4, match_threshold)
    matches45 = find_matches(des_dist45, keypts4, keypts5, match_threshold)

    homog34, ransac_matches34, error34, inliers34 = calc_ransac(matches34, ransac_iterations, ransac_threshold)
    homog45, ransac_matches45, error45, inliers45 = calc_ransac(matches45, ransac_iterations, ransac_threshold)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(ax, greyimg3, greyimg4, ransac_matches34)
    plt.show()
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(ax, greyimg4, greyimg5, ransac_matches45)
    plt.show()

    homog35 = homog45.dot(homog34)

    aux_triple_img = warp_and_stitch(img4, img3, homog34)
    triple_img = warp_and_stitch(img5, aux_triple_img, homog35)

    plt.figure(figsize=(12, 8))
    plt.imshow(triple_img)
    plt.show()