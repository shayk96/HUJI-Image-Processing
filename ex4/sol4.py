""""---------------------imports------------------------------------------------------------------------------------"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, map_coordinates, label, center_of_mass
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
import shutil
from imageio import imwrite

import sol4_utils

""""---------------------constants-----------------------------------------------------------------------------------"""
DERIVATIVE_X_FILTER = np.array([[1, 0, -1]])
DERIVATIVE_Y_FILTER = DERIVATIVE_X_FILTER.T
KERNEL_SIZE = 3
K = 0.04
PYRAMID_LEVEL_DIFF = 4
DESCRIPTOR_RADIUS = 3
BASE_IMAGE = 0
VERTICAL_RECTANGLE = 7
HORIZONTAL_RECTANGLE = 7
MIN_DIS_FROM_EDGE = 5
LVL_TO_EXTRACT_DESC = 2
MODE_CONSTANT = 'constant'
MODE_REFLECT = "reflect"

""""---------------------required functions--------------------------------------------------------------------------"""


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.  
  """
    Ix = convolve(im, DERIVATIVE_X_FILTER, mode=MODE_REFLECT)
    Iy = convolve(im, DERIVATIVE_Y_FILTER, mode=MODE_REFLECT)
    Ix2 = sol4_utils.blur_spatial(Ix * Ix, KERNEL_SIZE)
    Iy2 = sol4_utils.blur_spatial(Iy * Iy, KERNEL_SIZE)
    IxIy = sol4_utils.blur_spatial(Ix * Iy, KERNEL_SIZE)
    det = Ix2 * Iy2 - IxIy ** 2
    trace = Ix2 + Iy2
    yx = np.asarray(non_maximum_suppression(det - K * (trace ** 2))).nonzero()
    interest_points = np.fliplr(np.transpose(yx))
    return interest_points


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image. (3rd level in gaussian pyramid)
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.   
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    get_patch = lambda pos: np.meshgrid(np.arange(pos[0] - desc_rad, pos[0] + desc_rad + 1),
                                        np.arange(pos[1] - desc_rad, pos[1] + desc_rad + 1))
    y, x = np.moveaxis(np.apply_along_axis(get_patch, arr=pos, axis=1), 0, 1)  # input to the map coordinates function
    descriptors = map_coordinates(im, [x, y], order=1, prefilter=False, mode=MODE_CONSTANT)
    vec_shape = (descriptors.shape[0], descriptors.shape[1] * descriptors.shape[2])
    normalize = lambda desc: np.zeros(desc.shape) if np.linalg.norm(desc - np.mean(desc)) == 0 else \
        (desc - np.mean(desc)) / np.linalg.norm(desc - np.mean(desc))
    descriptors_normalized = np.apply_along_axis(normalize, arr=descriptors.reshape(vec_shape), axis=1)
    return descriptors_normalized.reshape(descriptors.shape)


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image. 
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    interest_points = spread_out_corners(pyr[BASE_IMAGE], VERTICAL_RECTANGLE, HORIZONTAL_RECTANGLE, MIN_DIS_FROM_EDGE)
    descriptors = sample_descriptor(pyr[LVL_TO_EXTRACT_DESC], interest_points / PYRAMID_LEVEL_DIFF, DESCRIPTOR_RADIUS)
    return [interest_points, descriptors]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    vec_shape = desc1.shape[1] * desc1.shape[2]
    scores = np.dot(desc1.reshape(desc1.shape[0], vec_shape), desc2.reshape(desc2.shape[0], vec_shape).T)
    row_max = np.partition(scores, -2)[:, -2].reshape(desc1.shape[0], 1)
    col_max = np.partition(scores.T, -2)[:, -2].reshape(desc2.shape[0], 1)
    matched = np.nonzero((scores > min_score) & (scores >= row_max) & (scores.T >= col_max).T)
    return matched


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    homogenous = np.insert(pos1, pos1.shape[1], 1, axis=1)
    translated_homogenous = (H12 @ homogenous.T).T
    not_zero = translated_homogenous[:, 2] != 0
    translated_pos = (translated_homogenous[not_zero] / translated_homogenous[:, 2][not_zero, np.newaxis])[:, :2]
    return translated_pos


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    inliers = np.array([])
    for i in range(num_iter):
        points = np.random.choice(points1.shape[0], 1 if translation_only else 2)
        rigid_matrix = estimate_rigid_transform(points1[points], points2[points], translation_only)
        points_transformed = apply_homography(points1, rigid_matrix)
        ok_inliers = np.nonzero((np.linalg.norm(points2 - points_transformed, axis=1) ** 2) < inlier_tol)
        inliers = ok_inliers[0] if ok_inliers[0].size > inliers.size else inliers
    rigid_matrix = estimate_rigid_transform(points1[inliers], points2[inliers], translation_only)
    return [rigid_matrix, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    images = np.hstack([im1, im2])
    plt.imshow(images, cmap="gray")
    points2_moved = np.copy(points2)
    points2_moved[:, 0] += im1.shape[1]
    outliers = np.delete(np.arange(points1.shape[0]), inliers)
    outliers1, outliers2 = points1[outliers], points2_moved[outliers]
    inliers1, inliers2 = points1[inliers], points2_moved[inliers]
    plt.plot((outliers1[:, 0], outliers2[:, 0]), (outliers1[:, 1], outliers2[:, 1]), mfc='r', c='b', lw=.4, ms=2,
             marker='.', mec='r')
    plt.plot((inliers1[:, 0], inliers2[:, 0]), (inliers1[:, 1], inliers2[:, 1]), mfc='r', c='y', lw=.4, ms=1,
             marker='.', mec='r')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    homographs = [np.empty((3, 3))] * (len(H_succesive) + 1)
    homographs[m] = np.eye(3)
    for i in range(m - 1, -1, -1):
        homographs[i] = (H_succesive[i] @ homographs[i + 1]) / homographs[i][2, 2]
    for j in range(m + 1, len(H_succesive) + 1):
        homographs[j] = homographs[j - 1] @ np.linalg.inv(H_succesive[j - 1]) / homographs[j][2, 2]
    return homographs


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    corners = apply_homography(np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]), homography)
    min_x, min_y = np.min(corners[:, 0]), np.min(corners[:, 1])
    max_x, max_y = np.max(corners[:, 0]), np.max(corners[:, 1])
    bounding_box = np.array([([int(min_x.round()), int(min_y.round())]), [int(max_x.round()), int(max_y.round())]])
    return bounding_box


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    homography_inverse = np.linalg.inv(homography)
    min_xy, max_xy = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_grid, y_grid = np.arange(min_xy[0], max_xy[0]), np.arange(min_xy[1], max_xy[1])
    panorama_x, panorama_y = np.meshgrid(x_grid, y_grid)
    coordinates = np.array([panorama_x.flatten(), (panorama_y.flatten())]).T
    original = apply_homography(coordinates, homography_inverse)
    warped = map_coordinates(image, [original[:, 1], original[:, 0]], order=1, prefilter=False, mode=MODE_CONSTANT)
    warped = warped.reshape(panorama_x.shape)
    return warped


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()



