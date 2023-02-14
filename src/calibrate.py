import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import open3d
import pickle
import glob
from pprint import pprint
from typing import List

from utils import natural_sort, load_pickle


def pick_2d_points(image_path):
    """
    Opens a GUI to pick 2d coordinates in an image file
    Bug: could not open the OpenCV window on M1 Mac (macOS 13.2)

    Returns:
    img_points: 3-D array
                An array of the image points extracted
    ----
    Parameters
    ----
    image_path: string
                The absolute file path of an image to be picked from
    """
    def click_event(event, x, y, flags, params):
        """
        Callback function for left moust button clicks on the image
        """
        if event == cv.EVENT_LBUTTONDOWN:
            print([x, y])
            cv.drawMarker(img, (x, y), (0, 0, 255), markerType=cv.MARKER_CROSS,
                          markerSize=40, thickness=2, line_type=cv.LINE_AA)
            cv.imshow("image", img)
            img_pts.append([x, y])

    img = cv.imread(image_path, 1)
    img_pts = []

    cv.imshow('image', img)
    cv.setMouseCallback('image', click_event, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_pts


def pick_3d_points(pcd_path):
    """
    Opens a GUI to pick 3d coordinates in a point cloud file

    Returns:
    obj_points: 3-D array
                An array of the object points extracted
    ----
    Parameters
    ----
    pcd_path: string
                The absolute file path of a pcd to be picked from

    """
    coor_frame_geometry = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
    pcd = open3d.io.read_point_cloud(pcd_path)

    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    print(vis.get_picked_points())
    obj_pts = []
    for i in vis.get_picked_points():
        # print(a.points[i])
        point = [1000 * pcd.points[i][1],
                 1000 * pcd.points[i][2],
                 1000 * pcd.points[i][0]]
        print(point)
        obj_pts.append(point)

    return obj_pts


def calibrate_intrisics(im_folder, pcd_folder, result_folder, image_type="png", random_selection=None):
    """
    Makes use of OpenCV's camera calibration functions to calibrate based on picked points

    Returns:
    K:      2-D array
            The calculated camera intrinsic matrix
    d:      array
            The distortion coefficients
    r:      array
            The rotation vectors
    t:      array
            The translation vectors
    ----
    Parameters
    ----
    im_folder:  string
                The absolute folder path of images to be picked from
    pcd_folder: string
                The absolute folder path of pcds to be picked from
    result_folder: string
                The absolute folder path of the result of the calibration
    image_type: string
                The image file extension - png by default
    random_selection: int
                If not None, the function will randomly select the number of images and PCDs to be used for calibration

    """

    # get lists of absolute image and pcd filepaths
    image_fpaths = natural_sort(
        glob.glob(os.path.join(im_folder, f"*.{image_type}")))
    pcd_fpaths = natural_sort(glob.glob(os.path.join(pcd_folder, "*.pcd")))
    assert len(image_fpaths) == len(
        pcd_fpaths), f"Number of images({len(image_fpaths)}) and pcd files({len(pcd_fpaths)}) do not match"

    if random_selection is not None and random_selection < len(image_fpaths):
        # randomly select images and pcd files
        rng = np.random.default_rng()
        indices = sorted(rng.choice(
            len(image_fpaths), size=random_selection, replace=False))
        image_fpaths = [image_fpaths[i] for i in indices]
        pcd_fpaths = [pcd_fpaths[i] for i in indices]

    all_img_pts = []
    for im in image_fpaths:
        img_mat = pick_2d_points(im)
        all_img_pts.append(img_mat)

    all_obj_pts = []
    for pcd in pcd_fpaths:
        obj_mat = pick_3d_points(pcd)
        all_obj_pts.append(obj_mat)

    first_im = cv.imread(image_fpaths[0])
    first_im_gray = cv.cvtColor(first_im, cv.COLOR_BGR2GRAY)

    all_img_pts_m = np.asarray(all_img_pts, dtype=np.float32)
    all_obj_pts_m = np.asarray(all_obj_pts, dtype=np.float32)

    pprint("Image Points:")
    pprint(all_img_pts_m)
    pprint("Object Points:")
    pprint(all_obj_pts_m)

    intrinsic_guess = np.asarray([
        [5000, 0, 1000],
        [0, 5000, 500],
        [0, 0, 1]
    ], dtype=np.float32)

    ret, k, d, r, t = cv.calibrateCamera(
        all_obj_pts_m,
        all_img_pts_m,
        first_im_gray.shape[::-1],
        intrinsic_guess,
        None,
        flags=cv.CALIB_USE_INTRINSIC_GUESS
    )
    print("Calibration matrix")
    print(k)
    print("Distortion Coeffs")
    print(d)
    print("R vecs")
    print(r)
    print("t vecs")
    print(t)

    with open(os.path.join(result_folder, "calib.pickle"), "wb") as f:
        pickle.dump((k, d, r, t, ret), f)

    with open(os.path.join(result_folder, "points.pickle"), "wb") as f:
        pickle.dump({
            "image": all_img_pts_m,
            "object": all_obj_pts_m
        }, f)

    with open(os.path.join(result_folder, "data.pickle"), "wb") as f:
        pickle.dump({
            "img_fpaths": image_fpaths,
            "pcd_fpaths": pcd_fpaths,
        }, f)

    return k, d, r, t


def reproject_world_points(pcd_path, image_path, cam_matrix, d, image_type=".png"):
    """
    Reprojects the 3D points from a given pcd file to the image plane

    Returns:
    ----
    None
    ----
    Parameters:
    ----
    pcd_path:   string
                The absolute path of the folder of pcds to reproject
    image_path: strin
                The absolute path of the folder of images to reproject
    cam_matrix: 3x3 Numpy array
                The calculated intrinsic matrix of the camera
    d:          1x5 Numpy array
                The calculated distortion coefficients of the camera
    """
    images = natural_sort([os.path.join(image_path, im)
                          for im in os.listdir(image_path) if image_type in im])
    print(images)
    pcds = natural_sort([os.path.join(pcd_path, pcd)
                        for pcd in os.listdir(pcd_path) if ".pcd" in pcd])
    print(pcds)

    all_obj_pts = []

    for pcd in pcds:
        obj_mat = pick_3d_points(pcd)
        all_obj_pts.append(obj_mat)

        all_obj_pts_m = np.asarray(all_obj_pts, dtype=np.float32)

        print("Object Points:")
        print(all_obj_pts_m)

        rvec = tvec = (0, 0, 0)

        cam_matrix_m = np.asarray(cam_matrix)
        d_m = np.asarray(d)

        img_pts = cv.projectPoints(
            all_obj_pts_m, rvec, tvec, cam_matrix_m, d_m)
        print(img_pts[0])

        image_fp = images[0]
        im = cv.imread(image_fp, 1)
        plt.imshow(im)
        for pt in img_pts[0]:
            print(pt)
            plt.scatter(pt[0][0], pt[0][1])
        plt.show()

    return (0)


def reproject_points_file(calib_folder: str):
    """
    Calculates reprojection errors from points file
    """
    mat = load_pickle(os.path.join(calib_folder, 'calib.pickle'))
    points = load_pickle(os.path.join(calib_folder, 'points.pickle'))
    img_pts = points["image"]
    obj_pts = points["object"]

    tmp = load_pickle(os.path.join(calib_folder, 'data.pickle'))
    img_fpaths = tmp["img_fpaths"]
    pcd_fpaths = tmp["pcd_fpaths"]

    newmat = mat[0]
    newmat[0][0] = -20000
    newmat[1][1] = -20000
    newmat[0][2] = 1000
    newmat[1][2] = 100

    err_arr = []

    for i, obj in enumerate(obj_pts):
        rvec = tvec = (0, 0, 0)
        img_pts_projected = cv.projectPoints(
            obj, rvec, tvec, newmat, mat[1])[0]
        img_pts_actual = img_pts[i]
        print('Projected: ')
        pprint(img_pts_projected)
        print('Actual: ')
        pprint(img_pts_actual)
        tot = 0
        for j in range(len(img_pts_actual)):
            norm_dist = np.sqrt(abs(
                (img_pts_projected[j][0][0] - img_pts_actual[j][0]) * (
                    img_pts_projected[j][0][1] - img_pts_actual[j][1])))
            pprint(norm_dist)
            tot += norm_dist

        err = tot / (len(img_pts_actual))
        pprint(err)
        err_arr.append(err)

        # visualize
        print('Visualization')
        pprint(img_pts_projected.shape)  # (n_corner, n_data, xy)
        img_pts_projected = np.swapaxes(
            img_pts_projected, 0, 1)    # (n_data, n_corner, xy)

        im = cv.imread(img_fpaths[i], 1)
        plt.imshow(im)
        for pt in img_pts_projected[0]:
            print(pt)
            plt.scatter(pt[0], pt[1])
        plt.show()

    pprint(err_arr)
    pprint(f'mean_error: {np.mean(err_arr)}')
    pprint(f'std_error: {np.std(err_arr)}')


if __name__ == "__main__":
    data_folder = "/Users/ikuta/Documents/data/WildPose/Calibration/ecal_meas/2023-02-04_15-35-57.407_wildpose_v1.1/"
    im_folder = os.path.join(data_folder, "sync_rgb")
    pcd_folder = os.path.join(data_folder, "lidar")
    calib_folder = os.path.join(data_folder, "calib")
    # mat = load_pickle(r"./results/5m/calib.pickle")
    # matrix_calib = [
    #     [ -22000,  0.00000000e+00, 1200],
    #     [ 0.00000000e+00,  -22000, -100],
    #     [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    # ]
    # ds = [ 0.06734681, -0.0479633,  -0.01882605,  0.10849954, -0.00990272]

    # print("matrix")
    # print(mat[0])
    # print("dist")
    # print(mat[1])

    # reproject_world_points(pcd_folder, im_folder, newmat, mat[1], image_type=".jpeg")
    # calibrate_intrisics(im_folder, pcd_folder, calib_folder, image_type="jpeg")
    reproject_points_file(calib_folder)
