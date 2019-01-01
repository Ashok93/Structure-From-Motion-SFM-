import cv2 as cv
import os
import numpy as np

from bundle_adjustment import bundle_adjustment
from plot_utils import viz_3d, viz_3d_matplotlib, draw_epipolar_lines

######################### Path Variables ##################################################
curr_dir_path = os.getcwd()
images_dir = curr_dir_path + '/data/images/observatory'
calibration_file_dir = curr_dir_path + '/data/calibration'
###########################################################################################

def get_camera_intrinsic_params():
    K = []
    with open(calibration_file_dir + '/cameras.txt') as f:
        lines = f.readlines()
        calib_info = [float(val) for val in lines[0].split(' ')]
        row1 = [calib_info[0], calib_info[1], calib_info[2]]
        row2 = [calib_info[3], calib_info[4], calib_info[5]]
        row3 = [calib_info[6], calib_info[7], calib_info[8]]

        K.append(row1)
        K.append(row2)
        K.append(row3)
    
    return K

def get_pinhole_intrinsic_params():
    K = []
    with open(calibration_file_dir + '/camera_observatory.txt') as f:
        lines = f.readlines()
        calib_info = [float(val) for val in lines[0].split(' ')]
        row1 = [calib_info[0], 0, calib_info[2]]
        row2 = [0, calib_info[1], calib_info[3]]
        row3 = [0, 0, 1]

        K.append(row1)
        K.append(row2)
        K.append(row3)
    
    return K

def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


if __name__ == "__main__":
    # Variables 
    iter = 0
    prev_img = None
    prev_kp = None
    prev_desc = None
    K = np.array(get_pinhole_intrinsic_params(), dtype=np.float)
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))
    pts_4d = []
    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    for filename in os.listdir(images_dir)[0:3]:
        
        file = os.path.join(images_dir, filename)
        img = cv.imread(file, 0)

        resized_img = img
        sift = cv.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(resized_img,None)
        
        if iter == 0:
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=100)
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(prev_desc,desc,k=2)
            good = []
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    good.append(m)
                    pts1.append(prev_kp[m.queryIdx].pt)
                    pts2.append(kp[m.trainIdx].pt)
                    
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)
            print("The fundamental matrix \n" + str(F))

            # We select only inlier points
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]

            #draw_epipolar_lines(pts1, pts2, prev_img, resized_img)

            E = np.matmul(np.matmul(np.transpose(K), F), K)

            print("The new essential matrix is \n" + str(E))

            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
            
            print("I+0 \n" + str(R_t_0))

            print("Mullllllllllllll \n" + str(np.matmul(R, R_t_0[:3,:3])))

            R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())

            print("The R_t_0 \n" + str(R_t_0))
            print("The R_t_1 \n" + str(R_t_1))

            P2 = np.matmul(K, R_t_1)

            print("The projection matrix 1 \n" + str(P1))
            print("The projection matrix 2 \n" + str(P2))

            pts1 = np.transpose(pts1)
            pts2 = np.transpose(pts2)

            print("Shape pts 1\n" + str(pts1.shape))

            points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
            points_3d /= points_3d[3]

            # P2, points_3D = bundle_adjustment(points_3d, pts2, resized_img, P2)
            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2[0])
            rep_error_fn(opt_variables, pts2, num_points)

            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))

            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc

        iter = iter + 1

    pts_4d.append(X)
    pts_4d.append(Y)
    pts_4d.append(Z)

    viz_3d(np.array(pts_4d))