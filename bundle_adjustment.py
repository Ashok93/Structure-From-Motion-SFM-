import numpy as np
from scipy.optimize import least_squares

def reprojection_loss_function(opt_variables, points_2d, num_pts):
    '''
    opt_variables --->  Camera Projection matrix + All 3D points
    '''
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]
        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])

    return np.array(rep_error).ravel()

def bundle_adjustment(points_3d, points_2d, img, projection_matrix):

    opt_variables = np.hstack((projection_matrix.ravel(), points_3d.ravel(order="F")))
    num_points = len(points_2d[0])

    corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(points_2d,num_points))

    print("The optimized values \n" + str(corrected_values))
    P = corrected_values.x[0:12].reshape(3,4)
    points_3d = corrected_values.x[12:].reshape((num_points, 4))

    return P, points_3d