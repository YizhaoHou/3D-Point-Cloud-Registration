import open3d as o3d
import numpy as np
import os



def apply_transformation(pcd, position, quaternion):
    """
    Applies translation and rotation transformations to a point cloud object.

    Parameters:
    pcd (open3d.geometry.PointCloud): The point cloud object to transform.
    position (list or np.array): A list or array containing the translation vector [x, y, z].
    quaternion (list or np.array): A list or array containing the quaternion [x, y, z, w] representing the rotation.

    Returns:
    open3d.geometry.PointCloud: The transformed point cloud object

    """
    transformation = np.eye(4)
    transformation[:3, 3] = position
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    transformation[:3, :3] = rotation
    pcd.transform(transformation)
    return pcd


def icpRegistration(source, target, threshold = 1e-5, iteration = 2000):
    """
    apply ICP registration

    Parameters:
    source (base points)
    target (points need to be moved)

    Returns:
    reg_p2p
    """
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))
    return reg_p2p


def GicpRegistration(source, target, threshold = 1e-5, iteration = 2000):
    """
    apply GICP registration
    
    Parameters:
    source (base points)
    target (points need to be moved)

    Returns:
    reg_p2p
    """
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))
    return reg_p2p

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
 
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def remove_outliers_radius(pcd, radius=0.05, min_neighbors=15):
    cl, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    return inlier_cloud, outlier_cloud





