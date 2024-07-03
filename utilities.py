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
