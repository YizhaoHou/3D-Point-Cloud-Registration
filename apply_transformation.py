import open3d as o3d
import numpy as np
import os
os.chdir('bunny/bunny/data')



def apply_transformation(pcd, position, quaternion):
    transformation = np.eye(4)
    transformation[:3, 3] = position
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    transformation[:3, :3] = rotation
    pcd.transform(transformation)
    return pcd

if __name__ == "__main__":
    positions = [
    [0, 0, 0],  # bun000.ply
    [2.20761e-05, -3.34606e-05, -7.20881e-05],  # bun090.ply
    [0.000116991, 2.47732e-05, -4.6283e-05],  # bun180.ply
    [0.000130273, 1.58623e-05, 0.000406764],  # bun270.ply
    ]
    quaternions = [
    [0, 0, 0, 1],  # bun000.ply
    [0.000335889, -0.708202, 0.000602459, 0.706009],  # bun090.ply
    [-0.00215148, 0.999996, -0.0015001, 0.000892527],  # bun180.ply
    [0.000462632, 0.707006, -0.00333301, 0.7072],  # bun270.ply
    ]
    pcd1 = o3d.io.read_point_cloud('bun000.ply')
    # pcd2 = o3d.io.read_point_cloud('bun045.ply')
    pcd3 = o3d.io.read_point_cloud('bun090.ply')
    pcd4 = o3d.io.read_point_cloud('bun180.ply')
    pcd5 = o3d.io.read_point_cloud('bun270.ply')
    point_clouds = [pcd1, pcd3, pcd4, pcd5]
    for pcd, pos, quat in zip(point_clouds, positions, quaternions):
        apply_transformation(pcd, pos, quat)
    o3d.visualization.draw_geometries(point_clouds)

