import open3d as o3d
import numpy as np
import os
from utilities import *
os.chdir('bunny/bunny/data')





if __name__ == "__main__":
    point_cloud_files = [
    ('bun000.ply', [0, 0, 0], [0, 0, 0, 1]),
    #('bun045.ply', [-0.0520211, -0.000383981, -0.0109223], [0.00548449, -0.294635, -0.0038555, 0.955586]),
    # ('bun090.ply', [2.20761e-05, -3.34606e-05, -7.20881e-05], [0.000335889, -0.708202, 0.000602459, 0.706009]),
    # ('bun180.ply', [0.000116991, 2.47732e-05, -4.6283e-05], [-0.00215148, 0.999996, -0.0015001, 0.000892527]),
    # ('bun270.ply', [0.000130273, 1.58623e-05, 0.000406764], [0.000462632, 0.707006, -0.00333301, 0.7072]),
    #('top2.ply', [-0.0530127, 0.138516, 0.0990356], [0.908911, -0.0569874, 0.154429, 0.383126]),
    ('top3.ply', [-0.0277373, 0.0583887, -0.0796939], [0.0598923, 0.670467, 0.68082, -0.28874]),
    # ('bun315.ply', [-0.00646017, -1.36122e-05, -0.0129064], [0.00449209, 0.38422, -0.00976512, 0.923179]),
    # ('chin.ply', [0.00435102, 0.0882863, -0.108853], [-0.441019, 0.213083, 0.00705734, 0.871807]),
    # ('ear_back.ply', [-0.0829384, 0.0353082, 0.0711536], [0.111743, 0.925689, -0.215443, -0.290169])
    ]

# Load point clouds and apply transformations
    point_clouds = []
    for file, position, quaternion in point_cloud_files:
        pcd = o3d.io.read_point_cloud(file)
        apply_transformation(pcd, position, quaternion)
        point_clouds.append(pcd)
    o3d.visualization.draw_geometries(point_clouds)
    voxel_size = 0.002
    source_down, source_fpfh = preprocess_point_cloud(point_clouds[0], voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(point_clouds[1], voxel_size)
    result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    trans = result.transformation
    point_clouds[0].transform(trans)
     
    for i in range(1, len(point_clouds)):
        reg = icpRegistration(point_clouds[i - 1], point_clouds[i], threshold= 0.0005, iteration=5000)
        print("Fitness:", reg.fitness)
        
    o3d.visualization.draw_geometries(point_clouds)

