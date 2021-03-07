import random
import numpy as np
import pandas as pd

from quaternion import as_rotation_matrix, as_euler_angles
import cv2
import torch
import torchvision
import scipy

import math
#import habitat
import matplotlib.pyplot as plt

def clever_resize(source, target, num_channel):
    assert source.shape[-1] == 512, "source should be 512"
    assert target == 128, "target size not supported"
    # in navigation
    if len(list(source.shape)) == 4:
        assert source.shape[0] == 1, "only support processing batch size 1"
        assert source.shape
        result = torch.zeros(source.shape[0], source.shape[1], target, target)
        for i in range(target):
            for j in range(target):
                area = source[0, :num_channel, i : i+4, j : j + 4]
                area = torch.argmax(area, dim=0)
                if torch.all(area == num_channel - 1):
                    result[0, i]
        
    else:    
        return
        
        

class ScalarMeanTracker(object):
        def __init__(self) -> None:
            self._sums = {}
            self._counts = {}
        def add_scalars(self, scalars):
            for k in scalars:
                if k not in self._sums:
                    self._sums[k] = scalars[k]
                    self._counts[k] = 1
                else:
                    self._sums[k] += scalars[k]
                    self._counts[k] += 1
        def pop_and_reset(self):
            means = {k: self._sums[k] / self._counts[k] for k in self._sums}
            self._sums = {}
            self._counts = {}
                         
            return means


def get_points(env, observations, roof_thre, floor_thre, ignore, num_channel,\
        id2cat, fov, semantics=None):

    # get transformation matrix of current state
    quaternion = env._sim.get_agent_state().sensor_states['depth'].rotation
    translation = env._sim.get_agent_state().sensor_states['depth'].position
    # Transformation matrix: camera sees -> location in world
    rotation = as_rotation_matrix(quaternion)
    T_world = np.eye(4)
    T_world[0:3, 0:3] = rotation
    T_world[0:3, 3] = translation
    cx, center, cz = translation[0], translation[1], translation[2]
    # get point_cloud(world) and corresponding rgb & semantics
    points = generate_pc(T_world, observations['depth'], fov)
    rgbs = color2local3d(observations['rgb'])
    depths = observations['depth'].flatten()

    if semantics is None:
        semantics = observations['semantic'].flatten()
    else:
        semantics = semantics.flatten()
    # remove ceiling & misc

    semantics = np.vectorize(lambda x: id2cat.get(x, num_channel))(semantics)
    valid = None
    for ctg in ignore:
        if valid is None:
            valid = (semantics != ctg)
        else:
            valid = valid & (semantics != ctg)

    points = points[valid, :]
    rgbs = rgbs[valid, :]
    semantics = semantics[valid] - 1
    depths = depths[valid]
    # remove roof & lower floor
    #roof_thre = center + offset * 2
    #floor_thre = center - height
    no_roof = (points[:, 1] < roof_thre) & (points[:, 1] >floor_thre)
    points =  points[no_roof, :]
    rgbs = rgbs[no_roof, :]
    semantics = semantics[no_roof]
    depths = depths[no_roof]

    return points, rgbs, depths, semantics
def get_height_map(env, pointss, rgbss, semanticss, depthss,\
        area_x, area_z, h, w):
    # map global point according to agent's rotation and position
    quaternion = env._sim.get_agent_state().rotation
    translation = env._sim.get_agent_state().position
    # Transformation matrix: camera sees -> location in world
    rotation = as_rotation_matrix(quaternion)
    T_world = np.eye(4)
    T_world[0:3, 0:3] = rotation
    T_world[0:3, 3] = translation
    pointss = pc2local(T_world, pointss)

    round_agent = (np.abs(pointss[:, 0]) < area_x / 2.)\
    & (np.abs(pointss[:, 2]) < area_z / 2.)

    pointss = pointss[round_agent, :]
    rgbss = rgbss[round_agent, :]
    semanticss = semanticss[round_agent]
    depthss = depthss[round_agent]

    # generate top-down height map
    scale_h = area_z / h
    scale_w = area_x / w


    X = (pointss[:, 0] + (area_x / 2.)) / float(scale_w)
    Y = pointss[:, 1]
    Z = (pointss[:, 2] + (area_z / 2.)) / float(scale_h)

    # 1 for 4
    XZ_ff = np.column_stack((np.floor(X), np.floor(Z))) #0, 1, ... n-1
    XZ_fc = np.column_stack((np.floor(X), np.ceil(Z)))
    XZ_cc = np.column_stack((np.ceil(X), np.ceil(Z)))
    XZ_cf = np.column_stack((np.ceil(X), np.floor(Z)))
    XZ = np.concatenate((np.concatenate((XZ_ff, XZ_fc), axis=0),\
            np.concatenate((XZ_cc, XZ_cf), axis=0)), axis=0) #0, 1, ... 4n - 1
    assert h == w, "Currently only support square!"
    XZ[XZ >= h] = h - 1
    XZ[XZ < 0.] = 0.
    Y = repeat4(Y)

    XYZ = np.column_stack((XZ, Y))
    df = pd.DataFrame(XYZ)
    idx = df.groupby([0, 1])[2].transform(max) == df[2]

    idx = idx.values

    height_rgb = np.zeros((h, w, 3))
    height_sem = -np.ones((h, w), dtype=np.int32)
    height_dep = np.zeros((h, w))

    rgbss_4 = repeat4(rgbss)
    semanticss_4 = repeat4(semanticss)
    depthss_4 = repeat4(depthss)

    XZ = XZ.astype(int)
    height_rgb[XZ[idx][:, 0], XZ[idx][:, 1], :] = rgbss_4[idx, :]
    height_sem[XZ[idx][:, 0], XZ[idx][:, 1]] = semanticss_4[idx]
    height_dep[XZ[idx][:, 0], XZ[idx][:, 1]] = depthss_4[idx]

    height_rgb = np.swapaxes(height_rgb, 0, 1)[:, :, [2, 1, 0]] # rgb to bgr
    height_sem = np.swapaxes(height_sem, 0, 1)
    height_dep = np.swapaxes(height_dep, 0, 1)

    return height_rgb, height_sem, height_dep
def get_panorama(env, pointss, rgbss, depthss, semanticss, roof_thre,\
        floor_thre, fov, ignore, \
        num_channel, id2cat):
    # suppose the agent is not looking down
    
    for k in range(3):
        try:
            env.step("LOOK_DOWN")
        except:
            return None
    # right+bottom, back+bottom, left+bottom, front+bottom 
    for m in range(4):
        try:
            observations = env.step("TURN_RIGHT")
        except:
            return None
        points, rgbs, depths, semantics = get_points(env, observations,\
                roof_thre, floor_thre,\
            ignore, num_channel, id2cat,fov)
        if pointss is None:
            pointss, rgbss, depthss, semanticss = points, rgbs, depths,\
                    semantics
        else:
            pointss = np.concatenate((pointss, points), axis=0)
            rgbss = np.concatenate((rgbss, rgbs), axis=0)
            semanticss = np.concatenate((semanticss, semantics), axis=0)
            depthss = np.concatenate((depthss, depths), axis=0)
    for k in range(2):
        env.step("LOOK_UP")
    # right, back, left, front
    for m in range(4):
        observations = env.step("TURN_RIGHT")
        points, rgbs, depths, semantics = get_points(env, observations,\
                roof_thre, floor_thre,\
            ignore, num_channel, id2cat, fov)
        pointss = np.concatenate((pointss, points), axis=0)
        rgbss = np.concatenate((rgbss, rgbs), axis=0)
        semanticss = np.concatenate((semanticss, semantics), axis=0)
        depthss = np.concatenate((depthss, depths), axis=0)


    return pointss, rgbss, semanticss, depthss

def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def depth2local3d_gpu(device, dep, FOV):

    depth = dep[..., 0]
    h, w = depth.shape
    f = float(0.5 / np.tan(FOV / 2.) * float(w))
    x = torch.linspace(0, w - 1, w)
    y = torch.linspace(0, h - 1, h)
    xv, yv = torch.meshgrid(x, y)
    xv, yv = xv.t().to(device), yv.t().to(device)
    dfl = depth.reshape(-1)

    result = torch.cat((\
            (dfl * (xv.reshape(-1) - w / 2.) / f).unsqueeze(-1),\
            - (dfl * (yv.reshape(-1) - h / 2.) / f).unsqueeze(-1),\
            - dfl.unsqueeze(-1)), dim=1)
    del depth
    del x
    del y
    del xv
    del yv
    del dfl

    return result

def generate_pc_gpu(device, T_world, depth, fov):
    dep = torch.from_numpy(depth).to(device)
    points = depth2local3d_gpu(device, dep, fov)
    cam = torch.cat((points.float(), torch.ones(points.shape[0], 1).float().to(device)), dim=1)
    result = torch.mm(T_world, cam.permute(1, 0)).permute(1, 0)[:, 0:3]
    del dep
    del points
    del cam
    return result

def pc2local_gpu(device, T_world, pointss):
    pointss = torch.from_numpy(pointss).to(device)
    T_world = T_world.to(device)
    cam = torch.cat((pointss.float(), torch.ones(pointss.shape[0],
        1).to(device).float()),
            dim=1)
    cam = torch.mm(cam, torch.inverse(T_world.t()).float())

    pointss, T_world, cam = pointss.cpu(), T_world.cpu(), cam.cpu()

    return cam[:, 0:3]

def color2local3d_gpu(device, color):

    color = torch.from_numpy(color).to(device)
    result = torch.cat((\
            color[..., 0].reshape(-1).unsqueeze(-1),\
            color[..., 1].reshape(-1).unsqueeze(-1),\
            color[..., 2].reshape(-1).unsqueeze(-1)),\
            dim = 1)
    del color
    return result


def get_T_gpu(device, env):
    # get transformation matrix of current state
    quaternion = env._sim.get_agent_state().sensor_states['depth'].rotation
    translation = env._sim.get_agent_state().sensor_states['depth'].position
    # Transformation matrix: camera sees -> location in world
    rotation = as_rotation_matrix(quaternion)
    T_world = torch.eye(4).to(device)
    T_world[0:3, 0:3] = torch.from_numpy(rotation).to(device)
    T_world[0:3, 3] = torch.from_numpy(translation).to(device)
    return T_world

def repeat4_gpu(Y):
    try:
        _ = Y.shape[1]
        return Y.repeat(4, 1)
    except:
        return Y.repeat(4)

def get_observations_gpu(device, env, observations, pointss, rgbss, semanticss, depthss, height, offset, area_x, area_z, h, w, fov):


    T_world = get_T_gpu(device, env)

    # get center of agent of current state
    center = env._sim.get_agent_state().position[1]

    # get point_cloud(world) and corresponding rgb & semantics
    points = generate_pc_gpu(device, T_world, observations['depth'], fov)
    rgbs = color2local3d_gpu(device, observations['rgb'])
    semantics = torch.from_numpy(observations['semantic'].astype(np.int64)).reshape(-1).to(device)
    depths = torch.from_numpy(observations['depth']).reshape(-1).to(device)
    # remove roof

    roof_thre = center + height / 2. + offset
    no_roof = points[:, 1] < roof_thre
    points =  points[no_roof, :]
    rgbs = rgbs[no_roof, :]
    semantics = semantics[no_roof]
    depths = depths[no_roof]


    # accumulate step
    # pointss: whole points viewed (world)
    # rgbss, semanticss: corresponding rgb, semantic coloring info
    if pointss is None:
        pointss = points
        rgbss = rgbs
        semanticss = semantics
        depthss = depths
    else:
        pointss = torch.from_numpy(pointss).to(device)
        rgbss = torch.from_numpy(rgbss).to(device)
        semanticss = torch.from_numpy(semanticss).to(device)
        depthss = torch.from_numpy(depthss).to(device)
        pointss = torch.cat((pointss, points), dim=0)
        rgbss = torch.cat((rgbss, rgbs), dim=0)
        semanticss = torch.cat((semanticss, semantics), dim=0)
        depthss = torch.cat((depthss, depths), dim=0)
    #Store global values
    c_pointss = pointss.clone()
    c_rgbss = rgbss.clone()
    c_semanticss = semanticss.clone()
    c_depthss = depthss.clone()

    env.step("LOOK_UP")
    T_world = get_T_gpu(device, env)
    pointss = pc2local_gpu(device, T_world, pointss)
    env.step("LOOK_DOWN")

    # clip point clouds = area around agent
    round_agent = (torch.abs(pointss[:, 0]) < area_x / 2.) & (torch.abs(pointss[:, 2]) < area_z / 2.)
    pointss = pointss[round_agent, :]
    rgbss = rgbss[round_agent, :]
    semanticss = semanticss[round_agent]
    depthss = depthss[round_agent]
    # generate top-down height map
    scale_h = area_z / h
    scale_w = area_x / w

    X = (pointss[:, 0] + area_x / 2.) / float(scale_w)
    Y = pointss[:, 1]
    Z = (pointss[:, 2] + area_z / 2.) / float(scale_h)

    # 1 for 4
    X = X.unsqueeze(-1)
    Z = Z.unsqueeze(-1)
    XZ_ff = torch.cat((torch.floor(X), torch.floor(Z)), dim=-1)
    XZ_fc = torch.cat((torch.floor(X), torch.ceil(Z)), dim=-1)
    XZ_cc = torch.cat((torch.ceil(X), torch.ceil(Z)), dim=-1)
    XZ_cf = torch.cat((torch.ceil(X), torch.floor(Z)), dim=-1)
    XZ = torch.cat((torch.cat((XZ_ff, XZ_fc), dim=0),\
            torch.cat((XZ_cc, XZ_cf), dim=0)), dim=0) #0, 1, ... 4n - 1
    assert h == w, "Currently only support square!"
    XZ[XZ >= h] = h - 1
    XZ[XZ < 0.] = 0.
    Y = repeat4_gpu(Y)
   # print(Y.shape)

    XYZ = torch.zeros(Y.shape[0], 3).to(device)
    XYZ[:, 2] = Y
    XYZ[:, 0:2] = XZ

    df = pd.DataFrame(XYZ.cpu().numpy())
    idx = df.groupby([0, 1])[2].transform(max) == df[2]
    idx = torch.from_numpy(idx.values).bool().to(device)

    height_rgb = torch.zeros(h, w, 3).byte().to(device)
    height_sem = -torch.ones(h, w).long().to(device)
    height_dep = torch.zeros(h, w).float().to(device)

    rgbss_4 = repeat4_gpu(rgbss)
    semanticss_4 = repeat4_gpu(semanticss)
    depthss_4 = repeat4_gpu(depthss)

    XZ = XZ.long()
    height_rgb[XZ[idx][:, 0], XZ[idx][:, 1], :] = rgbss_4[idx, :]
    height_sem[XZ[idx][:, 0], XZ[idx][:, 1]] = semanticss_4[idx]
    height_dep[XZ[idx][:, 0], XZ[idx][:, 1]] = depthss_4[idx]


    height_rgb = height_rgb.permute(1, 0, 2).detach().cpu().numpy()[:, :, [2, 1, 0]]
    height_sem = height_sem.permute(1, 0).detach().cpu().numpy()
    height_dep = height_dep.permute(1, 0).detach().cpu().numpy()

    rgb = observations['rgb']# rgb
    sem = observations['semantic']
    sem = sem.reshape(sem.shape[:2])
    dep = observations['depth']
    dep = dep.reshape(dep.shape[:2])

    # undo the modifications for next iter
    pointss = c_pointss.detach().cpu().numpy()
    rgbss = c_rgbss.detach().cpu().numpy()
    semanticss = c_semanticss.detach().cpu().numpy()
    depthss = c_depthss.detach().cpu().numpy()

    del T_world
    del points
    del rgbs
    del semantics
    del depths
    del c_pointss
    del c_rgbss
    del c_semanticss
    del c_depthss
    del round_agent
    del X
    del Y
    del Z
    del XZ_ff
    del XZ_fc
    del XZ_cc
    del XZ_cf
    del XZ
    del XYZ
    del idx
    del rgbss_4
    del semanticss_4
    del depthss_4
    
    torch.cuda.empty_cache()

    return pointss, rgbss, semanticss, depthss, rgb, sem, dep, height_rgb, height_sem, height_dep


# map depth to 3d point cloud
def depth2local3d(depth, FOV):
    depth = depth[..., 0]
    h, w = depth.shape
    f = 0.5 / np.tan(FOV / 2.) * float(w)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)
    dfl = depth.flatten()

    return np.concatenate(\
            [np.expand_dims(dfl * (xv.flatten() - w / 2.) / f, axis=-1),\
             - np.expand_dims(dfl * (yv.flatten() - h / 2.) / f, axis=-1),\
             - np.expand_dims(dfl, axis=-1),\
            ],\
            axis=1,\
        )
# color to correspodent point
def color2local3d(color):
    return np.concatenate([\
            np.expand_dims(color[..., 0].flatten(), axis=-1),\
            np.expand_dims(color[..., 1].flatten(), axis=-1),\
            np.expand_dims(color[..., 2].flatten(), axis=-1),\
            ], axis=1)
# translate point cloud from camera space to world space
def generate_pc(T_world, depth, fov):
    points = depth2local3d(depth, fov)
    cam = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    cam = np.matmul(cam, T_world.T)
    return cam[:, 0:3]

# translate point cloud from world space to camera space
def pc2local(T_world, pointss):
    cam = np.concatenate((pointss, np.ones((pointss.shape[0], 1))), axis=1)
    cam = np.matmul(cam, np.linalg.inv(T_world.T))
    return cam[:, 0:3]

# make sure every point cover four points
def repeat4(Y):
    Y = np.concatenate((Y, Y), axis=0)
    return np.concatenate((Y, Y), axis=0)

def get_observations(env, observations, pointss, rgbss, semanticss, depthss,\
        height, offset, area_x, area_z, h, w, fov, ignore, num_channel, id2cat, semantics=None): 
    # get transformation matrix of current state
    quaternion = env._sim.get_agent_state().sensor_states['depth'].rotation
    translation = env._sim.get_agent_state().sensor_states['depth'].position 
    # Transformation matrix: camera sees -> location in world
    rotation = as_rotation_matrix(quaternion)
    T_world = np.eye(4)
    T_world[0:3, 0:3] = rotation
    T_world[0:3, 3] = translation


    # get center of agent of current state

    cx, center, cz = env._sim.get_agent_state().position
    # get point_cloud(world) and corresponding rgb & semantics
    points = generate_pc(T_world, observations['depth'], fov)
    rgbs = color2local3d(observations['rgb'])
    depths = observations['depth'].flatten()

    if semantics is None:
        semantics = observations['semantic'].flatten()


    # accumulate step
    # pointss: whole points viewed (world)
    # rgbss, semanticss: corresponding rgb, semantic coloring info
    if pointss is None:
        pointss = points
        rgbss = rgbs
        semanticss = semantics
        depthss = depths
    else:
        pointss = np.concatenate((pointss, points), axis=0)
        rgbss = np.concatenate((rgbss, rgbs), axis=0)
        semanticss = np.concatenate((semanticss, semantics), axis=0)
        depthss = np.concatenate((depthss, depths), axis=0)
    #Store global values 
    c_pointss = pointss.copy()
    c_rgbss = rgbss.copy()
    c_semanticss = semanticss.copy()
    c_depthss = depthss.copy()

    # remove roof & lower floor
    roof_thre = center + height / 2. + offset * 4
    floor_thre = center - height / 2. - offset
    no_roof = (pointss[:, 1] < roof_thre) & (pointss[:, 1] >floor_thre)
    pointss =  pointss[no_roof, :]
    rgbss = rgbss[no_roof, :]
    semanticss = semanticss[no_roof]
    depthss = depthss[no_roof]
    
    # remove ceiling & misc
    t_sem = np.vectorize(lambda x: id2cat.get(x, num_channel))(semanticss)
    valid = None
    for ctg in ignore:
        if valid is None:
            valid = (t_sem != ctg)
        else:
            valid = valid & (t_sem != ctg)

    pointss = pointss[valid, :]
    rgbss = rgbss[valid, :]
    semanticss = semanticss[valid]
    depthss = depthss[valid]

    # map global point according to agent's rotation and position
    quaternion = env._sim.get_agent_state().rotation
    translation = env._sim.get_agent_state().position
    # Transformation matrix: camera sees -> location in world
    rotation = as_rotation_matrix(quaternion)
    T_world = np.eye(4)
    T_world[0:3, 0:3] = rotation
    T_world[0:3, 3] = translation
    pointss = pc2local(T_world, pointss)
    
    round_agent = (np.abs(pointss[:, 0]) < area_x / 2.)\
    & (np.abs(pointss[:, 2]) < area_z / 2.)

    pointss = pointss[round_agent, :]
    rgbss = rgbss[round_agent, :]
    semanticss = semanticss[round_agent]
    depthss = depthss[round_agent]

    # generate top-down height map
    scale_h = area_z / h
    scale_w = area_x / w


    X = (pointss[:, 0] + (area_x / 2.)) / float(scale_w)
    Y = pointss[:, 1]
    Z = (pointss[:, 2] + (area_z / 2.)) / float(scale_h)

    # 1 for 4
    XZ_ff = np.column_stack((np.floor(X), np.floor(Z))) #0, 1, ... n-1
    XZ_fc = np.column_stack((np.floor(X), np.ceil(Z))) 
    XZ_cc = np.column_stack((np.ceil(X), np.ceil(Z)))
    XZ_cf = np.column_stack((np.ceil(X), np.floor(Z)))
    XZ = np.concatenate((np.concatenate((XZ_ff, XZ_fc), axis=0),\
            np.concatenate((XZ_cc, XZ_cf), axis=0)), axis=0) #0, 1, ... 4n - 1 
    assert h == w, "Currently only support square!"
    XZ[XZ >= h] = h - 1
    XZ[XZ < 0.] = 0. 
    Y = repeat4(Y)

    XYZ = np.column_stack((XZ, Y))
    df = pd.DataFrame(XYZ)
    idx = df.groupby([0, 1])[2].transform(max) == df[2]
    
    idx = idx.values

    height_rgb = np.zeros((h, w, 3))
    height_sem = -np.ones((h, w), dtype=np.int32)
    height_dep = np.zeros((h, w))

    rgbss_4 = repeat4(rgbss)
    semanticss_4 = repeat4(semanticss)
    depthss_4 = repeat4(depthss)

    XZ = XZ.astype(int)
    height_rgb[XZ[idx][:, 0], XZ[idx][:, 1], :] = rgbss_4[idx, :]
    height_sem[XZ[idx][:, 0], XZ[idx][:, 1]] = semanticss_4[idx]
    height_dep[XZ[idx][:, 0], XZ[idx][:, 1]] = depthss_4[idx]
    
    height_rgb = np.swapaxes(height_rgb, 0, 1)[:, :, [2, 1, 0]] # rgb
    height_sem = np.swapaxes(height_sem, 0, 1)
    height_dep = np.swapaxes(height_dep, 0, 1)

    
    rgb = observations['rgb'][:, :, [2, 1, 0]] # rgb
    sem = observations['semantic']
    sem = sem.reshape(sem.shape[:2])
    dep = observations['depth']
    dep = dep.reshape(dep.shape[:2])

    # undo the modifications for next iter
    pointss = c_pointss
    rgbss = c_rgbss
    semanticss = c_semanticss
    depthss = c_depthss

    return pointss, rgbss, semanticss, depthss, rgb, sem, dep, height_rgb, height_sem, height_dep


#def preprocess(obs, unknown, p_type="sem", num_categories=40):
#    assert p_type in ["sem"], "Not supported preprocess type " + str(p_type)
#    if p_type == 'sem':
#        h, w = obs.shape[0], obs.shape[1]
#        x = obs.reshape(-1)
#        target = ((np.eye(num_categories)[x % num_categories]).T).reshape((num_categories, h, w))
#        target[:, unknown == 0] = 0
#        target = torch.tensor(target).type(torch.FloatTensor) 
#        target = torch.cat((target, (1. - torch.tensor(unknown).type(torch.FloatTensor).unsqueeze(0))), dim=0)
#        return target



def gaussian_blur(out, sigma=3):
    # out: B * 1 * H * W
    return scipy.ndimage.gaussian_filter(out, sigma)

# size_x, size_y: not essentially equal to size of Q output
def greedy_action(Q, obs, count, replay_start, initial_ep, final_ep,
        final_frame, in_channels, img_dir, tag): 
    h, w = obs.shape[-2], obs.shape[-1]
    assert (h % 2 == 0) and (w % 2 == 0), "Only works for even map shape"
    
    num_actions = h * w
    center_x = (float(h) - 1.) / 2.
    center_y = (float(w) - 1.) / 2.
    
    out = None
    # action_id: one of [0, 256*256]
    if count >= replay_start:
        eps = max(initial_ep - (initial_ep - final_ep) * float(count - replay_start) / float(final_frame - replay_start), final_ep)
        sample = random.random()

        if sample < eps: 
            action_id = random.choice(range(num_actions))
        else:
            out = Q(obs)[0].cpu()
            cv2.imwrite('%s/obs_used.png' % (img_dir),
                    d3_41_colors_rgb[torch.argmax(obs[0][:in_channels],
                        dim=0).detach().cpu().numpy()])

         #   if obs.shape[1] == 2 * in_channels 
        #        cv2.imwrite('%s/conf_used.png'% (img_dir), obs[0][-1].detach().cpu().numpy() * 255.)
        #    except:
        #        pass
            plt.imshow(out.detach().squeeze(0).cpu().numpy(), cmap='jet')
            plt.axis('off')
            plt.savefig("%s/q_map.png" % (img_dir))
            plt.clf() 
            #out = gaussian_blur(out)
            action_id = out.reshape(-1).max(0)[1] # [0, h*w - 1)
            with open('%s/goal_used.txt' % (img_dir), 'w') as f:
                f.write(str(tag))
            #action_id = torch.argmax(out)
            #print(action_id.shape)
           # assert False, "Pause"
    else:
        action_id = random.choice(range(num_actions))
    # (x, y) : grid of target
    y = float(action_id % w)
    x = float(action_id - y) / float(w)
    

    if out is None:
        out = np.zeros((h, w))
        out[int(x), int(y)] = 255.
    else:
        out = out.detach().numpy()
  #  cv2.imwrite(str(count) + "_Qmap.png", (out-np.min(out))/(np.max(out) - np.min(out)) * 255.)
    if center_x >= x:
        if center_y < y:
            # turn right for theta
            theta = np.arctan((y - center_y) / (center_x - x)) / np.pi * 180.
        else:
            # turn right for theta
            theta = np.arctan((center_y - y) / (center_x - x)) / np.pi * 180.
            theta = 360.- theta
    elif center_y < y:
        # turn right for theta
        theta = np.arctan((y - center_y) / (x - center_x)) / np.pi * 180.
        theta = 180. - theta
    else:
        # turn right for theta
        theta = np.arctan((center_y - y) / (x - center_x)) / np.pi * 180.
        theta = 180. + theta

    theta = int(theta)

    if theta < 180:
        return action_id, ["TURN_RIGHT"] * theta + ["MOVE_FORWARD"]
    else:
        return action_id, ["TURN_LEFT"] * (360 - theta) + ["MOVE_FORWARD"]

'''
def shortest_path(env, objs, success_distance_offset):
    shortest_path = np.inf
    for obj in objs:
        nav_points\
        = env.sim.get_straight_shortest_path_points(env.sim.get_agent_state().position,obj.aabb.center)
        if nav_points == []:
            continue
        radius = max(abs(obj.aabb.sizes[0]),abs(obj.aabb.sizes[2]))  * np.sqrt(2) /2.
        # find the last point in this path that is on the object
        j = 0
        while j < len(nav_points):
            dist = np.sqrt((nav_points[-1-j][0]- nav_points[-1][0])**2 + (nav_points[-1-j][2]- nav_points[-1][2])**2))
            if dist > radius:
                break
            j += 1
        # now: -1-j -> the first point not on the object, -1-k -> the last
        # point on the object
        # find the last point in this path that is not in the successful range
        k = j - 1
        while j < len(nav_points):
            dist\
            = env.sim.geodesic_distance(nav_points[-1-j],\
            nav_points[-1-k])
            if dist > success_distance_offset:
                break
            j += 1
        # now: -1-j -> the first point not successful
        # shortest path distance -> geodesic distance between agent start point
        # and the last successful point
        k = j - 1
        cur_distance\
        = env.sim.geodesic_distance(env.sim.get_agent_state().position, nav_points[-1-k])
        if cur_distance < shortest_path:
            shortest_path = cur_distance
    return shortest_path
'''

def geodesic_distance(env):
    p_agent = env._sim.get_agent_state().position
    return min([env._sim.geodesic_distance(p_agent, goal.position) for goal in env.episodes[0].goals])

# two outputs
# end? reward of this step?
def success(env, success_distance, success_reward):
    # end the episode successfully
    if geodesic_distance(env) < success_distance:
        return True, success_reward

    return False, 0.
def check_terminate(new_final_obs, in_channels, area_x, tag, success_distance):
    new_final_obs = new_final_obs.squeeze(0).cpu().numpy()
    assert new_final_obs.shape[-1] == new_final_obs.shape[-2], "Only support\
            square maps"
    limit = int(success_distance * new_final_obs.shape[-1] / (area_x))
    gap = area_x / new_final_obs.shape[-1]
    upper = int(new_final_obs.shape[-1] / 2.-1/2.+limit)
    if upper >= new_final_obs.shape[-1]:
        upper = new_final_obs.shape[-1]-1
    lower = int(new_final_obs.shape[-1] / 2.-limit)
    if lower < 0:
        lower = 0
    candidate = new_final_obs[:in_channels, lower:upper+1, lower:upper+1]
    if np.sum(candidate[tag]) == 0:
        return False
    if lower == upper:
        return candidate[tag, 0, 0] == 1
    cursor = 0
    size = upper - lower
    visited = [(int(size/2.), int(size/2.)),\
            (int(size/2.)+1, int(size/2.)),\
            (int(size/2.)+1, int(size/2.)+1),\
            (int(size/2.), int(size/2.)+1)]
    costs = [gap, gap, gap, gap]
    while cursor < len(visited):
        current_end = len(visited)
        for v in range(cursor, current_end):
            x, y = visited[v]
            cost = costs[v]
            if candidate[tag, x, y] == 1:
                return True
            if cost + gap > success_distance:
                continue
            if x>0:
                if y > 0:
                    if (x-1, y-1) not in visited:
                        if candidate[tag, x-1, y-1] == 1 or candidate[1, x-1,y-1] == 1 or\
                            candidate[-1, x-1, y-1] == 1:
                            visited.append((x-1, y-1))
                            costs.append(cost + gap)
                
                if y < upper-lower:
                    if (x-1, y+1) not in visited:
                        if candidate[tag, x-1, y+1] == 1 or candidate[1, x-1,y+1] == 1 or\
                            candidate[-1, x-1, y+1] == 1:
                            visited.append((x-1, y+1))
                            costs.append(cost+ gap)
                if (x-1, y) not in visited:
                    if candidate[tag, x-1, y] == 1 or candidate[1, x-1, y] ==\
                    1 or candidate[-1, x-1, y] == 1:
                        visited.append((x-1, y))
                        costs.append(cost+ gap)
            
            if x < upper - lower:
                if y > 0:
                    if (x+1, y-1) not in visited:
                        if candidate[tag, x+1, y-1] == 1 or candidate[1, x+1,y-1] == 1 or\
                            candidate[-1, x+1, y-1] == 1:
                            visited.append((x+1, y-1))
                            costs.append(cost+ gap)
                
                if y < upper-lower:
                    if (x+1, y+1) not in visited:
                        if candidate[tag, x+1, y+1] == 1 or candidate[1, x+1,y+1] == 1 or\
                            candidate[-1, x+1, y+1] == 1:
                            visited.append((x+1, y+1))
                            costs.append(cost+ gap)
                if (x+1, y) not in visited:
                    if candidate[tag, x+1, y] == 1 or candidate[1, x+1, y] ==\
                    1 or candidate[-1, x+1, y] == 1:
                        visited.append((x+1, y))
                        costs.append(cost+ gap)
            if y > 0:
                if (x, y-1) not in visited:
                    if candidate[tag, x, y-1] == 1 or candidate[1, x, y-1] ==\
                    1 or candidate[-1, x, y-1] == 1:
                        visited.append((x, y-1))
                        costs.append(cost+ gap)
            if y < upper - lower:
                if (x, y+1) not in visited:
                    if candidate[tag, x, y+1] == 1 or candidate[1, x, y+1] ==\
                    1 or candidate[-1, x, y+1] == 1:
                        visited.append((x, y+1))
                        costs.append(cost+ gap)
        cursor = current_end
    return False

d3_41_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
        [0, 0, 0]
    ],
    dtype=np.uint8,
)

d3_14_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [0, 0, 0]
    ],
    dtype=np.uint8,
)

tostr = np.vectorize(lambda x: str(x))
toint = np.vectorize(lambda x: int(x))

def preprocess(sem_gt, num_channel, id2cat, underfeet):
    assert num_channel == 41
    sem_gt = np.vectorize(lambda x: id2cat.get(x, num_channel))(sem_gt)
    sem_gt = toint(sem_gt)
    unknown = toint((sem_gt > 0) & (sem_gt < num_channel))

    sem_in = sem_gt.copy()
    sem_gt = torch.from_numpy(sem_gt) - 1
    sem_gt[unknown == 0] = num_channel - 1

    sem_in, unknown = one_hot(sem_in, unknown, num_channel)

    sem_gt = fill_holes(sem_gt, num_channel, underfeet)

    return sem_gt.long(), unknown, sem_in.float()

# fill in holes of sem_gt
def fill_holes(sem_gt, num_channel, underfeet):
    assert sem_gt.shape[0] == sem_gt.shape[1], "currently only support square"
    # fill in gaps
    tmp = sem_gt.clone()
    for i in range(sem_gt.shape[0]):
        for j in range(sem_gt.shape[1]):
            if sem_gt[i, j] != num_channel - 1:
                continue
            bkt = {}
            for p in range(num_channel):
                bkt[p] = 0
            if i != 0:
                bkt[int(sem_gt[i - 1, j])] += 1
            if j != 0:
                bkt[int(sem_gt[i, j - 1])] += 1
            if i != sem_gt.shape[0] - 1:
                bkt[int(sem_gt[i + 1, j])] += 1
            if j != sem_gt.shape[1] - 1:
                bkt[int(sem_gt[i, j + 1])] += 1
            cmax = 0
            current = -1
            for c in range(num_channel - 2): 
                if bkt[c] > cmax:
                    cmax = bkt[c]
                    current = c
            if current != -1:
                tmp[i, j] = current
    sem_gt = tmp

            
   # cv2.imwrite("smooth.png", d3_41_colors_rgb[sem_gt])

    # fill in holes underfeet
   # center = (sem_gt.shape[0] - 1.) / 2.
   # radius = underfeet / 2.
   # lower = int(center - radius)
   # upper = int(center + radius) + 1
   # black = sem_gt != (num_channel - 1)
   # not_null = black[lower : upper, lower : upper]
   # sem_gt[lower : upper, lower : upper][not_null==False] = 1
  #  cv2.imwrite("test.png", d3_41_colors_rgb[sem_gt])
  #  assert False, "Pause"
    return sem_gt



def one_hot(x, unknown, num_channel):
    # unknown: tensor, (h, w)
    # sem_in: tensor, (h, w)
    x -= 1
    x[unknown == 0] = num_channel - 1 # void: last
    h, w = x.shape[0], x.shape[1]
    x = x.reshape(-1)

    target = np.eye(num_channel)[x]
    target = target.reshape((h, w, num_channel))

    target[unknown == 0, :] = 0
    target[unknown == 0, num_channel - 1] = 1

    sem_in = torch.from_numpy(target)
    unknown = torch.from_numpy(unknown)

    sem_in = sem_in.permute(2, 0, 1)
    # random crop
    assert h == w, "Currently only support square image!"
    mask = random_region_mask(h)
    sem_in[:, mask == 1] = 0
    sem_in[num_channel - 1, mask == 1] = 1
    return sem_in, unknown.unsqueeze(0)

def random_region_mask(fine_size, res=1/128, density=0.25):
    assert int(res * fine_size) == res * fine_size, "Illegal resolution!"
    rsize = int(res * fine_size)
    m = torch.nn.Upsample(scale_factor=int(1. / res), mode='bicubic')
    while True:
        mask = m(torch.rand(rsize, rsize).uniform_(0, 1).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        lt, ht = mask < density, mask >= density
        mask[lt], mask[ht] = 1., 0.
        area = torch.mean(mask)
        if (area > 0.2) and (area < 0.3):
            break
    
    return mask

def cmplt_iou(sem_gts, sem_ins, outputs, mode='category'):
    num_channel = sem_ins.shape[1]
    sem_ins = torch.argmax(sem_ins, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    
    #masked but not unknown region
    roi = (sem_gts != sem_ins) & (sem_gts != (num_channel - 1))
    
    if mode == 'category':
        result = np.zeros((num_channel-1, 1))
        active = np.zeros((num_channel-1, 1))
        
        for k in range(num_channel - 1):
            its = (outputs == k) & (sem_gts == k) & roi
            union = ((sem_gts == k) | (outputs == k)) & roi
            if 0 == float(np.sum(toint(union.cpu().numpy()))):
                continue
            result[k, 0] = float(np.sum(toint(its.cpu().numpy()))) / float(np.sum(toint(union.cpu().numpy())))
            active[k, 0] = 1.
        return result, active
    else:
        assert False, "Undefined mode %s" % (mode)
    

def cmplt_accuracy(sem_gts, sem_ins, outputs, mode="cmplt"):
    num_channel = sem_ins.shape[1]
    # one-hot to class
    sem_ins = torch.argmax(sem_ins, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    

    roi =(sem_gts != sem_ins) & (sem_gts != num_channel-1) # unknown regions
    match = (sem_gts == outputs)
    
    roi_in = (sem_ins != num_channel-1)
    match_in = (sem_ins == outputs)

    if mode == 'cmplt':
        if float(np.sum(toint(roi.cpu().numpy()))) == 0:
            return None
        return float(np.sum(toint((roi & match).cpu().numpy()))) / float(np.sum(toint(roi.cpu().numpy())))
    elif mode == 'copy':
        if float(np.sum(toint(roi_in.cpu().numpy()))) == 0:
            return None
        return float(np.sum(toint((roi_in & match_in).cpu().numpy()))) / float(np.sum(toint(roi_in.cpu().numpy())))
    elif mode == 'category':
        result = np.zeros((num_channel-1, 1))
        active = np.zeros((num_channel-1, 1))
        for k in range(num_channel - 1): #41/14
            roi_ = roi & (sem_gts == k)
            match_ = match & (sem_gts == k)
            if float(np.sum(toint(roi_.cpu().numpy()))) == 0:
                continue
            result[k, 0] = float(np.sum(toint((roi_ & match_).cpu().numpy()))) / float(np.sum(toint(roi_.cpu().numpy())))
            active[k, 0] = 1.
        return result, active
    else:
        assert False, "Undefined mode %s" % (mode)    
            
Forty2Thirteen = {
    '0': '0',
    '1': '12',
    '2': '5',
    '3': '4',
    '4': '7',
    '5': '10',
    '6': '8',
    '7': '6',
    '8': '7',
    '9': '13',
    '10': '9',
    '11': '1',
    '12': '7',
    '13': '6',
    '14': '7',
    '15': '7',
    '16': '7',
    '17': '3',
    '18': '6',
    '19': '6',
    '20': '7',
    '21': '7',
    '22': '11',
    '23': '7',
    '24': '7',
    '25': '6',
    '26': '6',
    '27': '7',
    '28': '7',
    '29': '7',
    '30': '7',
    '31': '6',
    '32': '13',
    '33': '6',
    '34': '6',
    '35': '7',
    '36': '6',
    '37': '7',
    '38': '7',
    '39': '7',
    '40': '7',
    '41': '0'
}




def shrink_obs(obs, scale_factor=0.25):
    height = obs.shape[2]
    width = obs.shape[3]
    assert (int(height * scale_factor) == height * scale_factor) and (int(width * scale_factor) ==  width * scale_factor), "illegal scale_factor %s for image size %s x %s" % (scale_factor, height, width)
   
    resizer = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
    return resizer(obs)
   
   


    

if __name__ == '__main__':
    img = cv2.imread("../baseline_half/tmp.png")
    black = torch.from_numpy(np.sum(img, axis=2) == 0)
    
    sem_gt = torch.rand(512, 512) * 40

    sem_gt = sem_gt.int()

    sem_gt[black] = 40


    cv2.imwrite("pre.png", d3_41_colors_rgb[sem_gt])


    fill_holes(sem_gt, 41, 200)
