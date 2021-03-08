import cv2
import random
import torch
import torch.nn.functional as F
import numpy as np
from quaternion import as_rotation_matrix, as_euler_angles
try:
    from utils import generate_pc, color2local3d, repeat4, pc2local, pc2local_gpu, d3_41_colors_rgb
except:
    from .utils import generate_pc, color2local3d, repeat4, pc2local, pc2local_gpu, d3_41_colors_rgb
import habitat
import json
from skmpe import mpe
from tqdm import tqdm
import skimage
# a new memory, that stores a global map
class Mapper:
    def __init__(self, device, m_x_size, m_z_size, space_x, space_z, num_class,
            fov, ignore):
        assert m_x_size == m_z_size, "map has to be a square!"
        assert space_x == space_z, "map has to be a square!"
        self.fov = fov
        self.ignore = ignore
        self.id2cat = None
        self.roof_thre = None
        self.floor_thre = None
        self.num_class = num_class
        self.device = device
        self.map = torch.ones(m_x_size, m_z_size).int().to(device)
        self.map = self.map * (num_class)# global map
        self.m_x_size = m_x_size # x shape of global map
        self.m_z_size = m_z_size # y shape of global map
        self.space_x = space_x # x map size in meter of global map
        self.space_z = space_z # y map size in meter of global map
        self.aligned_translation = None
        self.aligend_quaternion =  None # get aligned angle to match 'facing upward' in real world
    # clear map record
    # put agent back to center, facing upward
    # new alignment needed
    def reset(self, id2cat, roof_thre, floor_thre):
        self.roof_thre = roof_thre
        self.floor_thre = floor_thre
        self.id2cat = id2cat
        self.map = torch.ones(self.map.shape[0],
                self.map.shape[1]).int().to(self.device)
        self.map = self.map * (self.num_class)# global map
        self.aligned_translation = None
        self.aligned_quaternion = None

    def append(self, quaternion, translation, observations, raw_semantics=None):
        
        if self.aligned_translation is None:
            self.aligned_translation = translation.copy()
            self.aligned_quaternion = quaternion.copy()

        # transform camera -> location in world
        rotation = as_rotation_matrix(quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = translation
        T_world = torch.from_numpy(T_world)

        # get points rgbs, depths and semantics
        depth = observations['depth'][..., 0]
        depth = torch.from_numpy(depth).to(self.device)
        h, w = depth.shape
        f = float(0.5 / np.tan(self.fov / 2.) * float(w))
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
        xv, yv = torch.meshgrid(x, y)
        xv, yv = xv.t().to(self.device), yv.t().to(self.device)
        dfl = depth.reshape(-1)
        points = torch.cat((\
            (dfl * (xv.reshape(-1) - w / 2.) / f).unsqueeze(-1),\
            - (dfl * (yv.reshape(-1) - h / 2.) / f).unsqueeze(-1),\
            - dfl.unsqueeze(-1)), dim=1)

        cam = torch.cat((points, torch.ones((points.shape[0],
            1)).to(self.device)),
                    dim=1).cpu()
        cam = torch.mm(cam, T_world.t().float()).to(self.device)
        points = cam[:, 0:3]

        # use system semantic observation or user semantic observation
        if raw_semantics is None:
            # get gt semantic sensor input
            semantics = observations['semantic'].flatten()
            # map from objId to category [0, num_channel-1]
            semantics = np.vectorize(lambda x: self.id2cat.get(x,\
                self.num_class + 1))(semantics)
            
            semantics = torch.from_numpy(semantics).to(self.device)
            semantics -= 1
            semantics[((semantics < 0) | (semantics >= self.num_class + 1))] = self.num_class
        else:
            semantics\
            = torch.from_numpy(raw_semantics).to(self.device).flatten()

        valid = None
        for ctg in self.ignore:
            if valid is None:
                valid = (semantics != (ctg - 1))
            else:
                valid = valid & (semantics != (ctg - 1))
        valid = valid & (points[:, 1] < self.roof_thre) & (points[:, 1]
                > self.floor_thre)
        points = points[valid]
        semantics = semantics[valid]

        rotation = as_rotation_matrix(self.aligned_quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = self.aligned_translation
        T_world = torch.from_numpy(T_world)

        cam = torch.cat((points.float(), torch.ones(points.shape[0],
            1).to(self.device).float()),
                            dim=1).cpu()
        cam = torch.mm(cam, torch.inverse(T_world.t()).float()).to(self.device)


        pointss = cam[:, 0:3]


        round_agent = (torch.abs(pointss[:, 0]) < self.space_x / 2.)\
                & (torch.abs(pointss[:, 2]) < self.space_z / 2.)
        pointss = pointss[round_agent, :]


        scale_h = self.space_z / self.m_z_size
        scale_w = self.space_x / self.m_x_size

      
        X = (pointss[:, 0:1] + (self.space_x / 2.)) / float(scale_w)
        Y = pointss[:, 1:2]
        Z = (pointss[:, 2:3] + (self.space_z / 2.)) / float(scale_h)


        XZ_ff = torch.cat((torch.floor(X), torch.floor(Z)), dim=1)
        XZ_fc = torch.cat((torch.floor(X), torch.ceil(Z)), dim=1)
        XZ_cc = torch.cat((torch.ceil(X), torch.ceil(Z)), dim=1)
        XZ_cf = torch.cat((torch.ceil(X), torch.floor(Z)), dim=1)
       
        XZ = torch.cat((torch.cat((XZ_ff, XZ_fc), dim=0), 
            torch.cat((XZ_cc, XZ_cf), dim=0)), dim=0)

        XZ[XZ >= self.m_x_size] = self.m_x_size - 1.
        XZ[XZ < 0.] = 0.
        Y = torch.cat((Y, Y), dim=0)
        Y = torch.cat((Y, Y), dim=0)


        XYZ = torch.cat((XZ, Y), dim=1)

        sort_ind =torch.argsort(XYZ[..., 2])

        XYZ = XYZ[sort_ind].long()
        

        height_sem = torch.ones(self.m_x_size, self.m_z_size, device=self.device) * (self.num_class)
        height_sem = height_sem.int()


        semanticss = torch.cat((semantics[round_agent],
            semantics[round_agent]), dim=0)
        semanticss = torch.cat((semanticss, semanticss), dim=0)
        semanticss = semanticss.int() 
        semanticss = semanticss[sort_ind]
        
        height_sem[XYZ[:, 0], XYZ[:, 1]] = semanticss[:]
        height_sem = torch.rot90(height_sem, 3, [0, 1])
        height_sem = torch.flip(height_sem, [1])


        self.map[height_sem != self.num_class] = height_sem[height_sem !=
                self.num_class]
    def get_orient(self, rotation):
        corner = rotation / self.aligned_quaternion
        matrix = as_rotation_matrix(corner) 
        cos = matrix[0, 0]
        sin = matrix[0, 2]
        sin = min(1., max(-1., sin))
        cos = min(1., max(-1., cos))
        if sin >= 0:
            if cos >= 0:
                angle = np.arcsin(sin)
            else:
                angle = np.pi - np.arcsin(sin)
        elif cos >= 0:
            angle = np.arcsin(sin)
        else:
            angle = -np.pi - np.arcsin(sin)
        angle = angle / np.pi * 180.
        if angle < 0.:
            angle += 360.
        angle = int(angle) // 5
        return angle

    def get_map_local(self, translation, size_x=None, size_z=None, has_map=True):

        size_x, size_z = int(size_x), int(size_z) 
        points = torch.from_numpy(translation).unsqueeze(0).to(self.device)
        rotation = as_rotation_matrix(self.aligned_quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = self.aligned_translation
        T_world = torch.from_numpy(T_world).to(self.device)

        cam = torch.cat((points.float(), torch.ones(points.shape[0],
            1).to(self.device).float()),
                            dim=1)
        cam = torch.mm(cam, torch.inverse(T_world.t()).float())

        pointss = cam[:, 0:3]
        anchor_x = pointss[0, 0] + self.space_x / 2.
        anchor_z = pointss[0, 2] + self.space_z / 2.
        anchor_x /= self.space_x / self.m_x_size
        anchor_z /= self.space_z / self.m_z_size
        anchor_x = int(anchor_x)
        anchor_z = int(anchor_z)
        anchor_x -= int(size_x / 2.)
        anchor_z -= int(size_z / 2.)
        if anchor_x < 0:
            anchor_x = 0
        elif anchor_x > self.m_x_size - size_x:
            anchor_x = self.m_x_size - size_x
        if anchor_z < 0:
            anchor_z = 0
        elif anchor_z > self.m_z_size - size_z:
            anchor_z = self.m_z_size - size_z
        if not has_map:
            return int(anchor_z + size_z /2), int(anchor_x + size_x / 2)
        height_sem = self.map[anchor_z : anchor_z + size_z, anchor_x : anchor_x
                + size_x]
        return self.render(height_sem)

    def get_map_local_rot(self, quaternion, translation, size_x=None, size_z=None, has_map=True):
        corner = quaternion / self.aligned_quaternion
        matrix = as_rotation_matrix(corner) 
        cos = matrix[0, 0]
        sin = matrix[0, 2]
        sin = min(1., max(-1., sin))
        cos = min(1., max(-1., cos))
        if sin >= 0:
            if cos >= 0:
                angle = np.arcsin(sin)
            else:
                angle = np.pi - np.arcsin(sin)
        elif cos >= 0:
            angle = np.arcsin(sin)
        else:
            angle = -np.pi - np.arcsin(sin)
        angle = angle / np.pi * 180.
        if angle < 0.:
            angle += 360.
        if angle >= 360.:
            angle -= 360.



        size_x, size_z = size_x * 2, size_z * 2 
        size_x, size_z = int(size_x), int(size_z) 
        points = torch.from_numpy(translation).unsqueeze(0).to(self.device)
        rotation = as_rotation_matrix(self.aligned_quaternion)
        T_world = np.eye(4)
        T_world[0:3, 0:3] = rotation
        T_world[0:3, 3] = self.aligned_translation
        T_world = torch.from_numpy(T_world).to(self.device)

        cam = torch.cat((points.float(), torch.ones(points.shape[0],
            1).to(self.device).float()),
                            dim=1)
        cam = torch.mm(cam, torch.inverse(T_world.t()).float())

        pointss = cam[:, 0:3]
        anchor_x = pointss[0, 0] + self.space_x / 2.
        anchor_z = pointss[0, 2] + self.space_z / 2.
        anchor_x /= self.space_x / self.m_x_size
        anchor_z /= self.space_z / self.m_z_size
        anchor_x = int(anchor_x)
        anchor_z = int(anchor_z)
        anchor_x -= int(size_x / 2.)
        anchor_z -= int(size_z / 2.)
        if anchor_x < 0:
            anchor_x = 0
        elif anchor_x > self.m_x_size - size_x:
            anchor_x = self.m_x_size - size_x
        if anchor_z < 0:
            anchor_z = 0
        elif anchor_z > self.m_z_size - size_z:
            anchor_z = self.m_z_size - size_z
        if not has_map:
            return int(anchor_z + size_z /2), int(anchor_x + size_x / 2)

        height_sem = self.map[anchor_z : anchor_z + size_z, anchor_x : anchor_x
                + size_x].cpu()

        height_sem = self.rotate_image(height_sem, angle)

        f_x, f_z = int(height_sem.shape[0] / 2), int(height_sem.shape[1] / 2)
        s_x, s_z = int(height_sem.shape[0] / 4), int(height_sem.shape[1] / 4)

        height_sem = height_sem[s_x: s_x + f_x, s_z : s_z + f_z]
        height_sem = torch.from_numpy(height_sem)

        return self.render(height_sem)
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(image.numpy(), rot_mat, image.shape[1::-1],
                  flags=cv2.INTER_NEAREST, borderValue=self.num_class)
        return result
    def get_map_global(self):
        return self.render(self.map)
    def render(self, src):
        height_sem = F.one_hot(src.long(), num_classes=self.num_class
                + 1) 

        height_sem = height_sem.permute(2, 0, 1)
        return height_sem.cpu()
    def cat2obst(self, tdmap):
        bmap = (tdmap == 1) | (tdmap == self.num_class) 
        bmap = (~bmap).int()
        return bmap

if __name__ == '__main__':
    device = torch.device("cuda")
    m_x_size, m_z_size = 1024, 1024
    space_x, space_z = 48., 48.
    l_x, l_z = 128, 128
    num_class = 40
    fov = np.pi / 2.
    ignore = [17, 40]
    height = 1.25
    offset = 0.3
    floor_threshold = 0.1

    config = habitat.get_config(config_paths="/local/crv/yiqing/SCNav/configs/agent_test.yaml")
    config.defrost()
    config.TASK.SUCCESS_DISTANCE = -float("inf")
    config.SIMULATOR.TURN_ANGLE = 30
    config.ENVIRONMENT.MAX_EPISODE_STEPS = float("inf")
    config.ENVIRONMENT.MAX_EPISODE_SECONDS = float("inf")
    config.freeze()
    env = habitat.Env(config=config)
    mapper = Mapper(device, m_x_size, m_z_size, space_x, space_z, num_class,
                fov, ignore)
    observations = env.reset()

    start_position = env.sim.get_agent_state().position
    id2cat = {int(obj.id.split("_")[-1]): obj.category.index() for obj in
                            env.sim.semantic_annotations().objects}

    roof_thre = start_position[1] + height/2. + offset
    floor_thre = start_position[1] - height/2. - floor_threshold
    mapper.reset(id2cat, roof_thre, floor_thre)

    for step in tqdm(range(100)):

        quaternion = env._sim.get_agent_state().sensor_states['depth'].rotation
        translation = env._sim.get_agent_state().sensor_states['depth'].position
        
        mapper.append(quaternion, translation, observations)
        orient = mapper.get_orient(quaternion)
        print(orient)
        tdmap = mapper.get_map_global()
        lmap = mapper.get_map_local(translation, l_x, l_z)
        cv2.imwrite("tdmap.png", d3_41_colors_rgb[torch.argmax(tdmap, dim=0).numpy()])
        cv2.imwrite("lmap.png", d3_41_colors_rgb[torch.argmax(lmap, dim=0).numpy()])
        cv2.imwrite("rgb.png", observations['rgb'][..., [2, 1, 0]])
        while True:
            aid = input("Get action")
            if aid in ['1', '2', '3']:
                break
        action = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"][int(aid)-1]
        observations = env.step(action)

    np.save("maze.npy", mapper.cat2obst(mapper.map).cpu().numpy())
    cv2.imwrite("maze.png", mapper.cat2obst(mapper.map).cpu().numpy() * 255.)
