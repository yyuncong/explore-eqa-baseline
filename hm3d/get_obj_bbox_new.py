import os
import numpy as np

dataset_path = "/project/pi_chuangg_umass_edu/yuncong/data/original_2d_gt_seg_70"
scene_ids = os.listdir(dataset_path)
for scene_id in scene_ids:
    scene_path = os.path.join(dataset_path, scene_id)
    npy_files = os.listdir(scene_path)
    npy_files = [f for f in npy_files if f.endswith(".npy")]
    depth_npy_files = [f for f in npy_files if "depth" in f]
    semantic_npy_files = [f for f in npy_files if "semantic" in f]
    depth_maps = [np.load(os.path.join(scene_path, f)) for f in depth_npy_files]
    semantic_maps = [np.load(os.path.join(scene_path, f)) for f in semantic_npy_files]
    # print(obj_ids)
    # break

import os
import cv2
import json
import itertools
import quaternion # Remove this will cause invalid pointer error !!!!
import habitat_sim
import numpy as np
from tqdm import tqdm
from utils import config
from utils.dataset_interface import Objaverse, HM3D, ObjectFolder
from multisensory_simulator import MultisensorySimulator
from utils.config import sim_conf
from utils.cloud_point_utils import Reconstruct3D
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from PIL import Image
import copy
from collections import defaultdict 
import random
from utils.cloud_point_utils import Reconstruct3D, crop_points
from matplotlib import pyplot as plt

val_scene_ids = [
    "00800-TEEsavR23oF",
    "00802-wcojb4TFT35",
    "00813-svBbv1Pavdk", 
    "00814-p53SfW6mjZe",
    "00820-mL8ThkuaVTM", 
    "00824-Dd4bFSTQ8gi",
    "00829-QaLdnwvtxbs",
    "00832-qyAac8rV8Zk",
    "00835-q3zU7Yy5E5s", 
    "00839-zt1RVoi7PcG",
    "00843-DYehNKdT76V",
    "00848-ziup5kvtCCR",
    "00853-5cdEh9F2hJL",
    "00873-bxsVRursffK",
    "00876-mv2HUxq3B53",
    "00877-4ok3usBNeis",
    "00878-XB4GS9ShBRE",
    "00880-Nfvxx8J5NCo",
    "00890-6s7QHgap2fW",
    "00891-cvZr5TUy5C5"
]

scene_list = [
    '00871-VBzV5z6i1WS',
    '00808-y9hTuugGdiq',
    '00821-eF36g7L6Z9M',
    '00847-bCPU9suPUw9',
    '00844-q5QZSEeHe5g',
    '00823-7MXmsvcQjpJ',
    '00862-LT9Jq6dN3Ea',
    '00861-GLAQ4DNUx5U',
    '00823-7MXmsvcQjpJ',
    '00827-BAbdmeyTvMZ'
]

DATA_PATH = "/project/pi_chuangg_umass_edu/yuncong/data/"


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    # save the images without using plt
    for i, img in enumerate(arr):
        img.save(f"sample_{titles[i]}.png")




class GridSampler(MultisensorySimulator):
    def __init__(self, scene, new_objs=None, audio=False):
        action_space = {
            "look_left": habitat_sim.agent.ActionSpec("look_left", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_up": habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_down": habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=90.0))}
        cfg = sim_conf(scene, audio=audio)
        cfg.agents[0].action_space.update(action_space)

        super().__init__(cfg, new_objs)

        self.scene = scene
        _spec = cfg.agents[0].sensor_specifications[0]
        self.reconstructor = Reconstruct3D(
            _spec.resolution[0],
            _spec.resolution[1],
            float(_spec.hfov),
            _spec.position
        )
        # self.encoder = encoder

    @staticmethod
    def inside_p(pt, box):
        if pt[0] < box[0][0] or pt[0] > box[1][0]: return False
        if pt[1] < box[0][1] or pt[1] > box[1][1]: return False
        if pt[2] < box[0][2] or pt[2] > box[1][2]: return False
        if box[1][0] - pt[0] < 0 or pt[0] - box[0][0] < 0: return False
        if box[1][2] - pt[2] < 0 or pt[2] - box[0][2] < 0: return False
        return True

    def scan_scene(self, all_ids, scene, return_features = True):
        gt_seg_dir = DATA_PATH + "/original_2d_gt_seg_70"

        if not os.path.exists(os.path.join(gt_seg_dir, scene)):
            os.mkdir(os.path.join(gt_seg_dir, scene))

        grid_points = np.load(os.path.join(config.SAMPLE_DIR, self.scene, "grid_points.npy"))


        # grid_points = crop_points(grid_points, [room_bbox])
        quats = [self.degree2quat(*x) for x in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]]
        # quats = [self.degree2quat(*x) for x in [(0, 0), (90, 0), (180, 0), (270, 0)]]

        points = []
        i = 0

        all_instance_feature_dict = defaultdict(list)
        all_instance_feature_dict_final = dict()

        # if len(grid_points) == 0:
        #     print("No grid points found")

        for x in [x for x in grid_points if self.pathfinder.is_navigable(x)]:    
            new_state = habitat_sim.AgentState(x, [0, 0, 0, 1])
            self.agents[0].set_state(new_state) # set position

            # Scan in sphere
            obs = [self.parse_visual_observation(self.get_sensor_observations()),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left"))]
            
            # self.agents[0].set_state(new_state)  # reset
            # obs.append(self.parse_visual_observation(self.step("look_up")))
            # self.agents[0].set_state(new_state)  # reset
            # obs.append(self.parse_visual_observation(self.step("look_down")))
            self.get_per_instance(i, scene, obs, all_ids)

            i += 4

            # convert to points
            coordinates = []
            valid_masks = []
            for o, q in zip(obs, quats):
                p, valid_mask = self.reconstructor.depth_map2points(o[:, :, 3], q, x)
                coordinates.append(p)
                valid_masks.append(valid_mask)
            coordinates = np.concatenate(coordinates, axis=0)
            valid_idx = np.where(np.concatenate(valid_masks, axis=0))[0]

            # Concat all
            obs = np.stack(obs, axis=0).reshape(-1, 5)
            _points = np.concatenate([coordinates, obs[:, :3], obs[:, 4:]], axis=1).astype(np.float16) # xzy, rgb, semantic
            _points = _points[valid_idx]
            points.append(_points)
        

        if not len(points): return np.zeros((1,1))

        print ("%d vertices inside the room"%len(points))
        points = np.concatenate(points, axis=0)
        print ("%d points inside the room"%points.shape[0])
        if points.shape[0] == 0: return np.zeros((1,1))

        idx = self.reconstructor.downsample_index(points[:, :3])
        points = points[idx, :]
        print ("%d points after sampling"%points.shape[0])
        
        return points

    def get_per_instance(self, i, room, obs, all_ids):
        instance_feature_dict = dict()
        gt_seg_dir = DATA_PATH + "/original_2d_gt_seg_70"

        for (j, frame) in enumerate(obs):
            image = frame[..., :3].astype(np.uint8)  
            depth = frame[..., 3].astype(np.uint8)

            pil_image = Image.fromarray(image)

            try:
                os.mkdir(f"{gt_seg_dir}/{room}")
            except:
                # print("path not valid/already exists")
                pass
            # assert os.path.exists(gt_seg_dir)

            np.save(f"{gt_seg_dir}/{room}/depth_{i+j}.npy", depth)
            pil_image.save(f"{gt_seg_dir}/{room}/image_{i+j}.jpg")
            all_semantics = frame[..., 4]
            np.save(f"{gt_seg_dir}/{room}/semantic_{i+j}.npy", all_semantics)
            # print(all_semantics)
            semantics = np.unique(all_semantics).astype(int)

            for semantic in semantics:
                if not (semantic in all_ids or str(semantic) in all_ids or semantic >= 10000): continue
                
                indices = np.where(all_semantics == semantic)
                if np.min(indices[0]) == 0 or np.min(indices[1]) == 0 or np.max(indices[1]) == 719 or np.max(indices[0]) == 719: continue
                if indices[0].shape[0] < 100: continue 
                
                ymin, ymax, xmin, xmax = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
                image_copy = copy.deepcopy(image)
                image_copy[all_semantics != semantic] = 255
                pil_image = Image.fromarray(image_copy)
                cropped_image = pil_image.crop((xmin-1, ymin-1, xmax+1, ymax+1))

                # if the area of the cropped image is smaller than 2500 pixels, skip
                # if cropped_image.size[0] * cropped_image.size[1] >= 2500:
                #     continue
                pil_image.save(f"{gt_seg_dir}/{room}/{semantic}_{i+j}.jpg")
                cropped_image.save(f"{gt_seg_dir}/{room}/{semantic}_{i+j}_cropped.jpg")

    # Add features
    def parse_visual_observation(self, obs):
        # rgb = obs["rgba"]
        # semantic = obs["semantic"]
        # depth = obs["depth"]
        # print(semantic)
        # display_sample(rgb, semantic, depth)
        # input()
        rgb = cv2.cvtColor(obs["rgba"], cv2.COLOR_RGBA2RGB)
        frame = np.concatenate([rgb, obs["depth"][:, :, np.newaxis], obs["semantic"][:, :, np.newaxis]], axis=-1)

        return frame

    def get_semantic_labels(self):
        id2cate = dict()
        if self.new_objs is not None:
            for i in self.new_objs:
                id2cate[i["semantic_id"]] = i["cate"]
        with open(os.path.join(config.HM3D_DIR, _scene, f"{_scene.split('-')[1]}.semantic.txt"), "r") as f:
            a = f.readlines()
        for i in a[1:]:
            i = i.strip()
            if len(i):
                _id = int(i.split(",")[0])
                _cate = i.split(",")[2].strip('"')
                id2cate[_id] = _cate
        return id2cate

    @staticmethod
    def degree2quat(z=0, x=0):
        assert (z * x) == 0
        if z:
            half_radians = np.deg2rad(z) / 2.0
            around_z_axis = [0, np.sin(half_radians), 0, np.cos(half_radians)]  # anticlockwise
            return around_z_axis
        elif x:
            half_radians = np.deg2rad(x) / 2.0
            around_x_axis = [np.sin(half_radians), 0, 0, np.cos(half_radians)]  # up is positive direction
            return around_x_axis
        else:
            return [0, 0, 0, 1]


if __name__ == "__main__":
    # bbox_dir = config.HM3D_BBOX_DIR
    # gt_seg_dir_old = config.DATA_DIR + "/original_2d_gt_seg"

    # saved_rooms = {
    #     dir_name.split("_")[0]: dir_name.split("_")[1]
    #     for dir_name in os.listdir(gt_seg_dir_old) if os.path.isdir(os.path.join(gt_seg_dir_old, dir_name))
    # }
    # with open("../saved_rooms.json", "r") as f:
    #     saved_rooms = json.load(f)

    hm3d = HM3D()

    # bbox_files = os.listdir(bbox_dir)
    # bbox_files = [x for x in bbox_files if x.endswith(".json") and int(x.split("-")[0]) >= 800]
    # bbox_files = sorted(bbox_files, key=lambda x: int(x.split("-")[0]))

    # room_bbox_dir = config.ROOM_BBOX_DIR
    # room_bbox_files = os.listdir(room_bbox_dir)

    # room_not_found_count = 0
    # 1 - 17
    # 31 - 32

    # not working?
    # 56 - 65 ?

    for scene_id in tqdm(train_scene_ids[-20:]):

        scene = scene_id
        scene_dir = os.path.join(config.HM3D_DIR, scene)
        semantic_file_path = os.path.join(scene_dir, f"{scene.split('-')[1]}.semantic.txt")
        with open(semantic_file_path, "r") as f:
            semantics = f.readlines()

        id2cate = dict()
        for i in semantics[1:]:
            i = i.strip()
            if len(i):
                _id = int(i.split(",")[0])
                _cate = i.split(",")[2].strip('"')
                # print(_id, _cate)
                id2cate[_id] = _cate

        final_bboxes = []
        original_bboxes = []
        all_ids = []
        k = 1
        for k, v in id2cate.items():
            if not v in ['floor', 'wall', 'ceiling']:
                all_ids.append(k)
                # final_bboxes.append(bbox)
                # original_bboxes.append(bbox)

        sampler = GridSampler(scene, audio=False)
        # print ("successfully building sampler")

        points = sampler.scan_scene(all_ids, scene, return_features=True)
        sampler.close()

        # if points.shape[0] == 1: 
        #     continue

        # np.save(os.path.join(config.SAMPLE_DIR, scene, f"{bbox_file_copy}.npy"), points)
    
    # folder_list = [folder for folder in  os.listdir(config.HM3D_DIR) if folder.startswith("00") and len(os.listdir(os.path.join(config.HM3D_DIR, folder))) == 4]

    # for i in os.listdir(config.HM3D_DIR):
    #     if os.path.isdir(os.path.join(config.HM3D_DIR, i)):
    #         sampler = GridSampler(i)
    #         print ("successfully building sampler")

    #         # points = sampler.scan_scene(room_bbox, all_ids, bbox_file_copy, return_features=True)
    #         sampler.close()

    #         # if points.shape[0] == 1: continue

    #         # np.save(os.path.join(config.SAMPLE_DIR, scene, f"{bbox_file_copy}.npy"), points)

            
