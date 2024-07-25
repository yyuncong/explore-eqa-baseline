# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import habitat_sim
import numpy as np
import tqdm
from config import make_cfg
from PIL import Image
import json
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hm3d-root",
        type=Path,
        default="/work/pi_chuangg_umass_edu/yuncong/data/scene_datasets/hm3d/val",
        help="path to hm3d scene data (default: data/scene_datasets/hm3d/val)",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default="data/frames/hm3d-v0",
        help="input path (default: data/frames/hm3d-v0)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="/work/pi_chuangg_umass_edu/yuncong/results/visualize_baseline",
        help="output path (default: data/frames/hm3d-v0)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="only extract rgb frames (default: false)",
    )
    parser.add_argument(
        "--consider_question",
        type = str,
        default = "data/generated_questions-eval.json",
    )
    args = parser.parse_args()
    return args


def get_config(
    scene_id: str, sensor_position: float
) -> habitat_sim.simulator.Configuration:
    settings = {
        "scene_id": scene_id,
        "sensor_hfov": 90,
        "sensor_width": 1920,
        "sensor_height": 1080,
        "sensor_position": sensor_position,  # height only
    }
    return make_cfg(settings)


def load_sim(path: Path, scene_id: str) -> habitat_sim.Simulator:
    data = pickle.load(path.open("rb"))
    #scene_id = data["scene_id"]
    scene_id = f"data/scene_datasets/hm3d/val/{scene_id}"
    config_file = f"{scene_id.split('-')[1]}.basis.glb"
    # adjust the scene_id here
    parent_folder = "/work/pi_chuangg_umass_edu/yuncong/"
    scene_id = parent_folder + scene_id + "/" + config_file
    agent_state = data["agent_state"]
    sensor_position = (
        agent_state.sensor_states["rgb"].position[1] - agent_state.position[1]
    )
    # this means cfg should be a valid habitat_sim.simulator.Configuration
    # we need to add the scene config here
    cfg = get_config(scene_id=scene_id, sensor_position=sensor_position)
    return habitat_sim.Simulator(cfg)


def save_intrinsics(path: Path) -> None:
    data = pickle.load(path.open("rb"))
    height, width = data["resolution"]
    hfov = np.deg2rad(data["hfov"])
    vfov = hfov * height / width
    K = np.array(
        [
            [width / np.tan(hfov / 2.0) / 2.0, 0.0, width / 2, 0.0],
            [0.0, height / np.tan(vfov / 2.0) / 2.0, height / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    output_path = path.parent / "intrinsic_color.txt"
    np.savetxt(output_path, K, fmt="%.6f")
    output_path = path.parent / "intrinsic_depth.txt"
    np.savetxt(output_path, K, fmt="%.6f")


def save_pose(path: Path, sim: habitat_sim.Simulator) -> None:
    camera_pose_path = str(path).replace(".pkl", ".txt")
    camera_pose = sim._sensors["rgb"]._sensor_object.node.absolute_transformation()
    np.savetxt(camera_pose_path, camera_pose, fmt="%.6f")


def save_depth(path: Path, sim: habitat_sim.Simulator) -> None:
    obs = sim.get_sensor_observations()
    depth_path = str(path).replace(".pkl", "-depth.png")
    depth = (obs["depth"] / 10 * 65535).astype(np.uint16)
    Image.fromarray(depth).save(depth_path)


def save_color(path: Path, sim: habitat_sim.Simulator) -> None:
    obs = sim.get_sensor_observations()
    parent_dir = path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    print("parent dir is", parent_dir)
    #exit(0)
    rgb_path = str(path).replace(".pkl", "-rgb.png")
    Image.fromarray(obs["rgb"]).convert("RGB").save(rgb_path)


def extract_frames(in_folder: Path, scene_id: str, args: argparse.Namespace) -> None:
    print("Extracting frames to: {}".format(in_folder))
    files = sorted(in_folder.glob("*.pkl"))
    files = files[0:5]
    print(files)
    sim = load_sim(files[0],scene_id)

    print("Processing {} agent positions...".format(len(files)))
    for idx, path in enumerate(files):
        # set agent state
        #print(path)
        #exit(0)
        data = pickle.load(path.open("rb"))
        agent = sim.get_agent(0)
        agent.set_state(data["agent_state"])

        # save data
        outpath = args.output_directory / path
        #print(outpath)
        #exit(0)
        if not args.rgb_only:
            if idx == 0:
                save_intrinsics(outpath)
            save_pose(outpath, sim)
            save_depth(outpath, sim)
        save_color(outpath, sim)

    sim.close()
    print("Extracting frames to: {} done!".format(args.output_directory))

def gather_test_scene(path):
    with open(path,'r') as f:
        questions = json.load(f)
    scenes = {}
    for q in questions:
        scenes[q['episode_history'].split('-')[1]] = q['episode_history']
    return scenes
    
def select_files(files: List[Path], scenes) -> List[Path]:
    #scenes = set(s.split("-")[-1] for s in scenes)
    #print(scenes)
    selected_files = []
    for f in files:
        fname = f.stem.split("-")
        fname = fname[-1]
        if fname in scenes.keys():
            selected_files.append((f,str(scenes[fname])))
    return selected_files

def main(args):
    # we should skip all files not in generated_questions-eval.json to ensure stability
    # gather feasible scenes
    scenes = gather_test_scene(args.consider_question)
    folders = sorted(args.input_directory.glob("*"))
    folders = select_files(folders, scenes)
    #print(folders)
    #exit(0)
    for folder, scene_id in tqdm.tqdm(folders):
        extract_frames(folder, scene_id, args)


if __name__ == "__main__":
    main(parse_args())