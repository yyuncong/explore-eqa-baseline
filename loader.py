import json
import os
import random
from easydict import EasyDict
import logging
import matplotlib.pyplot as plt
import shutil


def load_info(question_data_path):
    questions = os.listdir(question_data_path)
    scenes = set()
    filtered_questions = []
    for qid in questions:
        if int(qid.split('-')[0]) < 800:
            continue
        sid = qid.split('_')[0]
        scenes.add(sid)
        filtered_questions.append(qid)
    scenes = sorted(list(scenes))
    random.shuffle(scenes)
    return scenes, filtered_questions

def load_question_data(question_path,question_id):
    with open(os.path.join(question_path,question_id,'metadata.json'), 'r') as f:
        metadata = json.load(f)
    scene = metadata['scene']
    question = metadata['question']
    answer = metadata['answer']
    init_pts = metadata['init_pts']
    init_angle = metadata['init_angle']
    return scene, question, answer, init_pts, init_angle

def load_scene(cfg,scene_id):
    scene_path = cfg.scene_data_path_train if int(scene_id.split("-")[0]) < 800 else cfg.scene_data_path_val
    scene_features_path = cfg.scene_features_path_train if int(scene_id.split("-")[0]) < 800 else cfg.scene_features_path_val
    scene_mesh_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".basis.glb")
    navmesh_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
    semantic_texture_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".semantic.glb")
    scene_semantic_annotation_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".semantic.txt")
    bbox_data_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
    
    if not os.path.exists(scene_mesh_path) or not os.path.exists(navmesh_path) or not os.path.exists(semantic_texture_path) or not os.path.exists(scene_semantic_annotation_path):
        logging.info(f"Scene {scene_id} not found, skip")
        return None
    if not os.path.exists(bbox_data_path):
        logging.info(f"Scene {scene_id} bbox data not found, skip")
        return None
    if not os.path.exists(scene_features_path):
        logging.info(f"Scene {scene_id} features not found, skip")
        return None
    
    return EasyDict(
        scene_path = scene_path,
        scene_features_path = scene_features_path,
        scene_mesh_path = scene_mesh_path,
        navmesh_path = navmesh_path,
        semantic_texture_path = semantic_texture_path,
        scene_semantic_annotation_path = scene_semantic_annotation_path,
        bbox_data_path = bbox_data_path
    )
    

def build_output_dir(output_dir, question_id):
    episode_data_dir = os.path.join(output_dir, question_id)
    episode_observations_dir = os.path.join(episode_data_dir, 'observations')
    episode_object_observe_dir = os.path.join(episode_data_dir, 'object_observations')
    episode_frontier_dir = os.path.join(episode_data_dir, 'frontier')
    episode_semantic_dir = os.path.join(episode_data_dir, 'semantic')
    os.makedirs(episode_data_dir, exist_ok=True)
    os.makedirs(episode_observations_dir, exist_ok=True)
    os.makedirs(episode_object_observe_dir, exist_ok=True)
    os.makedirs(episode_frontier_dir, exist_ok=True)
    os.makedirs(episode_semantic_dir, exist_ok=True)
    return episode_data_dir, episode_observations_dir, episode_object_observe_dir, episode_frontier_dir, episode_semantic_dir

def store_observations(observe_dir, observations):
    for view_idx, observation in enumerate(observations):
        plt.imsave(
            os.path.join(save_dir, f"{view_idx}.png"), rgb
        )

# copy the last k observations from the observation directory to the object observation directory 
# as the observation for the object
def extract_last_k_observations(src_dir, dst_dir, idx, k = 5):
    files = os.listdir(src_dir)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    start = max(idx-k,0)
    for i in range(start,idx):
        shutil.copy2(os.path.join(src_dir,files[i]),os.path.join(dst_dir,f'{i-start}.png'))