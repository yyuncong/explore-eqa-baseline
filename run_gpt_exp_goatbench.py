import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
from collections import defaultdict
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import glob
import time
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_to_angle_axis, quat_from_coeffs
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion
)
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf import TSDFPlanner
from loader import *
from gpt_utils_goatbench import get_confidence, get_directions, get_global_value



def main(cfg, start_ratio=0.0, end_ratio=1.0):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)

    # split the test data by scene
    # scene_data_list = scene_data_list[int(start_ratio * num_scene):int(end_ratio * num_scene)]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), 'r') as f:
            num_episode += int(len(json.load(f)['episodes']) * (end_ratio - start_ratio))
    logging.info(f"Total number of episodes: {num_episode}")
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path_train + '/train') + os.listdir(cfg.scene_data_path_val + '/val')

    # load result stats
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl")):
        success_by_distance = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        success_by_distance = {}  # subtask id -> success
    if os.path.exists(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl")):
        spl_by_distance = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        spl_by_distance = {}  # subtask id -> spl
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl")):
        success_by_task = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        # success_by_task = {}  # task type -> success
        success_by_task = defaultdict(list)
    if os.path.exists(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl")):
        spl_by_task = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        # spl_by_task = {}  # task type -> spl
        spl_by_task = defaultdict(list)
    assert len(success_by_distance) == len(spl_by_distance), f"{len(success_by_distance)} != {len(spl_by_distance)}"
    assert sum([len(task_res) for task_res in success_by_task.values()]) == sum(
        [len(task_res) for task_res in spl_by_task.values()]) == len(
        success_by_distance), f"{sum([len(task_res) for task_res in success_by_task.values()])} != {sum([len(task_res) for task_res in spl_by_task.values()])} != {len(success_by_distance)}"

    question_idx = -1
    for scene_data_file in scene_data_list:
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(open(os.path.join(cfg.test_data_dir, scene_data_file), "r"))
        total_episodes = len(scene_data["episodes"])

        navigation_goals = scene_data["goals"]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data["episodes"][int(start_ratio * total_episodes):int(end_ratio * total_episodes)]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            # filter the task according to goatbench
            filtered_tasks = []
            for goal in episode["tasks"]:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [
                    x
                    for x in navigation_goals.values()
                    if x[0]["object_category"] == goal_category
                ]

                if goal_type == "description":
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                        filtered_tasks.append(goal)
                else:
                    filtered_tasks.append(goal)

            all_subtask_goals = []
            all_subtask_goal_types = []
            for goal in filtered_tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                all_subtask_goal_types.append(goal_type)

                dset_same_cat_goals = [
                    x
                    for x in navigation_goals.values()
                    if x[0]["object_category"] == goal_category
                ]
                children_categories = dset_same_cat_goals[0][0][
                    "children_object_categories"
                ]
                for child_category in children_categories:
                    goal_key = f"{scene_name}.basis.glb_{child_category}"
                    if goal_key not in navigation_goals:
                        print(f"!!! {goal_key} not in navigation_goals")
                        continue
                    print(f"!!! {goal_key} added")
                    dset_same_cat_goals[0].extend(navigation_goals[goal_key])

                assert (
                        len(dset_same_cat_goals) == 1
                ), f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    all_subtask_goals.append(dset_same_cat_goals[0])
                else:
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    all_subtask_goals.append(goal_inst)

            # check whether this episode has been processed
            finished_subtask_ids = list(success_by_distance.keys())
            finished_episode_subtask = [subtask_id for subtask_id in finished_subtask_ids if subtask_id.startswith(f"{scene_id}_{episode_id}_")]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            init_pts = episode["start_position"]
            init_quat = quat_from_coeffs(episode["start_rotation"])

            pts = np.asarray(init_pts)
            angle, axis = quat_to_angle_axis(init_quat)
            angle = angle * axis[1] / np.abs(axis[1])
            rotation = get_quaternion(angle, 0)

            for subtask_idx, (goal_type, subtask_goal) in enumerate(zip(all_subtask_goal_types, all_subtask_goals)):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"

                # load scene
                split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
                if split == "train":
                    scene_mesh_path = os.path.join(cfg.scene_data_path_train, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
                    navmesh_path = os.path.join(cfg.scene_data_path_train, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
                else:
                    scene_mesh_path = os.path.join(cfg.scene_data_path_val, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
                    navmesh_path = os.path.join(cfg.scene_data_path_val, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")

                # Set up scene in Habitat
                try:
                    simulator.close()
                except:
                    pass
                sim_settings = {
                    "scene": scene_mesh_path,
                    "default_agent": 0,
                    "sensor_height": cfg.camera_height,
                    "width": img_width,
                    "height": img_height,
                    "hfov": cfg.hfov,
                    "scene_dataset_config_file": cfg.scene_dataset_config_path,
                }
                sim_cfg = make_simple_cfg(sim_settings)
                simulator = habitat_sim.Simulator(sim_cfg)
                pathfinder = simulator.pathfinder
                pathfinder.seed(cfg.seed)
                pathfinder.load_nav_mesh(navmesh_path)
                agent = simulator.initialize_agent(sim_settings["default_agent"])
                agent_state = habitat_sim.AgentState()

                # determine the navigation goals
                goal_category = subtask_goal[0]["object_category"]
                goal_obj_ids = [x["object_id"] for x in subtask_goal]
                goal_obj_ids = [int(x.split('_')[-1]) for x in goal_obj_ids]
                if goal_type != "object":
                    assert len(goal_obj_ids) == 1, f"{len(goal_obj_ids)} != 1"

                goal_positions = [x["position"] for x in subtask_goal]

                viewpoints = [
                    view_point["agent_state"]["position"] for goal in subtask_goal for view_point in goal["view_points"]
                ]
                # get the shortest distance from current position to the viewpoints
                all_distances = []
                for viewpoint in viewpoints:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = pts
                    path.requested_end = viewpoint
                    found_path = pathfinder.find_path(path)
                    if not found_path:
                        all_distances.append(np.inf)
                    else:
                        all_distances.append(path.geodesic_distance)
                start_end_subtask_distance = min(all_distances)

                logging.info(f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}")

                subtask_object_observe_dir = os.path.join(str(cfg.output_dir), f"{subtask_id}", 'object_observations')
                if os.path.exists(subtask_object_observe_dir):
                    os.system(f"rm -r {subtask_object_observe_dir}")
                os.makedirs(subtask_object_observe_dir, exist_ok=False)

                # Prepare metadata for the subtask
                # format question according to the goal type
                if goal_type == "object":
                    question = f"Where is the {goal_category}?"
                    ref_image = None
                elif goal_type == "description":
                    question = f"Could you find the object described as \'{subtask_goal[0]['lang_desc']}\'?"
                    ref_image = None
                else:  # goal_type == "image"
                    view_pos_dict = random.choice(subtask_goal[0]["view_points"])['agent_state']

                    agent_state.position = view_pos_dict["position"]
                    agent_state.rotation = view_pos_dict["rotation"]
                    agent.set_state(agent_state)
                    obs = simulator.get_sensor_observations()

                    plt.imsave(os.path.join(str(cfg.output_dir), f"{subtask_id}", "image_goal.png"), obs["color_sensor"])

                    question = f"Could you find the object captured in the following image?"
                    ref_image = Image.fromarray(obs["color_sensor"], mode="RGBA").convert("RGB")

                rotation = get_quaternion(angle, camera_tilt)
                pts_normal = pos_habitat_to_normal(pts)
                floor_height = pts_normal[-1]
                tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
                num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
                logging.info(
                    f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
                )

                # Initialize TSDF
                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height_offset=0,
                    pts_init=pos_habitat_to_normal(pts),
                    init_clearance=cfg.init_clearance * 2,
                )

                # Set data dir for this question - set initial data to be saved
                episode_data_dir = os.path.join(cfg.output_dir, str(subtask_id))
                print('output_dir:',cfg.output_dir)
                os.makedirs(episode_data_dir, exist_ok=True)

                # build up the directory for output
                (episode_data_dir,
                 episode_observations_dir,
                 episode_object_observe_dir,
                 episode_frontier_dir,
                 episode_semantic_dir) = build_output_dir(cfg.output_dir, subtask_id)

                # Run steps
                pts_pixs = np.empty((0, 2))  # for plotting path on the image
                path_length = 0
                early_stopped = False
                max_answer = {
                    'relevancy':0,
                    'position':None,
                    'rotation':None
                }
                for cnt_step in range(num_step):
                    logging.info(f"\n== step: {cnt_step}")

                    # Save step info and set current pose
                    step_name = f"step_{cnt_step}"
                    logging.info(f"Current pts: {pts}")
                    agent_state.position = pts
                    agent_state.rotation = rotation
                    agent.set_state(agent_state)
                    pts_normal = pos_habitat_to_normal(pts)

                    # Update camera info for TSDF
                    sensor = agent.get_state().sensor_states["depth_sensor"]
                    quaternion_0 = sensor.rotation
                    translation_0 = sensor.position
                    cam_pose = np.eye(4)
                    cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                    cam_pose[:3, 3] = translation_0
                    cam_pose_normal = pose_habitat_to_normal(cam_pose)
                    cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                    # Get observation at current pose - skip black image, meaning robot is outside the floor
                    # get current observation
                    obs = simulator.get_sensor_observations()
                    rgb = obs["color_sensor"]
                    depth = obs["depth_sensor"]
                    if cfg.save_obs:
                        plt.imsave(
                            os.path.join(episode_observations_dir, "{}.png".format(cnt_step)), rgb
                        )

                    num_black_pixels = np.sum(
                        np.sum(rgb, axis=-1) == 0
                    )  # sum over channel first
                    # black pixel means unuseful information
                    # when there is at least some useful information, explore the scene
                    if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:

                        # TSDF fusion
                        tsdf_planner.integrate(
                            color_im=rgb,
                            depth_im=depth,
                            cam_intr=cam_intr,
                            cam_pose=cam_pose_tsdf,
                            obs_weight=1.0,
                            margin_h=int(cfg.margin_h_ratio * img_height),
                            margin_w=int(cfg.margin_w_ratio * img_width),
                        )
                        # no need to predict choices in open-ended questions
                        # Get VLM relevancy
                        rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                        #prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
                        # logging.info(f"Prompt Rel: {prompt_rel}")
                        #smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                        smx_vlm_rel = get_confidence(question=question, ref_image=ref_image, obs_image=rgb_im)
                        logging.info(f"Rel - Prob: {smx_vlm_rel}")

                        # Get frontier candidates
                        prompt_points_pix = []
                        if cfg.use_active:
                            prompt_points_pix, fig = (
                                tsdf_planner.find_prompt_points_within_view(
                                    pts_normal,
                                    img_width,
                                    img_height,
                                    cam_intr,
                                    cam_pose_tsdf,
                                    **cfg.visual_prompt,
                                )
                            )
                            fig.tight_layout()
                            plt.savefig(
                                os.path.join(
                                    episode_frontier_dir, "{}_prompt_points.png".format(cnt_step)
                                )
                            )
                            plt.close()

                        # Visual prompting
                        draw_letters = ["A", "B", "C", "D"]  # always four
                        fnt = ImageFont.truetype(
                            "data/Open_Sans/static/OpenSans-Regular.ttf",
                            30,
                        )
                        actual_num_prompt_points = len(prompt_points_pix)
                        if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                            rgb_im_draw = rgb_im.copy()
                            draw = ImageDraw.Draw(rgb_im_draw)
                            for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                                # draw "A", "B", "C", "D in each frontier point
                                draw.ellipse(
                                    (
                                        point_pix[0] - cfg.visual_prompt.circle_radius,
                                        point_pix[1] - cfg.visual_prompt.circle_radius,
                                        point_pix[0] + cfg.visual_prompt.circle_radius,
                                        point_pix[1] + cfg.visual_prompt.circle_radius,
                                    ),
                                    fill=(200, 200, 200, 255),
                                    outline=(0, 0, 0, 255),
                                    width=3,
                                )
                                draw.text(
                                    tuple(point_pix.astype(int).tolist()),
                                    draw_letters[prompt_point_ind],
                                    font=fnt,
                                    fill=(0, 0, 0, 255),
                                    anchor="mm",
                                    font_size=12,
                                )
                            rgb_im_draw.save(
                                os.path.join(episode_frontier_dir, f"{cnt_step}_draw.png")
                            )

                            # get VLM reasoning for exploring
                            if cfg.use_lsv:
                                lsv = get_directions(question, ref_image, rgb_im_draw, draw_letters[:actual_num_prompt_points])
                                lsv *= actual_num_prompt_points / 3
                            else:
                                lsv = (
                                    np.ones(actual_num_prompt_points) / actual_num_prompt_points
                                )

                            # base - use image without label
                            if cfg.use_gsv:
                                #prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                                # logging.info(f"Prompt Exp base: {prompt_gsv}")
                                #gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
                                gsv = get_global_value(question, ref_image, rgb_im)[0]
                                gsv = (
                                    np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                                )  # scale before combined with lsv
                            else:
                                gsv = 1
                            sv = lsv * gsv
                            logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                            # Integrate semantics only if there is any prompted point
                            tsdf_planner.integrate_sem(
                                sem_pix=sv,
                                radius=1.0,
                                obs_weight=1.0,
                            )  # voxel locations already saved in tsdf class

                    else:
                        logging.info("Skipping black image!")

                    # track the maxrelevancy and record corresponding positions and rotations
                    if smx_vlm_rel[0] > max_answer['relevancy']:
                        max_answer['relevancy'] = smx_vlm_rel[0]
                        max_answer['position'] = pts
                        max_answer['angle'] = angle
                    # check early stop condition
                    if cfg.early_stop:
                        # the
                        if smx_vlm_rel[0] > cfg.confidence_threshold:
                            logging.info("Early stop due to high confidence!")
                            logging.info("Current relevancy: {}".format(smx_vlm_rel[0]))
                            early_stopped = True
                            # continue to solve the next question
                            break
                    # Determine next point
                    if cnt_step < num_step - 1:
                        pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
                            pts=pts_normal,
                            angle=angle,
                            flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                            **cfg.planner,
                        )
                        pts_pixs = np.vstack((pts_pixs, pts_pix))
                        pts_normal = np.append(pts_normal, floor_height)
                        # record the previous positions and next positions
                        prev_pts, pts = pts, pos_normal_to_habitat(pts_normal)
                        # add path length
                        path_length += float(np.linalg.norm(pts - prev_pts))

                        # Add path to ax5, with colormap to indicate order
                        ax5 = fig.axes[4]
                        ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                        ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                        fig.tight_layout()
                        plt.savefig(
                            os.path.join(episode_semantic_dir, "{}_map.png".format(cnt_step + 1))
                        )
                        plt.close()
                    rotation = get_quaternion(angle, camera_tilt)
                #--------------------------
                # calculate the distance to the nearest view point
                all_distances = []
                for viewpoint in viewpoints:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = pts
                    path.requested_end = viewpoint
                    found_path = pathfinder.find_path(path)
                    if not found_path:
                        all_distances.append(np.inf)
                    else:
                        all_distances.append(path.geodesic_distance)
                agent_subtask_distance = min(all_distances)
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance[subtask_id] = 1.0
                    logging.info(f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!")
                else:
                    success_by_distance[subtask_id] = 0.0
                    logging.info(f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!")

                spl_by_distance[subtask_id] = (success_by_distance[subtask_id] * start_end_subtask_distance /
                                               max(start_end_subtask_distance, path_length))

                success_by_task[goal_type].append(success_by_distance[subtask_id])
                spl_by_task[goal_type].append(spl_by_distance[subtask_id])

                logging.info(f"Subtask {subtask_id} finished with {cnt_step} steps, {path_length} length")
                logging.info(f"spl by distance: {spl_by_distance[subtask_id]}")

                logging.info(f"Success rate by distance: {100 * np.mean(np.asarray(list(success_by_distance.values()))):.2f}")
                logging.info(f"SPL by distance: {100 * np.mean(np.asarray(list(spl_by_distance.values()))):.2f}")

                for task_name, success_list in success_by_task.items():
                    logging.info(f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}")
                for task_name, spl_list in spl_by_task.items():
                    logging.info(f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}")

                assert len(success_by_distance) == len(spl_by_distance), f"{len(success_by_distance)} != {len(spl_by_distance)}"
                assert sum([len(task_res) for task_res in success_by_task.values()]) == sum(
                    [len(task_res) for task_res in spl_by_task.values()]) == len(
                    success_by_distance), f"{sum([len(task_res) for task_res in success_by_task.values()])} != {sum([len(task_res) for task_res in spl_by_task.values()])} != {len(success_by_distance)}"

                with open(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                    pickle.dump(success_by_distance, f)
                with open(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                    pickle.dump(spl_by_distance, f)
                with open(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                    pickle.dump(success_by_task, f)
                with open(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                    pickle.dump(spl_by_task, f)
                
                if not cfg.save_visualization:
                    # after finishing the subtask, clear up the saved data
                    os.system(f"rm -r {episode_data_dir}")

    logging.info(f"All scene finish")

    # aggregate the results into a single file
    filenames_to_merge = ['success_by_distance', 'spl_by_distance']
    for filename in filenames_to_merge:
        all_results = {}
        all_results_paths = glob.glob(os.path.join(str(cfg.output_dir), f"{filename}_*.pkl"))
        for results_path in all_results_paths:
            with open(results_path, "rb") as f:
                all_results.update(pickle.load(f))
        logging.info(f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}")
        with open(os.path.join(str(cfg.output_dir), f"{filename}.pkl"), "wb") as f:
            pickle.dump(all_results, f)
    filenames_to_merge = ['success_by_task', 'spl_by_task']
    for filename in filenames_to_merge:
        all_results = {}
        all_results_paths = glob.glob(os.path.join(str(cfg.output_dir), f"{filename}_*.pkl"))
        for results_path in all_results_paths:
            with open(results_path, "rb") as f:
                separate_stat = pickle.load(f)
                for task_name, task_res in separate_stat.items():
                    if task_name not in all_results:
                        all_results[task_name] = []
                    all_results[task_name] += task_res
        for task_name, task_res in all_results.items():
            logging.info(f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}")
        with open(os.path.join(str(cfg.output_dir), f"{filename}.pkl"), "wb") as f:
            pickle.dump(all_results, f)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log")

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")


    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


    # Set up the logging configuration
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio)
