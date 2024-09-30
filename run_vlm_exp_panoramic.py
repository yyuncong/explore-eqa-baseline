"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["DISABLE_IPV6"] = "True"
#os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_to_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from loader import *
hf_token = os.environ["HF_TOKEN"]
from huggingface_hub import login
import json

# turn the agent around and collect the observation for answering the question
def take_round_observation(agent,simulator,camera_tilt,pts,angle,num_obs,save_dir):
    
    angle_increment = 2*np.pi/num_obs
    all_angles = [angle + angle_increment*(i - num_obs//2) for i in range(num_obs)]
    # let the main viewing angle be the last to avoid overwriting
    main_angle = all_angles.pop(num_obs//2)
    all_angles.append(main_angle)
    
    # set agent state
    for view_idx, ang in enumerate(all_angles):
        agent_state_obs = habitat_sim.AgentState()
        agent_state_obs.position = pts
        agent_state_obs.rotation = get_quaternion(ang, camera_tilt)
        agent.set_state(agent_state_obs)
        obs = simulator.get_sensor_observations()
        rgb = obs["color_sensor"]
        plt.imsave(
            os.path.join(save_dir, f"round_{view_idx}.png"), rgb
        )
    

def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    # Load dataset
    '''
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    logging.info(f"Loaded {len(questions_data)} questions.")
    '''
    #scenes, question_ids = load_info(cfg.question_data_path)
    scenes, questions = load_info_eval(cfg.question_data_path)
    # Load VLM
    vlm = VLM(cfg.vlm)
    # use a placeholder for now
    #vlm = None

    # Run all questions
    cnt_data = 0
    results_all = {}
    step_length = {}
    success = 0
    #for question_ind in tqdm(range(len(questions_data))):
    for question_idx in tqdm(range(len(questions))):

        # Extract question
        '''
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        '''
        #scene, question, answer, init_pts, init_angle = load_question_data(cfg.question_data_path,question_id)
        scene, question_id, question, answer, init_pts, init_angle = load_question_eval(questions[question_idx])
        logging.info(f"current scene: {scene}")
        logging.info(f"current question: {question}")
        #logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")
        scene_path_dict = load_scene(cfg,scene)

        # Re-format the question to follow LLaMA style
        '''
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        logging.info(f"Question:\n{question} \nAnswer: {answer}")
        '''
        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_id))
        print('output_dir:',cfg.output_dir)
        os.makedirs(episode_data_dir, exist_ok=True)
        #result = {"question_id": question_id}
        result = {}
        
        # build up the directory for output
        (episode_data_dir,
         episode_observations_dir,
         episode_object_observe_dir, 
         episode_frontier_dir,
         episode_semantic_dir) = build_output_dir(cfg.output_dir, question_id)

            

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        '''
        scene_mesh_dir = os.path.join(
            scene_path_dict.scene_mesh_path
        )
        navmesh_file = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        '''
        sim_settings = {
            "scene": scene_path_dict.scene_mesh_path,
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
        pathfinder.load_nav_mesh(scene_path_dict.navmesh_path)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        '''
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        '''
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
            # prepare panoramic views for frontier exploration
            if cnt_step == 0:
                angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_2
            else:
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_1 
            all_angles = [angle + angle_increment * (i - total_views // 2) for i in range(total_views)]
            # let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2)
            all_angles.append(main_angle)
            
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            # the position is fixed
            result[step_name] = {"pts": pts, "angle": all_angles}
            agent_state.position = pts
            pts_normal = pos_habitat_to_normal(pts)
            for view_idx, ang in enumerate(all_angles):
                # 1. get the observation (rgb, depth, ...)
                # case black pixel < ...
                # 2. tsdf_planner.integrate
                
                # case black pixel >= ...
                
                # 3. jump the case
                
                # as the final angle is the egocentric view we need, if view_idx = ... & black pixel < ...:
                # 4. get the VLM relevancy
                agent_state.rotation = get_quaternion(ang, camera_tilt)
                agent.set_state(agent_state)
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"]
                plt.imsave(
                    os.path.join(episode_observations_dir, f"{cnt_step}_{view_idx}.png"), rgb
                )

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
                # sketch for panoramic exploration
                if num_black_pixels >= cfg.black_pixel_ratio * img_width * img_height:
                    if view_idx == total_views - 1:
                        logging.info("Skipping black image!")
                        #result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                        result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])
                    continue

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
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
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
                        prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                        # logging.info(f"Prompt Exp: {prompt_text}")
                        lsv = vlm.get_loss(
                            rgb_im_draw,
                            prompt_lsv,
                            draw_letters[:actual_num_prompt_points],
                        )
                        lsv *= actual_num_prompt_points / 3
                    else:
                        lsv = (
                            np.ones(actual_num_prompt_points) / actual_num_prompt_points
                        )

                    # base - use image without label
                    if cfg.use_gsv:
                        prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                        # logging.info(f"Prompt Exp base: {prompt_gsv}")
                        gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
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

                # the main angle is the last angle, so we can use the final view to get the prediction
                # no need to predict choices in open-ended questions
                # Get VLM relevancy
                if view_idx == total_views - 1:
                    prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
                    # logging.info(f"Prompt Rel: {prompt_rel}")
                    smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                    logging.info(f"Rel - Prob: {smx_vlm_rel}")
                    result[step_name]["smx_vlm_rel"] = smx_vlm_rel
            
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
                    # also save the round observation
                    '''
                    take_round_observation(
                        agent,simulator,
                        camera_tilt,pts,angle,
                        cfg.object_obs,episode_object_observe_dir)
                    '''
                    extract_last_k_observations(
                        episode_observations_dir,episode_object_observe_dir,cnt_step,cfg.num_last_views
                    )
                    result['explore_path_length'] = path_length
                    early_stopped = True
                    success += 1
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
            '''
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
            '''
            rotation = get_quaternion(angle, camera_tilt)
        #--------------------------
        # Check if success using weighted prediction(no need consider candidates)
        # gather the predictions and relevancy within all steps, and get the max one as the answer
        # select the step with the highest relevancy as the answer in cases:
        # 1. do not use early stop
        # 2. use early stop but not early stopped (relevancy < confidence_margin)
        relevancy_all = [
            result[f"step_{step}"]["smx_vlm_rel"][0] 
            for step in range(num_step)
            if f"step_{step}" in result.keys()
        ]
        # the weighted prediction over for choices([pa*r,pb*r,pc*r,pd*r])
        # only keep option 2: use the max of the relevancy
        # get the most confident step, and use the prediction at that step
        max_relevancy_step = np.argmax(relevancy_all)
        relevancy_ord = np.flip(np.argsort(relevancy_all))
        if not early_stopped and cfg.use_best:
            logging.info(f"Use information in the step with the hightes relevancy {max_answer['relevancy']}")
            '''
            take_round_observation(
                agent,simulator,
                camera_tilt,max_answer['position'],max_answer['angle'],
                cfg.object_obs,episode_object_observe_dir)
            '''
            extract_last_k_observations(
                episode_observations_dir,episode_object_observe_dir,max_relevancy_step,cfg.num_last_views
            )
        result['explore_path_length'] = path_length
        step_length[question_id] = path_length

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Index: {question_idx}, Scene: {scene}")
        logging.info(f"Question:\n{question}\nAnswer: {answer}")
        logging.info(f"Max relevancy: {max_answer['relevancy']}")
        logging.info(
            f"Top 3 steps with highest relevancy with value: {relevancy_ord[:3]} {[relevancy_all[i] for i in relevancy_ord[:3]]}"
        )
        logging.info(f"path length: {path_length}")
        '''
        for rel_ind in range(3):
            logging.info(f"Prediction: {smx_vlm_all[relevancy_ord[rel_ind]]}")
        '''
        # Save data
        results_all[question_id] = result
        
        cnt_data += 1
        logging.info(f"Success rate: {success/cnt_data}")
        # dummy setting for function test
        '''
        if cnt_data % cfg.save_freq == 0:
            with open(
                os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
            ) as f:
                pickle.dump(results_all, f)
        '''
        with open(os.path.join(cfg.output_dir,"path_length_list.pkl"),"wb") as f:
            pickle.dump(step_length,f)
            
        
    # Save all data again
    with open(os.path.join(cfg.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_all, f)
    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")
    


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    # log in to use llama model
    login(hf_token)
    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
