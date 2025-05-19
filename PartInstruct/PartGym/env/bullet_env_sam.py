from PartInstruct.PartGym.env.bullet_env import BulletEnv
from PartInstruct.PartGym.env.backend.utils.sam_utils import *
from sam2.build_sam import build_sam2_camera_predictor


class BulletEnv_SAM(BulletEnv):
    """
    BulletEnvSam extends BulletEnv to add SAM (Segment Anything Model) functionality.
    This class inherits all the base functionality from BulletEnv and adds SAM-specific features.
    """

    def __init__(self, config_path="lgm_bc/config/config.yaml", gui=False, obj_class=None, random_sample=False, 
                 evaluation=False, split='val', task_type=None, record=False, check_tasks=True, track_samples=False, 
                 replica_scene=False, skill_mode=True):
        super(BulletEnv_SAM, self).__init__(config_path, gui, obj_class, random_sample, 
                                          evaluation, split, task_type, record, track_samples, 
                                          replica_scene, skill_mode)
        
        self.sam2_video_predictor = sam2_video_predictor
        self.grounding_success = False
        self.grounding_success_num = 0
        self.iou = []
        self.accuracy = []
        self.frame_id = 0
        self.check_tasks = check_tasks

    def set_sam(self, image, obj, part):
        """
        Use SAM to segment a part of an object in an image.
        
        Args:
            image: The image to segment
            obj: The object name
            part: The part name to segment
            
        Returns:
            A tuple (success, mask) where success is a boolean and mask is the segmentation mask
        """
        scene_text_input = obj
        part_text_input = f"The {part} of {obj}"
        self.sam2_video_predictor.load_first_frame(image)
        image = Image.fromarray(image)
        sampled_points = phrase_grounding_and_segmentation(
                            image=image,
                            scene_text_input=scene_text_input,
                            part_text_input=part_text_input
                        )
        if not sampled_points:
            return False
        print("sampled_points", sampled_points)
        sampled_points = [(index, np.array([y, x])) for index, (x, y) in sampled_points]
        points = np.array([coord for _, coord in sampled_points], dtype=np.float32)
        if points.shape[0] != 2:
            return False
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1,0], dtype=np.int32)
        
        ann_obj_id = (1)
        _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        show_points(points, labels, plt.gca())
        mask_np = show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        return True, mask_np

    def step(self, action, gain=0.01, gain_gripper=0.01, gain_near_target=0.01):
        # Execute one time step within the environment
        self.action_list.append(action)
        self._take_action(action, gain, gain_gripper, gain_near_target)
        self.num_steps += 1
        self.frame_id += 1
        observation = self._get_observation()
        self.last_state = self.current_state
        self.current_state = copy.deepcopy(observation)
        position, orientation = self.world.get_body_pose(self.obj)
        self.current_state["obj_pose"] = Transform(Rotation.from_quat(list(orientation)), list(position))

        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        self.current_state["tcp_pose"] = Transform(Rotation.from_quat(list(tcp_orientation)), list(tcp_position))

        reward = 1.0  # Dummy reward

        resample_spatial = False

        if self.skill_mode:
            done_cur_skill, done = self._check_if_done()
            self.done_cur_skill = done_cur_skill
            cur_skill = copy.deepcopy(self.cur_skill)
            if done_cur_skill:
                self.cur_skill_idx += 1
                if not done:
                    self.cur_skill = self.chain_params[self.cur_skill_idx]["skill_name"]
                    self.cur_skill_params = self.chain_params[self.cur_skill_idx]["params"]
                    self.cur_target_part = safe_get([self.cur_skill_params[key] for key in self.cur_skill_params.keys() if "part" in key], 0)
                    # DEBUG
                    print("check cur_target_part", self.cur_target_part)
                    resample_spatial = True
                    self.initial_obj_pose = self.obj.get_pose()
                if not done and self.cur_target_part:
                    frame_input = observation['agentview_rgb'].transpose((1, 2, 0))
                    result = self.set_sam(frame_input, self.obj_class, self.cur_target_part)
                    if isinstance(result, tuple):
                        self.grounding_success, mask_np = result
                    else:
                        self.grounding_success = result
                        mask_np = None
                    if self.grounding_success:
                        self.grounding_success_num += 1
        else:
            done = self._check_if_done_test_eval()
            if not self.skill_chain in ["8", "10", "12", "13", "14", "15", "16", "17"]:
                resample_spatial = True
        
        self.semantic_grounding(resample_spatial)

        info = {
            "Success": done,
            "Completion Rate": self.completion_rate,
            "Steps": self.num_steps,
            "Object Pose": list(self.current_state["obj_pose"].to_list()),
            "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
            "Joint States": list(self.current_state["joint_states"]),
            "Gripper State": list(self.current_state["gripper_state"]),
            "Action": list(action),
            "chain_params": self.chain_params,
            "Grounding Success": (self.grounding_success_num) / len(self.chain_params),
            "iou": self.iou,
            "accuracy": self.accuracy
        }
        if self.skill_mode:
            info.update({
                "Current Skill": cur_skill,
                "Current Skill Success": done_cur_skill,
                "Completion Rate": self.cur_skill_idx/len(self.chain_params),
            })
        
        if self.evaluation:
            info.update({
                "ep_id": self.ep_id,
                "Action": self.action_list,
                "Instruction": self.instruction,
                "Object id": self.obj_id,
                "Task type": self.task_type,
                "Object scale": self.obj_scale,
                "Object init pose": self.obj_init_position,
                "Object init orient": self.obj_init_orientation,
                "obj_class": self.obj_class
            })
        self.info = info

        if self.record:
            self.state_sequence_buffer.append(info)

        return observation, reward, done, info