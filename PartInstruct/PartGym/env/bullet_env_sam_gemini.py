import gym
from gym import spaces
import numpy as np
import copy
import os
import cv2
import json
from torch import nn
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops
from transformers import T5EncoderModel, T5Tokenizer
import shutil
from concurrent.futures import ThreadPoolExecutor
from lgplm_sim.panda_arm import PandaArm
import pybullet
from lgplm_sim.bullet_planner_task_val import OracleChecker
from omegaconf import OmegaConf
import lgplm_sim.utils.bullet_sim as bullet_sim
from lgplm_sim.utils.bullet_sim import _save_image, _save_npz, _save_depth, _look_at_to_extrinsic
from lgplm_sim.utils.vision_utils import *
from lgplm_sim.utils.transform import *
from lgplm_sim.utils.perception import *
from lgplm_sim.utils.scene_utils import *
from lgplm_sim.semantic_parser import SemanticParser, mapping_rotate, mapping_translate
import os
import json
import openai
import csv
import re
from omegaconf import OmegaConf
import argparse
import base64
import requests
from sentence_transformers import SentenceTransformer
from itertools import chain
import re
import google.generativeai as genai
from enum import Enum
import openai


import sys
import os
sam_path = '/scratch/tshu2/yyin34/segment-anything-2-real-time'
sys.path.append(sam_path)

from lgplm_sim.utils.sam_utils import *
from sam2.build_sam import build_sam2_camera_predictor

openai.api_key = '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'
SAM2_CHECKPOINT = "/scratch/tshu2/yyin34/Grounded-SAM-2/checkpoints/sam2_hiera_small.pt"
SAM2_CONFIG = "sam2_hiera_s.yaml"
sam2_video_predictor = build_sam2_camera_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)

api_key = 'AIzaSyB3R05qL4FVpVi4ADTqENJdqSDpBA5M9VE'
genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash-002')
model = genai.GenerativeModel('gemini-2.0-flash-exp')
class EnvParams(Enum):
    USE_CONFIG = -1
    SAMPLE = 0

class T5Encoder(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path='t5-small'):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, text: str) -> dict:
        output = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        output_np = {key: value.numpy() for key, value in output.items()}
        return output_np
    
    def encode_from_tokenized(self, tokenized: dict):
        outputs = self.t5_encoder(**tokenized)
        # Mean pooling across the sequence dimension (sequence length is the second dimension)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeddings

    def decode_tokenized(self, tokenized: dict) -> list:
        input_ids = tokenized['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy()
        
        reshaped = False
        if len(input_ids.shape) == 3:
            T, B, N = input_ids.shape
            input_ids = input_ids.reshape(T * B, N)
            reshaped = True
        
        decoded_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        if reshaped:
            decoded_text = np.array(decoded_text).reshape(T, B).tolist()
        
        return np.array(decoded_text)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        outputs = self.t5_encoder(**inputs)
        # Mean pooling across the sequence dimension (sequence length is the second dimension)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeddings

    def output_shape(self):
        hidden_size = self.t5_encoder.config.d_model
        return (hidden_size,)

lang_encoder = T5Encoder()

from PartInstruct.PartGym.env.bullet_env_sam import BulletEnv_SAM

class BulletEnv_SAM_Gemini(BulletEnv_SAM):
    """
    BulletEnv_SAM_Gemini extends BulletEnv_SAM to add Gemini functionality.
    This class inherits all the base functionality from BulletEnv_SAM and adds Gemini-specific features.
    """

    def __init__(self, task_instruction, episode_info, metadata_file, benchmark_file, api_key, openai_api_key):
        super().__init__(task_instruction, episode_info, metadata_file, benchmark_file)
        
        # Initialize Gemini model
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        google.generativeai.configure(api_key=self.api_key)
        self.model = google.generativeai.GenerativeModel('gemini-pro-vision')
        
        # Initialize Gemini task inference
        self.gemini_inference = GeminiTaskInference(metadata_file, benchmark_file)
        self.gemini_inference.model = self.model
        
        # Initialize state variables
        self.first_frame_av = None
        self.last_frame_av = None
        self.first_tcp_pose = None
        self.last_tcp_pose = None
        self.gripper_state = None
        self.executed_skill_chain = []
        
        # Initialize skill average steps
        self.skill_avg_steps = {
            "grasp_obj": 10,
            "move_gripper": 15,
            "rotate_obj": 20,
            "touch_obj": 10,
            "release_obj": 5
        }

    def set_sam(self, image, obj, part):
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
        points =  np.array([coord for _, coord in sampled_points], dtype=np.float32)
        if points.shape[0] != 2:
            return False
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1,0], dtype=np.int32)
        # print("sampled_points for SAM", sampled_points)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(12, 8))
        # plt.imshow(image)
        # plt.title(f"frame 0")
        # print("self.idx !!!!", self.idx)
        # plt.savefig(f"/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/sam_test/{self.idx}_first_frame.png")
        
        ann_obj_id = (1)
        _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        show_points(points, labels, plt.gca())
        mask_np = show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        # plt.savefig(f"/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/sam_test/{self.idx}_frame_marked.png")
        # plt.close()

        return True, mask_np

    class GeminiTaskInference:
        def __init__(self, metadata_file, benchmark_file):
            self.metadata_file = metadata_file
            self.benchmark_file = benchmark_file
            self.data = self._load_json(metadata_file)
            self.benchmark_task_data = self._load_csv(benchmark_file)
            self.train_data = self._load_train_data(self.data)
            self.skill_chain = None
            self.obj_class = "object"

        def _load_json(self, file_path):
            """Load JSON file."""
            with open(file_path, 'r') as f:
                return json.load(f)

        def _load_csv(self, file_path):
            """Load benchmark tasks from CSV."""
            benchmark_task_data = []
            with open(file_path, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    benchmark_task_data.append(row)
            return benchmark_task_data

        def _load_train_data(self, data):
            """Sample one episode per skill_chain (1-10) from the 'mug' object."""
            train_data = {}
            if 'mug' in data and 'train' in data['mug']:
                mug_data = data['mug']['train']
                skill_chain_set = set()
                for skill_chain_label, episode_list in mug_data.items():
                    for episode in episode_list:
                        if int(skill_chain_label) in range(1, 11) and skill_chain_label not in skill_chain_set:
                            skill_chain_set.add(skill_chain_label)
                            if 'mug' not in train_data:
                                train_data['mug'] = []
                            train_data['mug'].append(episode)
                            if len(skill_chain_set) == 10:
                                break
            return train_data

        def train_gemini_on_train(self):
            """Train Gemini on the training dataset."""
            training_map = {}
            for obj_name, obj_episodes in self.train_data.items():
                for episode in obj_episodes:
                    task_instruction = episode.get('task_instruction')
                    chain_params = episode.get('chain_params')
                    skill_instructions = episode.get('skill_instructions')
                    if task_instruction and chain_params:
                        training_map[task_instruction] = {
                            "task_instruction": task_instruction,
                            "skill_instructions": skill_instructions,
                            "chain_params": chain_params
                        }
            return training_map

        def gemini_infer_task(self, user_input):
            """Use Gemini to infer the correct task sequence based on the user's natural language input."""
            object_name = self._extract_object_name(user_input)
            self.obj_class = object_name
            
            training_map = self.train_gemini_on_train()
            task_options = [chain for chain in training_map.values()]
            task_options_str = "\n".join([f"{i+1}. {option}" for i, option in enumerate(task_options)])
            benchmark_str = "\n".join([f"{row['Order']}. {row['Task Description']} -> {row['Chain of Base Skills']}" for row in self.benchmark_task_data])

            prompt = f"""
            You are a task instruction inference expert. Based on the user's instruction: '{user_input}',
            generate the best-matching task sequence from the given options.
            The sequence should only be in the form of a valid Python list of dictionaries,
            with no extra text, reasoning or formatting like `json` or 'python' or '```'.
            The correct output format is: [{{"skill_name": "grasp_obj", "params": {{"part_grasp": "pressing lid"}}}}]. You shall scrictly follow the format without extra output like ```python.
            Important:
            - Ensure that the part names used in the 'params' section exactly match the terms given by the user in the input, without making assumptions or changes.
            - Here are the unique part names associated with each object as listed below:
                - Scissors: blade, handle, screw, left, right, top, bottom, front, back
                - Kitchen Pot: base body, lid, left, right, top, bottom, front, back
                - Laptop: base frame, screen, touchpad, keyboard, screen frame, left, right, top, bottom, front, back
                - Eyeglasses: base body, leg, left, right, top, bottom, front, back
                - Bucket: handle, base body, left, right, top, bottom, front, back
                - Display: base support, surface, frame, screen, left, right, top, bottom, front, back
                - Pliers: base body, leg, outlier, left, right, top, bottom, front, back
                - Bottle: mouth, lid, body, neck, left, right, top, bottom, front, back
                - Knife: base body, translation blade, rotation blade, left, right, top, bottom, front, back
                - Stapler: base body, lid, body, left, right, top, bottom, front, back
                - Kettle: handle, base body, lid, left, right, top, bottom, front, back
                - Mug: handle, body, containing things, left, right, top, bottom, front, back
                - Box: rotation lid, base body, left, right, top, bottom, front, back
                - Dispenser: base body, pressing lid, head, handle, outlier, left, right, top, bottom, front, back
            - The dir_move can only be top, left, right, bottom, front, back. There are no dir_move called up, down, upwards and downwards, reverse.
            - Do not modify or assume alternate names for object parts.
            - The task sequence should follow the user's input as strictly as possible.
            - For move_gripper, the dir_move can only be one of top, bottom, left, right
            - Do not replace object parts with similar or inferred names.
            - For box, it has rotation lid rather than lid. Only box has rotation lid.
            - Only bucket, mug and scissors have part called handle. Don't infer handle part name for other objects.
            - Skill chain for release_obj shall look like {{"skill_name": "release_obj", "params": {{}}}}
            - If the instruction involves multiple steps (e.g., rotating in two steps), generate a task sequence that matches this. You may need to include an intermediate release and re-grasp.
            Skill descriptions:
            1. **grasp_obj**:
                - **Description**: This skill grasps an object by a specific part.
                - **Parameters**:
                    - **part_grasp**: The exact part of the object to be grasped. Must match the user's input (e.g., 'blade', 'lid').

            2. **move_gripper**:
                - **Description**: This skill moves the gripper in a specified direction while optionally keeping an object grasped.
                - **Parameters**:
                    - **dir_move**: Direction to move the gripper. Can be 'top', 'bottom', 'left', or 'right'.
                    - **grasping**: Boolean indicating whether the gripper is still grasping the object (true/false).
                    - **put_down**: Boolean indicating whether the object is put down during the movement.
                    - **touching**: Boolean indicating whether the object is touched during the movement.

            3. **rotate_obj**:
                - **Description**: This skill rotates an object in a specific direction based on a given part.
                - **Parameters**:
                    - **dir_rotate**: Direction to rotate the object. Must be one of 'top', 'bottom', 'left', 'right'.
                    - **part_rotate**: The part of the object that should be rotated.

            4. **touch_obj**:
                - **Description**: This skill touches a part of an object.
                - **Parameters**:
                    - **part_touch**: The part of the object to be touched.

            5. **release_obj**:
                - **Description**: This skill releases an object from the gripper.
                - **Parameters**: None.
            
            Here are the task sets(1-10) with task descriptions and correspondning chain of skills:
            {benchmark_str} where RLFB in move_gripper represents four direction right, left, front, back respectively while U represents up and D represents down.
            You can infer the correct skill chain from the `task_instruction` using above information. Here are some suggestions about how to split the task instruction:
            1. **Break down the task**: Split the task instruction into individual steps. For example, if the task says "push the stapler towards the left by touching the right, then rotate back to point to front", it consists of two actions: (1) push the stapler left while touching the right, (2) rotate it to face the front.
            
            2. **Map actions to skills**: 
                - **Touching an object**: If the instruction involves touching an object or a part, use the `touch_obj` skill. The parameter for `touch_obj` should specify the part being touched (e.g., `"part_touch": "right"` for touching the right part).
                - **Moving an object**: If the instruction involves moving the object, use the `move_gripper` skill. The parameter `dir_move` specifies the direction (e.g., `"dir_move": "left"` for moving left). Ensure the correct `grasping`, `touching`, and `put_down` flags are set.
                - **Releasing an object**: If the instruction indicates releasing or letting go of the object, use the `release_obj` skill. This skill does not require any parameters.
                - **Grasping an object**: If the instruction requires picking up or holding an object, use the `grasp_obj` skill. The parameter `part_grasp` should specify the part to be grasped (e.g., `"part_grasp": "back"` for grasping the back).
                - **Rotating an object**: If the instruction indicates rotating or reorienting the object, use the `rotate_obj` skill. The parameters `dir_rotate` (e.g., `"dir_rotate": "front"`) and `part_rotate` (e.g., `"part_rotate": "left"`) specify the direction and part being rotated.

            3. **Example of full task breakdown**: 
            **Task Instruction**: "While keeping it on the table, push the stapler towards the left by touching the right, then rotate back to point to front."
            - First, touch the stapler at its right: This maps to `touch_obj` with `"part_touch": "right"`.
            - Then, push it to the left: This maps to `move_gripper` with `"dir_move": "left"`, `grasping: false`, and `touching: true`.
            - Release the stapler: This maps to `release_obj` with no parameters.
            - Grasp the stapler again, this time at its back: This maps to `grasp_obj` with `"part_grasp": "back"`.
            - Finally, rotate it so the left side faces the front: This maps to `rotate_obj` with `"dir_rotate": "front"` and `"part_rotate": "left"`.

            For Skill Selection and Inference:
            1. Refer to `task_options_str`, which contains examples of typical skill patterns from seen tasks. When an exact pattern match is possible, generate skills and parameters accordingly.
            2. When no exact pattern match is found, use reasoning to infer a skill sequence that best aligns with the user's task instruction, even for unseen task types.
            Here are the examples of task instrutions and corresponding decomposed skill-level instructions and chain params:
            {task_options_str}
            """

            try:
                response = self.model.generate_content(prompt)
                gemini_output = response.text
                print("gemini_output: " + str(gemini_output))
                gemini_output = re.sub(r'```(?:json|python)?', '', gemini_output).strip()
                gemini_output = gemini_output.replace("'", '"').replace("True", "true").replace("False", "false").strip()

                parsed_skill_chain = json.loads(gemini_output)
                skill_instructions = self._convert_to_instructions(parsed_skill_chain)
                if parsed_skill_chain[0]["skill_name"] == "move_gripper":
                    if parsed_skill_chain[0]["params"]["put_down"] == True:
                        parsed_skill_chain[0]["params"]["dir_move"] = "bottom"

                return parsed_skill_chain, object_name, skill_instructions

            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini response: {gemini_output}")
                return f"Error: {e}"

        def _extract_object_name(self, user_input):
            """Extract the object name from the user's input."""
            match = re.search(r'\b(the|a|an)\s+(\w+)', user_input)
            if match:
                return match.group(2)
            else:
                return 'object'

        def _convert_to_instructions(self, skill_chain):
            """Convert the parsed skill chain into a list of string instructions."""
            skill_instructions = []
            for skill in skill_chain:
                self.cur_skill = skill['skill_name']
                self.cur_skill_params = skill['params']
                instruction = self._get_instruction()
                skill_instructions.append(instruction)
            return skill_instructions

        def _get_instruction(self):
            """Generate a human-readable instruction based on the current skill and its parameters."""
            if self.cur_skill == "grasp_obj":
                region_str = ""
                instruction = f"Grasp the {self.obj_class} at {region_str} its {self.cur_skill_params['part_grasp']}"
            elif self.cur_skill == "rotate_obj":
                dir_str = self.mapping_rotate(self.cur_skill_params['dir_rotate'])
                instruction = f"Reorient the {self.cur_skill_params['part_rotate']} of the {self.obj_class} to face {dir_str}"
            elif self.cur_skill == "move_gripper":
                dir_str = self.mapping_translate(self.cur_skill_params['dir_move'])
                instruction = f"Move {dir_str}"
            elif self.cur_skill == "touch_obj":
                region_str = ""
                instruction = f"Touch the {self.obj_class} at {region_str} its {self.cur_skill_params['part_touch']}"
            elif self.cur_skill == "release_obj":
                instruction = "Release"
            return instruction

        def mapping_rotate(self, word):
            return {
                'front': 'front',
                'back': 'back',
                'top': 'upwards',
                'bottom': 'downwards',
                'left': 'left',
                'right': 'right'
            }.get(word)

        def mapping_translate(self, word):
            return {
                'front': 'forwards',
                'back': 'backwards',
                'top': 'upwards',
                'bottom': 'downwards',
                'left': 'to the left',
                'right': 'to the right'
            }.get(word)

    def reset(self, obj_class=EnvParams.SAMPLE, obj_id=EnvParams.SAMPLE, skill_chain=EnvParams.SAMPLE, chain_params=EnvParams.SAMPLE, 
              obj_scale=0.1, obj_position=[0.0, -0.031480762319536344, 0.15], obj_orientation=list(Rotation.identity().as_quat())):
        """Reset the environment and initialize task execution."""
        super().reset(obj_class, obj_id, skill_chain, chain_params, obj_scale, obj_position, obj_orientation)
        
        # Reset state variables
        self.first_frame_av = None
        self.last_frame_av = None
        self.first_tcp_pose = None
        self.last_tcp_pose = None
        self.gripper_state = None
        self.executed_skill_chain = []
        
        # Get initial task plan
        self.gemini_skill_chain, object_name, self.gemini_skill_instructions = self.gemini_inference.gemini_infer_task(self.task_instruction)
        self.chain_params = self.gemini_skill_chain
        
        return self.get_observation()

    def step(self, action):
        """Execute one step in the environment."""
        # Get current skill and parameters
        current_skill = self.chain_params[0]
        skill_name = current_skill["skill_name"]
        params = current_skill["params"]
        
        # Execute the skill
        success = self.execute_skill(skill_name, params)
        
        # Check if skill is completed
        if success:
            completion = self.check_skill_completion(skill_name, params)
            if completion:
                # Remove completed skill from chain
                self.chain_params.pop(0)
        
        # Get observation
        obs = self.get_observation()
        
        # Check if task is done
        done = len(self.chain_params) == 0
        
        # Calculate reward
        reward = 1.0 if done else 0.0
        
        return obs, reward, done, {}

    def get_observation(self):
        """Get the current observation."""
        return {
            "image": self.get_frame(),
            "gripper_state": self.robot.get_gripper_state(),
            "tcp_pose": self.current_state["tcp_pose"].to_list(),
            "skill_chain": self.chain_params,
            "executed_skills": self.executed_skill_chain
        }

    def get_embedding(self, text):
        emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return emb_model.encode(text, convert_to_tensor=True)

    def semantic_grounding(self, resample_spatial=True):
        self.parser.update_part_pcds(resample_spatial)
        if self.use_language:
            self.instruction, self.tokenized_instruction = self._get_instruction()
    # def semantic_grounding(self, resample_spatial=True):
    #     if self.cur_skill == "grasp_obj":
    #         missing_keys = set(["part_grasp"]) - self.cur_skill_params.keys()
    #         # Assert if any keys are missing
    #         assert not missing_keys, f"Missing required keys: {missing_keys}"
    #         part_grasp = self.cur_skill_params["part_grasp"]
    #         if "region_on_part" in self.cur_skill_params:
    #             region_on_part = self.cur_skill_params["region_on_part"]
    #         else:
    #             region_on_part = ""
    #         # Get the grounded pcd bbox
    #         print("-----")
    #         print("1")
    #         self.parser.update_part_pcds(resample_spatial)
    #         print("2")
    #         grounded_pcd = self.parser.part_pcds[self.obj_class] if part_grasp=="" else self.parser.part_pcds[part_grasp]
    #         print("3")
    #         grounded_pcd = grounded_pcd if region_on_part=="" else self.parser.spatial_sampler.sample_query(grounded_pcd, region_on_part)
    #         print("4")
    #         self.target_grasp_bbox = np.array([np.min(grounded_pcd, axis=0), np.max(grounded_pcd, axis=0)])
    #         print("5")
    #         self.target_grasp_bbox_center =  np.mean(grounded_pcd, axis=0)
    #         print("6")
    #         half_lengths = (self.target_grasp_bbox[1] - self.target_grasp_bbox[0]) / 2
    #         print("7")
    #         self.target_grasp_expanded_bbox = np.array([self.target_grasp_bbox_center - self.config.grasp_bbox_ext_ratio * half_lengths, 
    #                                 self.target_grasp_bbox_center + self.config.grasp_bbox_ext_ratio * half_lengths])
    #         print("8")
    #     elif self.cur_skill == "move_gripper":
    #         missing_keys = set(["dir_move"]) - self.cur_skill_params.keys()
    #         # Assert if any keys are missing
    #         assert not missing_keys, f"Missing required keys: {missing_keys}"
    #         self.parser.update_part_pcds(resample_spatial)
    #         dir_move = self.cur_skill_params["dir_move"]
    #         direction = np.array(self.parser.spatial_sampler.gaussian_param_mappings[dir_move]['mu'])
    #         direction = direction/np.linalg.norm(direction)
    #         position = self.robot.get_tcp_pose().translation
    #         # position, orientation = self.world.get_body_pose(self.obj)
    #         self.target_translation = np.array(position)+direction*self.config.translate_distance
    #         self.target_translation_direction = direction

    #     elif self.cur_skill == "rotate_obj":
    #         missing_keys = set(["part_rotate", "dir_rotate"]) - self.cur_skill_params.keys()
    #         # Assert if any keys are missing
    #         assert not missing_keys, f"Missing required keys: {missing_keys}"
    #         part_rotate = self.cur_skill_params["part_rotate"]
    #         dir_rotate = self.cur_skill_params["dir_rotate"]
    #         position, orientation = self.world.get_body_pose(self.obj)
    #         direction = np.array(self.parser.spatial_sampler.gaussian_param_mappings[dir_rotate]['mu'])
    #         self.parser.update_part_pcds(resample_spatial)
    #         grounded_pcd = self.parser.part_pcds[part_rotate]
    #         grounded_center =  np.mean(grounded_pcd, axis=0)
    #         rel_quaternion, _ = rotation_quaternion(grounded_center-np.array(position), direction)
    #         rel_rotation = Rotation(rel_quaternion)
    #         current_rotation = Rotation(list(orientation))
    #         self.target_rotation = rel_rotation*current_rotation
        
    #     elif self.cur_skill == "touch_obj":
    #         missing_keys = set(["part_touch"]) - self.cur_skill_params.keys()
    #         # Assert if any keys are missing
    #         assert not missing_keys, f"Missing required keys: {missing_keys}"
    #         part_touch = self.cur_skill_params["part_touch"]
    #         if "region_on_part" in self.cur_skill_params:
    #             region_on_part = self.cur_skill_params["region_on_part"]
    #         else:
    #             region_on_part = ""
    #         # Get the grounded pcd bbox 
    #         self.parser.update_part_pcds(resample_spatial)
    #         grounded_pcd = self.parser.part_pcds[self.obj_class] if part_touch=="" else self.parser.part_pcds[part_touch]
    #         grounded_pcd = grounded_pcd if region_on_part=="" else self.parser.spatial_sampler.sample_query(grounded_pcd, region_on_part)
    #         self.target_touch_bbox = np.array([np.min(grounded_pcd, axis=0), np.max(grounded_pcd, axis=0)])
    #         self.target_touch_bbox_center =  np.mean(grounded_pcd, axis=0)
    #         half_lengths = (self.target_touch_bbox[1] - self.target_touch_bbox[0]) / 2
    #         self.target_touch_expanded_bbox = np.array([self.target_touch_bbox_center - self.config.touch_bbox_ext_ratio * half_lengths, 
    #                                   self.target_touch_bbox_center + self.config.touch_bbox_ext_ratio * half_lengths])

    #     elif self.cur_skill == "release_obj":
    #         self.parser.update_part_pcds(resample_spatial)
    #         init_tcp_pose = self.robot.get_tcp_pose()
    #         init_tcp_position = init_tcp_pose.translation
    #         init_tcp_orientation = init_tcp_pose.rotation.as_matrix()

            # # Define the distance to move along the -z axis of the TCP coordinate
            # release_distance_tcp_z = self.config.release_distance_tcp_z
            # # Calculate the target position along the -z axis of the TCP coordinate
            # target_tcp_position = init_tcp_position + release_distance_tcp_z * init_tcp_orientation[:, 2]

            # # Move the robot to the target position along the -z axis of the TCP coordinate
            # self.release_new_tcp_pose = Transform(init_tcp_pose.rotation, target_tcp_position)
            # new_tcp_position = self.release_new_tcp_pose.translation

            # # Check if the z-axis of the TCP is not parallel to the z-axis of the world frame
            # tcp_z_axis_world = init_tcp_orientation[:, 2]
            # world_z_axis = np.array([0, 0, 1])

            # # Calculate the angle between the TCP z-axis and the world z-axis
            # angle_between_z_axes = np.arccos(np.dot(tcp_z_axis_world, world_z_axis))

            # # If the angle is not zero (not parallel), move a little along the +z axis of the world frame
            # if angle_between_z_axes > 1e-3:  # Allowing for a small numerical tolerance
            #     release_distance_world_z = self.config.release_distance_world_z  # Example distance; adjust as needed
            #     target_world_position = new_tcp_position + release_distance_world_z * world_z_axis

            #     # Move the robot to the target position along the +z axis of the world frame
            #     self.release_new_tcp_pose = Transform(self.release_new_tcp_pose.rotation, target_world_position)

    #     # Update language instruction
    #     if self.use_language:
    #         self.instruction, self.tokenized_instruction = self._get_instruction()
        
    #     # Update the current part
        part_keys = [key for key in self.cur_skill_params if "part" in key]
        # print(part_keys)
        if part_keys:
            part_key = self.cur_skill_params[part_keys[0]]
        else:
            part_key = []
        if part_key:
            self.cur_target_part = part_key
            print("key", part_key)
        else:
            self.cur_target_part = None
    
    def save_frame_as_base64(self, frame):
        # Convert from (Channels, Height, Width) -> (Height, Width, Channels) if needed
        if frame.shape[0] == 3 or frame.shape[0] == 4:
            frame = frame.transpose(1, 2, 0)
        # Encode the frame as a base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def gpt_completion_checker(self, base64_image1, base64_image2, skill_instruction, gripper_state, first_tcp_pose, last_tcp_pose):
        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        # Combine skill instruction with the rest of the text
        combined_text = (
            f"Here are two images: the first and last frames from a global camera for current skill."
            f"Here are current skill instruction: {skill_instruction}."
            f"This is the state of gripper: {gripper_state}. The gripper is open when the value is around 0.04 and it is closed when the value is less or around 0.018.\
                This is the tcp pose of the first frame: {first_tcp_pose} and this the tcp pose of the last frame: {last_tcp_pose}. \
                The tcp format is np.r_[self.rotation.as_quat(), self.translation] where the first four elements are the rotation quaternion and the last three are the translation element.\
                You can use the tcp poses difference to infer if the move_gripper skill is complete (the Euclidean distance of two tcp positions should be at least 0.05)."
            "Please analyze the images and determine whether the task is completed between the first and last frames. "
            "Focus on the position of the object relative to the gripper. "
            "A small gap between the gripper and the object can be allowed."
            "Respond with only 'true' if the instruction is completed, and only 'false' otherwise."
        )
        # Payload for GPT-4
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": combined_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image1}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image2}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10  # Keeping it short for a true/false response
        }
        # Send request to OpenAI API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            response_json = response.json()
            
            if 'choices' not in response_json:
                print(f"Unexpected response: {response_json}")
                return "Error: Unexpected response structure"
            
            # Parse and return the boolean response
            response_text = response_json['choices'][0]['message']['content'].strip().lower()
            
            if response_text == 'true' or 'true' in response_text or 'True' in response_text:
                return True
            elif response_text == 'false' or 'false' in response_text or 'False' in response_text:
                return False
            else:
                return False
        except Exception as e:
            return f"Error processing API response: {str(e)}"

    def step(self, action, gain=0.01, gain_gripper=0.01, gain_near_target=0.01):
        print("start step")
        self.action_list.append(action)
        current_position = self.robot.get_tcp_pose().translation    
        # print("current_position_step")
        # print(current_position)
        # Execute one time step within the environment
        self._take_action(action, gain, gain_gripper, gain_near_target)
        self.num_steps+=1
        self.frame_id+=1
        observation = self._get_observation()
        # print("GET OBS")
        self.last_state = self.current_state
        self.current_state = copy.deepcopy(observation)
        position, orientation = self.world.get_body_pose(self.obj)
        # print("GET POS")
        self.current_state["obj_pose"] = Transform(Rotation.from_quat(list(orientation)), list(position))

        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        self.current_state["tcp_pose"] = Transform(Rotation.from_quat(list(tcp_orientation)), list(tcp_position))
        reward = 1.0
        print("Before done check")
        done_check = self._check_if_done_test_eval()
        print("After done check")
        # done_check = False
        done = done_check
        # LLM
        done_cur_skill = False
        part_keys = [key for key in self.cur_skill_params if "part" in key]
        # Part name does not exist
        if part_keys != [] and self.cur_skill_params[part_keys[0]] not in self.parser.part_pcds: 
            current_part_embedding = self.get_embedding(self.cur_skill_params[part_keys[0]]).cpu().numpy()
            part_embeddings = {part: self.get_embedding(part).cpu().numpy() for part in self.parser.part_pcds}
            best_match = None
            best_similarity = -1
            print("items9")
            for part, part_embedding in part_embeddings.items():
                similarity = np.dot(current_part_embedding, part_embedding) / (np.linalg.norm(current_part_embedding) * np.linalg.norm(part_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = part
            if best_match:
                print(f"Updated '{self.cur_skill_params[part_keys[0]]}' to closest match: '{best_match}' (Similarity: {best_similarity})")
                self.cur_skill_params[part_keys[0]] = best_match
            else:
                done = True
                print(f"No suitable match found for '{self.cur_skill_params[part_keys[0]]}'")
                self.cur_skill_params[part_keys[0]] = ""
                info = {
                    "ep_id": self.ep_id,
                    "Task Success": done_check,
                    "part_name_error": False,
                    "Skill_timeout": False,
                    "Task_timeout": False,
                    "Completion Rate": self.completion_rate,
                    "Current Skill": cur_skill,
                    "GPT Current Skill Success": done_cur_skill,
                    "Steps": self.num_steps,
                    "Object Pose": list(self.current_state["obj_pose"].to_list()),
                    "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
                    "Joint States": list(self.current_state["joint_states"]),
                    "Gripper State": list(self.current_state["gripper_state"]),
                    "Action": self.action_list,
                    "Reward": reward,
                    "Instruction": self.task_instruction,
                    "gpt_chain_params": self.executed_skill_chain,
                    "actual_chain_params": self.actual_chain_params,
                    "is_skill_done": self.is_skill_done,
                    "Grounding Success":(self.grounding_success_num) / len(self.chain_params),
                    "iou": self.iou,
                    "accuracy": self.accuracy,
                    "Object id": self.obj_id,
                    "Task type": self.task_type,
                    "Object scale": self.obj_scale,
                    "Object init pose": self.obj_init_position,
                    "Object init orient": self.obj_init_orientation,
                    "obj_class": self.obj_class
                }
        
                return observation, reward, done, info
        print("----------------------------")
        # print(len(self.gpt_skill_instructions))
        # print(self.current_state["gripper_state"][0])
        # print("Current skill: " + str(self.gpt_skill_instructions[self.cur_skill_idx]))
        print(len(self.actual_chain_params))
        print(self.cur_skill_idx)
        print(self.counter)
        print(self.cur_skill_params)
        print("----------------------------")
        skill_avg = self.skill_avg_steps[self.cur_skill]
        print("++")
        if self.counter == 0:
            self.first_frame_av = observation["agentview_rgb"]
            # self.first_frame_wr = observation["wrist_rgb"]
            self.first_tcp_pose = list(self.current_state["tcp_pose"].to_list())
        print("C4")
        cur_skill = copy.deepcopy(self.cur_skill)
        if self.counter > self.skill_avg_steps[self.cur_skill]:
            is_current_skill_done = self.checker.is_skill_done()
            self.is_skill_done.append(is_current_skill_done)
            print(is_current_skill_done)
            print("Skill " + str(self.cur_skill_idx) + " completed.")
            self.cur_skill_idx+=1
            self.counter = 0
            if self.cur_skill_idx >= len(self.actual_chain_params) * 2:
                done = True
                info = {
                    "ep_id": self.ep_id,
                    "Task Success": done_check,
                    "part_name_error": False,
                    "Skill_timeout": False,
                    "Task_timeout": False,
                    "Completion Rate": self.completion_rate,
                    "Current Skill": cur_skill,
                    "GPT Current Skill Success": done_cur_skill,
                    "Steps": self.num_steps,
                    "Object Pose": list(self.current_state["obj_pose"].to_list()),
                    "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
                    "Joint States": list(self.current_state["joint_states"]),
                    "Gripper State": list(self.current_state["gripper_state"]),
                    "Action": self.action_list,
                    "Reward": reward,
                    "Instruction": self.task_instruction,
                    "gpt_chain_params": self.executed_skill_chain,
                    "actual_chain_params": self.actual_chain_params,
                    "is_skill_done": self.is_skill_done,
                    "Grounding Success":(self.grounding_success_num) / len(self.chain_params),
                    "iou": self.iou,
                    "accuracy": self.accuracy,
                    "Object id": self.obj_id,
                    "Task type": self.task_type,
                    "Object scale": self.obj_scale,
                    "Object init pose": self.obj_init_position,
                    "Object init orient": self.obj_init_orientation,
                    "obj_class": self.obj_class
                }
                return observation, reward, done, info
            if not done:
                print("--")
                current_rgb_base64 = self.save_frame_as_base64(self.current_state["agentview_rgb"])
                self.gripper_state = self.robot.get_gripper_state()
                self.last_tcp_pose = list(self.current_state["tcp_pose"].to_list())
                next_skill_params, object_name, next_skill_instruction = self.gemini_inference.gemini_infer_next_skill(
                    self.task_instruction,  # Current task instruction
                    self.executed_skill_chain,  # Already executed skills
                    current_rgb_base64,  # Current RGB image
                    self.gripper_state, self.first_tcp_pose, self.last_tcp_pose
                )
                self.checker.reset_skill(next_skill_params["skill_name"], next_skill_params["params"])
                # while next_skill_params == None:
                #     next_skill_params, object_name, next_skill_instruction = self.gemini_inference.gemini_infer_next_skill(
                #         self.task_instruction,  # Current task instruction
                #         self.executed_skill_chain,  # Already executed skills
                #         current_rgb_base64,  # Current RGB image
                #         self.gripper_state, self.first_tcp_pose, self.last_tcp_pose
                #     )
                self.executed_skill_chain.append({
                    "skill_name": next_skill_params["skill_name"],
                    "params": next_skill_params["params"]
                })
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print(next_skill_params)
                print(next_skill_instruction)
                self.cur_skill_instruction = next_skill_instruction
                self.cur_skill = next_skill_params["skill_name"]
                self.cur_skill_params = next_skill_params["params"]
                part_keys = [key for key in self.cur_skill_params if "part" in key]
                # For the next skill instruction, part keys does not exist
                print("****************************")
                print(part_keys)
                if part_keys != [] and self.cur_skill_params[part_keys[0]] not in self.parser.part_pcds:
                    current_part_embedding = self.get_embedding(self.cur_skill_params[part_keys[0]]).cpu().numpy()
                    part_embeddings = {part: self.get_embedding(part).cpu().numpy() for part in self.parser.part_pcds}
                    best_match = None
                    best_similarity = -1
                    print("items10")
                    for part, part_embedding in part_embeddings.items():
                        similarity = np.dot(current_part_embedding, part_embedding) / (np.linalg.norm(current_part_embedding) * np.linalg.norm(part_embedding))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = part
                    if best_match:
                        print(f"Updated '{self.cur_skill_params[part_keys[0]]}' to closest match: '{best_match}' (Similarity: {best_similarity})")
                        self.cur_skill_params[part_keys[0]] = best_match
                    else:
                        print(f"No suitable match found for '{self.cur_skill_params[part_keys[0]]}'")
                        self.cur_skill_params[part_keys[0]] = ""
                        done = True
                        info = {
                            "ep_id": self.ep_id,
                            "Task Success": done_check,
                            "part_name_error": False,
                            "Skill_timeout": False,
                            "Task_timeout": False,
                            "Completion Rate": self.completion_rate,
                            "Current Skill": cur_skill,
                            "GPT Current Skill Success": done_cur_skill,
                            "Steps": self.num_steps,
                            "Object Pose": list(self.current_state["obj_pose"].to_list()),
                            "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
                            "Joint States": list(self.current_state["joint_states"]),
                            "Gripper State": list(self.current_state["gripper_state"]),
                            "Action": self.action_list,
                            "Reward": reward,
                            "Instruction": self.task_instruction,
                            "gpt_chain_params": self.executed_skill_chain,
                            "actual_chain_params": self.actual_chain_params,
                            "is_skill_done": self.is_skill_done,
                            "Grounding Success":(self.grounding_success_num) / len(self.chain_params),
                            "iou": self.iou,
                            "accuracy": self.accuracy,
                            "Object id": self.obj_id,
                            "Task type": self.task_type,
                            "Object scale": self.obj_scale,
                            "Object init pose": self.obj_init_position,
                            "Object init orient": self.obj_init_orientation,
                            "obj_class": self.obj_class
                        }
        
                        return observation, reward, done, info
                if self.cur_skill=="move_gripper":
                    self.pre_move_tcp_pose = self.robot.get_tcp_pose()
                elif self.cur_skill=="rotate_obj":
                    self.pre_rotate_pose = self.obj.get_pose()
                    # DEBUG
                    # self.world.draw_frame_axes(self.pre_move_tcp_pose.translation, self.pre_move_tcp_pose.rotation.as_matrix())
                if self.skill_chain in ["8", "10", "12", "13", "14", "15", "16", "17"]:
                    resample_spatial = False
                else:
                    resample_spatial = True
                self.semantic_grounding(resample_spatial)
        elif self.use_gpt_checker:
            self.counter = self.counter + 1

        info = {
            "ep_id": self.ep_id,
            "Task Success": done_check,
            "part_name_error": False,
            "Skill_timeout": False,
            "Task_timeout": False,
            "Completion Rate": self.completion_rate,
            "Current Skill": cur_skill,
            "GPT Current Skill Success": done_cur_skill,
            "Steps": self.num_steps,
            "Object Pose": list(self.current_state["obj_pose"].to_list()),
            "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
            "Joint States": list(self.current_state["joint_states"]),
            "Gripper State": list(self.current_state["gripper_state"]),
            "Action": self.action_list,
            "Reward": reward,
            "Instruction": self.task_instruction,
            "gpt_chain_params": self.executed_skill_chain,
            "actual_chain_params": self.actual_chain_params,
            "is_skill_done": self.is_skill_done,
            "Grounding Success":(self.grounding_success_num) / len(self.chain_params),
            "iou": self.iou,
            "accuracy": self.accuracy,
            "Object id": self.obj_id,
            "Task type": self.task_type,
            "Object scale": self.obj_scale,
            "Object init pose": self.obj_init_position,
            "Object init orient": self.obj_init_orientation,
            "obj_class": self.obj_class
        }
        
        self.info = info

        if self.record:
            self.state_sequence_buffer.append(info)
            # self.state_sequence.append(info)
        print("end step")
        return observation, reward, done, info

    def clear_buffers(self):
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []
        self.num_steps = self.previous_steps
        self.cur_skill_idx = self.state_copy["cur_skill_idx"]
        self.cur_skill = self.state_copy["cur_skill"]
        self.cur_skill_params = self.state_copy["cur_skill_params"]

    def dump_buffers(self):
        self.render_sequence+=self.render_sequence_buffer
        self.state_sequence+=self.state_sequence_buffer
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []
        self.previous_steps = self.num_steps
        self.state_copy = {"cur_skill_idx": self.cur_skill_idx, "cur_skill": self.cur_skill, "cur_skill_params": self.cur_skill_params}

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])
        view_matrix = self.world.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.config.cam_static_target,
                                                            distance=self.config.cam_static_dist,
                                                            yaw=self.config.cam_static_yaw,
                                                            pitch=self.config.cam_static_pitch,
                                                            roll=0,
                                                            upAxisIndex=self.config.up_axis_index)
        proj_matrix = self.world.p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.config.render_width) /
                                                        self.config.render_height,
                                                        nearVal=self.config.cam_near,
                                                        farVal=self.config.cam_far)
        (_, _, px, _, _) = self.world.p.getCameraImage(width=self.config.render_width,
                                                height=self.config.render_height,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self.config.render_height, self.config.render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def render_all(self):
        renders = {}
        # render agent view images
        agent_view_matrix = self.world.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.config.cam_static_target,
                                                            distance=self.config.cam_static_dist,
                                                            yaw=self.config.cam_static_yaw,
                                                            pitch=self.config.cam_static_pitch,
                                                            roll=0,
                                                            upAxisIndex=self.config.up_axis_index)
        agent_proj_matrix = self.world.p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.config.render_width) /
                                                        self.config.render_height,
                                                        nearVal=self.config.cam_near,
                                                        farVal=self.config.cam_far)
        result = self.world.p.getCameraImage(width=self.config.render_width,
                                                height=self.config.render_height,
                                                viewMatrix=agent_view_matrix,
                                                projectionMatrix=agent_proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(result[2]).reshape(result[1], result[0], -1)[:, :, :3]
        z_buffer = np.array(result[3]).reshape(result[1], result[0])
        segmentation_mask = np.array(result[4]).reshape(result[1], result[0])
        depth = (
            1.0 * self.config.cam_far * self.config.cam_near / (self.config.cam_far - (self.config.cam_far - self.config.cam_near) * z_buffer)
        )
        ## DEBUG
        cv2.imwrite("agentview_rgb.png", rgb)
        renders["agentview"] = {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "segmentation": segmentation_mask
        }
        # ## DEBUG
        # color_image = visualize_segmentation_mask(segmentation_mask)
        # cv2.imwrite("instance_mask.png", color_image)

        if self.use_pcd or self.use_part_pcd_gt:
            table_seg_mask = (renders["agentview"]["segmentation"] == 1) | (renders["agentview"]["segmentation"] == 0) | (renders["agentview"]["segmentation"] == -1)
            depth_filtered = apply_segmentation_mask(depth, ~table_seg_mask)
            valid_pixels = np.where(~table_seg_mask)
            pixels_y_filtered = valid_pixels[0]
            pixels_x_filtered = valid_pixels[1]
            original_indices = pixels_y_filtered * self.config.render_width + pixels_x_filtered
            self.raw_indices = original_indices
            
            point_cloud, indices = self._depth_to_pcd(depth_filtered, self.record_camera.intrinsic.K, agent_view=True)
            self.indices = indices.cpu().numpy()
            # pcd_points_np = np.asarray(point_cloud.points)
            scene_pcd = point_cloud[self.indices].squeeze(0)
            renders["agentview"]["pcd"] = scene_pcd
            # np.save("scene_pcd_filtered.npy", scene_pcd)
            # raise ValueError("Manual breakpoint here")
        if self.use_part_pcd_gt or self.use_mask_gt:
            if not self.cur_target_part:
                part_pcd = np.zeros((self.pcd_size, 3))
            else:
                print("cur_target_part", self.cur_target_part)
                part_pcd = self.parser.get_one_part_pcd(self.cur_target_part) # world frame

            if self.use_part_pcd_gt:
                part_mask = project_point_cloud_to_mask(part_pcd, self.config.render_width, self.config.render_height, agent_view_matrix, agent_proj_matrix)
                pixels_y, pixels_x = np.where(part_mask == 1)
                pcd_indices = pixels_y * self.config.render_width + pixels_x
                intersection_elements = np.intersect1d(self.raw_indices, pcd_indices)
                indices_in_raw = np.searchsorted(self.raw_indices, intersection_elements)
                part_indices = np.intersect1d(indices_in_raw, self.indices)
                part_channel = np.expand_dims(np.zeros(self.indices.shape[1]), axis=1)
                boolean_mask = np.isin(self.indices, part_indices).transpose(1,0)
                part_channel[boolean_mask] = 1
                scene_pcd_with_part = np.hstack((scene_pcd, part_channel))
                # np.save("scene_pcd_filtered.npy", scene_pcd_with_part)
                # print("Vised !!")
                renders["agentview"]["part"] = scene_pcd_with_part
                # np.save("scene_pcd_with_part.npy", scene_pcd_with_part)

                # visualize_and_save_two_point_clouds(renders["agentview"]["pcd"], scene_part_pcd, [1, 0, 0], [0, 1, 0], save_path="scene_pcd.ply", visualize=False)
                # print("Vised !!")

            if self.use_mask_gt or self.use_part_pcd_gt:
                # depth_to_pcd() (pixel, pcd_index)
                part_mask = project_point_cloud_to_mask(part_pcd, self.config.render_width, self.config.render_height, agent_view_matrix, agent_proj_matrix)
                renders["agentview"]["mask"] = part_mask
                ## DEBUG
                # color_image = visualize_segmentation_mask(part_mask)
                # cv2.imwrite("part_mask.png", color_image)
                
            # ## DEBUG
            # visualize_and_save_two_point_clouds(dsp_pcd, renders["agentview"]["pcd"], save_path="combined_pcd.ply", visualize=False)
            
        if self.use_wrist_camera:
            wrist_rgb, wrist_depth, wrist_segmentation = self.robot.wrist_camera.render(extrinsic=self.robot.wrist_camera.extrinsic, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            renders["wrist"] = {
                "rgb": wrist_rgb,
                "depth": wrist_depth,
                "segmentation": wrist_segmentation
            }
            # if self.use_pcd:
            #     # Convert extrinsic to view matrix for wrist camera
            #     gl_view_matrix = self.robot.wrist_camera.extrinsic.as_matrix()
            #     gl_view_matrix[2, :] *= -1  # flip the Z axis
            #     wrist_view_matrix = gl_view_matrix.flatten(order="F")
            #     extrinsic = np.array(wrist_view_matrix).reshape(4, 4, order='F')
            #     extrinsic[2, :]*= -1
            #     wrist_proj_matrix = self.robot.wrist_camera.proj_matrix
            #     wrist_pcd = self._depth_to_pcd(wrist_depth, self.robot.wrist_camera.intrinsic.K)
            #     renders["wrist"]["pcd"] = wrist_pcd

            # if self.use_part_pcd_gt or self.use_mask_gt:
            #     if not self.cur_target_part:
            #         wrist_part_pcd = np.zeros((3, self.observation_space["wrist_pcd"].shape[1]))
            #     else:
            #         wrist_part_pcd = self.parser.get_one_part_pcd(self.cur_target_part) # world frame
            #     wrist_dsp_pcd = resample_pcd(wrist_part_pcd, self.observation_space["wrist_pcd"].shape[1])

            #     wrist_extrinsic = self.robot.wrist_camera.extrinsic.as_matrix()
            #     wrist_view_tranform = Transform.from_matrix(wrist_extrinsic)

            #     wrist_dsp_pcd = transform_point_cloud(wrist_dsp_pcd, wrist_view_tranform.translation, wrist_view_tranform.rotation.as_quat())
            #     if self.use_part_pcd_gt:
            #         renders["wrist"]["part"] = wrist_dsp_pcd
            #     if self.use_mask_gt:
            #         wrist_part_mask = project_point_cloud_to_mask(wrist_part_pcd, self.config.render_width, self.config.render_height, wrist_view_matrix, wrist_proj_matrix)
            #         renders["wrist"]["mask"] = wrist_part_mask
                
                # ## DEBUG
                # visualize_and_save_two_point_clouds(wrist_dsp_pcd, renders["wrist"]["pcd"], save_path="combined_pcd_wrist.ply", visualize=False)

        self.renders = renders
        renders_save = copy.deepcopy(renders)
        if self.use_pcd:
            del renders_save["agentview"]["pcd"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["pcd"]
        if self.use_part_pcd_gt:
            del renders_save["agentview"]["part"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["part"]
        if self.use_mask_gt:
            del renders_save["agentview"]["mask"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["mask"]
        if self.record:
            self.render_sequence_buffer.append(renders_save)
        return renders

    def get_current_renders(self):
        return self.renders

    def close(self):
        # Clean up PyBullet connection
        self.world.close()

    def _downsample_pcd(self, pcd, downsample_size):
        pcd = pcd.reshape(-1, pcd.shape[-1])
        _, indices = torch3d_ops.sample_farthest_points(points=pcd.unsqueeze(0), K=downsample_size)
        return indices

    def pad_point_cloud(self, pcd, target_size):
        current_size = pcd.shape[0]
        if current_size < target_size:
            # Get the last point of the current point cloud
            last_point = pcd[-1, :]
            padding = np.tile(last_point, (target_size - current_size, 1))
            pcd = np.vstack((pcd, padding))
        return pcd

    def _depth_to_pcd(self, depth_image, intrinsic_matrix, depth_scale=1.0, depth_trunc=3.0, height=300, width=300, downsample_size=1024, agent_view=False):
        # Convert depth image to Open3D format
        depth_o3d = o3d.geometry.Image(depth_image)

        # Create an Open3D camera intrinsic object
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.intrinsic_matrix = intrinsic_matrix
        
        # Convert depth image to point cloud
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            camera_intrinsics,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )
        pcd = np.asarray(point_cloud.points)
        pcd_np = pcd.reshape(-1, pcd.shape[-1])
        pcd = torch.tensor(pcd_np, dtype=torch.float32).to(self.device)
        indices = self._downsample_pcd(pcd, downsample_size)
        return pcd_np, indices

    def _take_action(self, action, gain=0.01, gain_gripper=0.001, gain_near_target=0.01):
        self.robot.apply_action(action, gain, gain_gripper, gain_near_target=gain_near_target)
        self.world.step()

    def _get_instruction(self):
        if self.cur_skill == "grasp_obj":
            region_str =""
            instruction = f"Grasp the {self.obj_class} at {region_str} its {self.cur_skill_params['part_grasp']}"
        elif self.cur_skill == "rotate_obj":
            dir_str = mapping_rotate(self.cur_skill_params['dir_rotate'])
            instruction = f"Reorient the {self.cur_skill_params['part_rotate']} of the {self.obj_class} to face {dir_str}"
        elif self.cur_skill == "move_gripper":
            dir_str = mapping_translate(self.cur_skill_params['dir_move'])
            instruction = f"Move {dir_str}"
        elif self.cur_skill == "touch_obj":
            region_str = ""
            instruction = f"Touch the {self.obj_class} at {region_str} its {self.cur_skill_params['part_touch']}"
        elif self.cur_skill == "release_obj":
            instruction = "Release"
        else:
            instruction = ""

        assert isinstance(lang_encoder, nn.Module)
        tokenized = lang_encoder.tokenize(instruction)

        return instruction, tokenized

    def _get_observation(self):
        renders = self.render_all()
        frame_input = renders["agentview"]["rgb"]
        agentview_rgb = frame_input.transpose((2, 0, 1))
        # import ipdb; ipdb.set_trace()
        mask_np = None

        if self.num_steps == 0:
            if self.cur_target_part:
                result = self.set_sam(frame_input, self.obj_class, self.cur_target_part)
                if isinstance(result, tuple):
                    self.grounding_success, mask_np = result
                else:
                    self.grounding_success = result
                    mask_np = None
                if self.grounding_success:
                    self.grounding_success_num += 1
        else:
            if self.grounding_success and self.cur_target_part:
                out_obj_ids, out_mask_logits = self.sam2_video_predictor.track(frame_input)
                frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
                # plt.figure(figsize=(12, 8))
                # plt.imshow(frame)
                mask_np = show_mask(
                    (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
                )
                # color_image = visualize_segmentation_mask(mask_np)
                # cv2.imwrite(f"/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/sam_test/part_mask_{self.idx}_frame_{self.num_steps}.png", color_image)
                # Save each visualization to disk instead of showing
                # plt.savefig(f"/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/sam_test/env_{self.idx}_frame_{self.num_steps}.png")
                plt.close()
            

        # Get the current TCP position and orientation
        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        tcp_position = np.array(tcp_position)  # Convert translation to numpy array
        tcp_orientation = np.array(tcp_orientation)  # Convert quaternion to numpy array

        # Combine the position and orientation into a single array
        tcp_pose_combined = np.concatenate((tcp_orientation, tcp_position))

        # Get the current joint states
        joint_states = self.robot.get_joint_states()
        gripper_state = self.robot.get_gripper_state()

        # Construct the observation dictionary
        observation = {
            "agentview_rgb": agentview_rgb,
            "tcp_pose": tcp_pose_combined,
            "joint_states": np.array(joint_states),
            "gripper_state": np.array(gripper_state).reshape(1,)
        }
        if self.use_pcd:
            agentview_pcd = renders["agentview"]["pcd"]
            # print("agentview_pcd", agentview_pcd.shape)
            observation["agentview_pcd"] = agentview_pcd

        if self.use_mask_gt or self.use_part_pcd_gt:
            gt_mask = renders["agentview"]["mask"]
            observation["agentview_mask"] = np.zeros_like(gt_mask)
            print("self.grounding_success", self.grounding_success)
            if gt_mask is None or mask_np is None:
                iou = -1
                accuracy = None
            else:
                iou, accuracy = calculate_iou(gt_mask, mask_np)
            print("HERE ####")

            if self.grounding_success and self.cur_target_part and iou > 0:
                # color_image = visualize_segmentation_mask(gt_mask)
                # cv2.imwrite(f"/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/sam_test/dt_part_mask_{self.idx}_frame_{self.num_steps}.png", color_image)
                observation["agentview_mask"] = mask_np
                if self.use_part_pcd_gt:
                    part_pcd_mask = mask_np
                    pixels_y, pixels_x = np.where(part_pcd_mask == 1)
                    pcd_indices = pixels_y * self.config.render_width + pixels_x
                    intersection_elements = np.intersect1d(self.raw_indices, pcd_indices)
                    indices_in_raw = np.searchsorted(self.raw_indices, intersection_elements)
                    part_indices = np.intersect1d(indices_in_raw, self.indices)
                    part_channel = np.expand_dims(np.zeros(self.indices.shape[1]), axis=1)
                    boolean_mask = np.isin(self.indices, part_indices).transpose(1,0)
                    part_channel[boolean_mask] = 1
                    scene_pcd_with_part = np.hstack((agentview_pcd, part_channel))
                    # np.save("/scratch/tshu2/yyin34/projects/lgpm/lgm_baselines/TEST_sam/scene_pcd_filtered.npy", scene_pcd_with_part)
                    # print("Vised !!")
                    # import ipdb; ipdb.set_trace()
                    observation["agentview_part_pcd"] = scene_pcd_with_part
                self.iou.append(iou)
                self.accuracy.append(accuracy)

            elif self.use_part_pcd_gt:
                agentview_part = renders["agentview"]["part"] ### renders["agentview"]["pcd"]
                agentview_part[:, 3] = 0
                observation["agentview_part_pcd"] = agentview_part
            
            if self.use_part_pcd_gt:
                agentview_part = renders["agentview"]["part"] ### renders["agentview"]["pcd"]
                observation["agentview_part_pcd"] = agentview_part
        # Get instruction
        if self.use_language:
            assert self.tokenized_instruction
            observation.update(self.tokenized_instruction)
            # observation["instruction"] = self.instruction

        # Get wrist camera
        if self.use_wrist_camera:
            wrist_rgb = renders["wrist"]["rgb"]
            wrist_rgb = wrist_rgb.transpose((2, 0, 1))
            observation["wrist_rgb"] = wrist_rgb
            # if self.use_pcd:
            #     wrist_pcd = renders["wrist"]["pcd"]
            #     # wrist_pcd = wrist_pcd.transpose((2, 0, 1))
            #     observation["wrist_pcd"] = wrist_pcd
            # if self.use_part_pcd_gt:
            #     observation["wrist_part"] = renders["wrist"]["part"]
            # if self.use_mask_gt:
            #     observation["wrist_mask"] = renders["wrist"]["mask"]

        return observation
    
    def _calculate_reward(self):
        # Calculate and return the reward for the current state
        reward = -1
        current_obj_pose = self.current_state["obj_pose"]
        last_obj_pose = self.last_state["obj_pose"]
        current_tcp_pose = self.current_state["tcp_pose"]
        last_tcp_pose = self.last_state["tcp_pose"]
        current_gripper_tip_pose = self.current_state["tcp_pose"]*Transform(Rotation.identity(), np.array([0, 0, self.robot.finger_depth+self.robot.finger_depth/5]))
        last_gripper_tip_pose = self.last_state["tcp_pose"]*Transform(Rotation.identity(), np.array([0, 0, self.robot.finger_depth+self.robot.finger_depth/5]))
        current_gripper_state = self.current_state["gripper_state"]
        last_gripper_state = self.last_state["gripper_state"]

        if self.cur_skill == "grasp_obj":
            dis_last = np.linalg.norm(self.target_grasp_bbox_center-last_tcp_pose.translation)
            dis_current = np.linalg.norm(self.target_grasp_bbox_center-current_tcp_pose.translation)
            if dis_current<dis_last:
                reward += 2
            else:
                reward += -1
            is_inside = np.all(current_tcp_pose.translation >= self.target_grasp_expanded_bbox[0]) and \
                        np.all(current_tcp_pose.translation <= self.target_grasp_expanded_bbox[1])
            if is_inside:
                reward += 10
            if is_inside and self.robot.detect_grasp_contact():
                reward += 1000

        elif self.cur_skill == "move_gripper":
            dis_last = np.linalg.norm(self.target_translation-last_tcp_pose.translation)
            dis_current = np.linalg.norm(self.target_translation-current_tcp_pose.translation)
            if dis_current<dis_last:
                reward += 2
            else:
                reward += -1
            displacement = current_tcp_pose.translation-last_tcp_pose.translation
            direction = displacement/np.linalg.norm(displacement)
            if np.linalg.norm(direction-self.target_translation_direction)<self.config.translate_dir_norm_dis_thred:
                reward += 10
            if dis_current<self.config.translate_target_dis_thred:
                reward += 1000

        elif self.cur_skill == "rotate_obj":
            dis_last = quaternion_distance(self.target_rotation.as_quat(), last_obj_pose.rotation.as_quat())
            dis_current = quaternion_distance(self.target_rotation.as_quat(), current_obj_pose.rotation.as_quat())
            if dis_current<dis_last:
                reward += 2
            else:
                reward += -1
            if dis_current<self.config.rotate_target_dis_thred:
                reward += 1000

        elif self.cur_skill == "touch_obj":
            dis_last = np.linalg.norm(self.target_touch_bbox_center-last_gripper_tip_pose.translation)
            dis_current = np.linalg.norm(self.target_touch_bbox_center-current_gripper_tip_pose.translation)
            if dis_current<dis_last:
                reward += 2
            else:
                reward += -1
            is_inside = np.all(current_gripper_tip_pose.translation >= self.target_touch_expanded_bbox[0]) and \
                        np.all(current_gripper_tip_pose.translation <= self.target_touch_expanded_bbox[1])
            if is_inside:
                reward += 1000
        
        elif self.cur_skill == "release_obj":
            dis_last = np.linalg.norm(self.release_new_tcp_pose.translation-last_tcp_pose.translation)
            dis_current = np.linalg.norm(self.release_new_tcp_pose.translation-current_tcp_pose.translation)
            if dis_current<dis_last:
                reward += 2
            else:
                reward += -1

            release_target_dis_thred = self.config.release_target_dis_thred
            release_target_rot_thred = self.config.release_target_rot_thred
            if (abs(current_gripper_state-self.robot.GRIPPER_OPEN_JOINT_POS)<0.001) and (dis_current<release_target_dis_thred) and quaternion_distance(self.release_new_tcp_pose.rotation.as_quat(), current_tcp_pose.rotation.as_quat())<release_target_rot_thred:
                reward += 1000
               
        return reward

    def _check_if_done(self):
        # Determine if the episode is done
        done_cur_skill = False
        done = False

        cur_skill_params = self.cur_skill_params.copy()
        if self.cur_skill == "move_gripper":
            last_gripper_position = self.pre_move_tcp_pose.translation
            cur_skill_params.update({"last_gripper_position": last_gripper_position})
            cur_skill_params.update({"distance": self.config.translate_distance})
        from lgplm_sim.bullet_planner import BulletPlanner
        effects = getattr(BulletPlanner, f"effects_{self.cur_skill}")(**cur_skill_params)
        # print(f"effects_{self.cur_skill}")
        # print(effects)
        # print("self.cur_skill")
        # print(self.cur_skill)
        # print("cur_skill_params")
        # print(cur_skill_params)
        
        effects_satisfied = True
        for predicate, items in effects.items():
            # print(predicate)
            result = getattr(BulletPlanner, predicate)(self, **items["params"])
            # print(result)
            if result != items["value"]:
                effects_satisfied = False
                # break
        if effects_satisfied:
            done_cur_skill = True
            if self.cur_skill_idx >= len(self.chain_params)-1:
                done = True

        return done_cur_skill, done
    def _check_if_done_test_eval(self):
        # Determine if the episode is done
        done = False

        from lgplm_sim.bullet_planner_task_val import BulletPlanner
        assert self.task_type
        print(self.task_type)
        effects = getattr(BulletPlanner, f"effects_type{self.task_type}")(last_gripper_position=self.initial_obj_pose.translation, 
                                                                    distance = self.config.translate_distance,
                                                                    chain_params = self.actual_chain_params)
        cnt=0
        if self.two_phases:
            effects_list = copy.deepcopy(effects)
            effects=effects_list[self.task_phase]
            total_predicates = len(effects_list[0].keys())+len(effects_list[1].keys())
            if self.task_phase==1:
                cnt=len(effects_list[0].keys())
        else:
            total_predicates = len(effects.keys())
            
        # print(f"effects_{self.cur_skill}")
        # print(effects)
        # print("self.cur_skill")
        # print(self.cur_skill)
        # print("cur_skill_params")
        # print(cur_skill_params)
        
        effects_satisfied = True
        
        for predicate, items in effects.items():
            # print(predicate)
            result = getattr(BulletPlanner, predicate)(self, **items["params"])
            # print(result)
            if result != items["value"]:
                effects_satisfied = False
            else:
                cnt+=1
        self.completion_rate = cnt/total_predicates
        if self.two_phases and effects_satisfied and self.task_phase==0:
            self.task_phase+=1
        # print("effects_satisfied")
        # print(effects_satisfied)
        if effects_satisfied:
            done = True

        return done

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
    
    def find_episode_info(self, dataset_meta, obj_class, target_ep_id):
        for test_key, test_value in dataset_meta[obj_class].items():
            for episode_list in test_value.values():
                for episode_info in episode_list:
                    if episode_info["ep_id"] == target_ep_id:
                        # print("")
                        return episode_info
        return None
    
    def sample_chain_params(self):
        with open(self.dataset_meta_path, 'r') as file:
            dataset_meta = json.load(file)
        if not self.track_samples:
            self.seed()
        chain_params_list = dataset_meta[self.obj_class][self.split][self.skill_chain]
        episode_info = self.np_random.choice(chain_params_list)
        # episode_info = self.find_episode_info(dataset_meta, self.obj_class, "1468")

        # episode_info = chain_params_list[2]
        # episode_info = next((params for params in chain_params_list if params["obj_id"] == 3618), None)
        self.obj_id = episode_info["obj_id"]
        print("self.obj_id", self.obj_id)
        self.ep_id = episode_info["ep_id"]
        print("self.ep_id", self.ep_id)
        self.task_instruction = episode_info["task_instruction"]
        # self.task_instruction = "Take hold of the right of the bucket."
        print(self.task_instruction)
        self.gemini_skill_chain, object_name, self.gemini_skill_instructions = self.gemini_inference.gemini_infer_task(self.task_instruction)
        self.chain_params = self.gemini_skill_chain
        self.actual_chain_params = episode_info["chain_params"]
        print(self.chain_params)
        self.obj_init_position = episode_info["obj_pose"][4:]
        self.obj_init_position[-1] = 0.2
        self.obj_init_orientation = episode_info["obj_pose"][:4]
        self.obj_scale = episode_info["obj_scale"]
    def get_task_instrution(self):
        return self.task_instruction

    def save_renders(self, video_path):
        assert self.record
        rgbs = [renders["agentview"]["rgb"] for renders in self.render_sequence]
        depths = [renders["agentview"]["depth"] for renders in self.render_sequence]
        segmentations = [renders["agentview"]["segmentation"] for renders in self.render_sequence]

        if self.use_wrist_camera:
            wrist_rgbs = [renders["wrist"]["rgb"] for renders in self.render_sequence]
            wrist_depths = [renders["wrist"]["depth"] for renders in self.render_sequence]
            wrist_segmentations = [renders["wrist"]["segmentation"] for renders in self.render_sequence]

        video_dir, filename = os.path.split(video_path)
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        name, _ = os.path.splitext(filename)
        wrist_name = name+"_wrist"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        out_rgb = cv2.VideoWriter(os.path.join(video_dir, name+'.mp4'), fourcc, self.control_hz, (self.config.render_width,self.config.render_height))
        
        # Save RGB video
        for frame in rgbs:
            out_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_rgb.release()
        print(f"RGB video saved at {os.path.join(video_dir, name + '.mp4')}")

        if self.use_wrist_camera:
            out_wrist_rgb = cv2.VideoWriter(os.path.join(video_dir, wrist_name+'.mp4'), fourcc, self.control_hz, (self.robot.wrist_camera.intrinsic.width,self.robot.wrist_camera.intrinsic.height))
            # Save RGB video
            for frame in wrist_rgbs:
                out_wrist_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out_wrist_rgb.release()

        with ThreadPoolExecutor() as executor:
            depth_dir = os.path.join(video_dir, name + '_depth')
            os.makedirs(depth_dir, exist_ok=True)
            for i, depth_frame in enumerate(depths):
                depth_image_path = os.path.join(depth_dir, f'depth_{i:04d}.png')
                executor.submit(_save_depth, depth_image_path, depth_frame)

            segmentation_dir = os.path.join(video_dir, name + '_segmentation')
            os.makedirs(segmentation_dir, exist_ok=True)
            for i, segmentation_frame in enumerate(segmentations):
                segmentation_image_path = os.path.join(segmentation_dir, f'segmentation_{i:04d}.png')
                executor.submit(_save_image, segmentation_image_path, segmentation_frame)

            if self.use_wrist_camera:
                depth_dir = os.path.join(video_dir, wrist_name + '_depth')
                os.makedirs(depth_dir, exist_ok=True)
                for i, depth_frame in enumerate(wrist_depths):
                    depth_image_path = os.path.join(depth_dir, f'depth_{i:04d}.png')
                    executor.submit(_save_depth, depth_image_path, depth_frame)

                segmentation_dir = os.path.join(video_dir, wrist_name + '_segmentation')
                os.makedirs(segmentation_dir, exist_ok=True)
                for i, segmentation_frame in enumerate(wrist_segmentations):
                    segmentation_image_path = os.path.join(segmentation_dir, f'segmentation_{i:04d}.png')
                    executor.submit(_save_image, segmentation_image_path, segmentation_frame)

    def save_states(self, state_path):
        assert self.record
        with open(state_path, 'w') as json_file:
            json.dump(self.state_sequence, json_file, indent=4)

    def execute_skill(self, skill_name, params):
        """Execute a skill with the given parameters."""
        self.first_frame_av = self.save_frame_as_base64()
        self.first_tcp_pose = list(self.current_state["tcp_pose"].to_list())
        self.gripper_state = self.robot.get_gripper_state()
        
        # Execute the skill
        if skill_name == "grasp_obj":
            self.grasp_obj(params["part_grasp"])
        elif skill_name == "move_gripper":
            self.move_gripper(params["dir_move"], params.get("grasping", False), 
                            params.get("put_down", False), params.get("touching", False))
        elif skill_name == "rotate_obj":
            self.rotate_obj(params["dir_rotate"], params["part_rotate"])
        elif skill_name == "touch_obj":
            self.touch_obj(params["part_touch"])
        elif skill_name == "release_obj":
            self.release_obj()
        
        self.last_frame_av = self.save_frame_as_base64()
        self.last_tcp_pose = list(self.current_state["tcp_pose"].to_list())
        self.gripper_state = self.robot.get_gripper_state()
        
        # Add to executed skill chain
        self.executed_skill_chain.append({"skill_name": skill_name, "params": params})
        
        return True

    def check_skill_completion(self, skill_name, params):
        """Check if a skill has been completed successfully."""
        # Get current state
        current_frame = self.save_frame_as_base64()
        current_gripper_state = self.robot.get_gripper_state()
        
        # Use GPT to check completion
        completion, _ = self.gpt_completion_checker(
            self.first_frame_av,
            current_frame,
            current_gripper_state,
            self.first_tcp_pose,
            self.last_tcp_pose
        )
        
        return completion

    def gpt_completion_checker(self, first_frame, last_frame, gripper_state, first_tcp_pose, last_tcp_pose):
        """Use GPT to check if a skill has been completed successfully."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robot skill completion checker. Your job is to determine if a skill has been completed successfully by analyzing the before and after images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                Analyze these two images and determine if the skill has been completed successfully.
                                First image is before the skill execution, second image is after.
                                
                                Additional context:
                                - Gripper state: {gripper_state} (0.04 means open, <0.018 means closed)
                                - First TCP pose: {first_tcp_pose}
                                - Last TCP pose: {last_tcp_pose}
                                
                                Return only 'True' if the skill is completed successfully, 'False' otherwise.
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{first_frame}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{last_frame}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10
            )
            
            completion = response.choices[0].message.content.strip().lower() == "true"
            return completion, None
            
        except Exception as e:
            print(f"Error in GPT completion checker: {str(e)}")
            return False, str(e)

    def save_frame_as_base64(self):
        """Save the current frame as a base64 encoded string."""
        frame = self.get_frame()
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def get_frame(self):
        """Get the current frame from the environment."""
        frame = self.render(mode='rgb_array')
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
