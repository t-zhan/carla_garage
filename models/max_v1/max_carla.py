from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_path))

from torch import nn
import torch
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, PreTrainedModel, AutoProcessor
from team_code.nav_planner import LateralPIDController
from team_code.transfuser_utils import PIDController
from transformers.generation import GenerationMixin
from models.max_v1.config import RegHeadConfig, MaxConfig
import team_code.transfuser_utils as t_u


class RegHead(PreTrainedModel):
    config_class = RegHeadConfig

    def __init__(self, config: RegHeadConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=config.reg_head_bias),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=config.reg_head_bias),
        )
        self.point_decoder = nn.Linear(config.hidden_size, 2, bias=config.reg_head_bias)

    def forward(self, features):
        output = self.dropout(features)
        residual = output
        output = self.norm(output)
        output = self.ffn(output) + residual
        output = self.point_decoder(output)
        return output


class Max(PreTrainedModel, GenerationMixin):
    config_class = MaxConfig

    def __init__(self, config, is_finetuned=False):
        super().__init__(config)

        # 如果config文件中没有qwen_config，使用这两行加载模型
        qwen_config = Qwen2_5_VLConfig.from_pretrained(config.qwen_model_dir)
        self.qwen_2_5_VL = Qwen2_5_VLForConditionalGeneration(qwen_config)

        # 如果保存模型时同时将qwen_config写入config文件，即可使用这一行加载模型
        # self.qwen_2_5_VL = Qwen2_5_VLForConditionalGeneration(config.qwen_config)

        if not is_finetuned:
            self.qwen_2_5_VL = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.qwen_model_dir)
 
        self.point_embed_layer = nn.Linear(2, self.qwen_2_5_VL.config.hidden_size, bias=False, device=self.qwen_2_5_VL.device)
        
        reg_head_config = RegHeadConfig(
            hidden_size=self.qwen_2_5_VL.config.hidden_size,
            pred_len=config.pred_len,
        )
        self.reg_head = RegHead(reg_head_config)

        self.lateral_pid_controller = LateralPIDController(config)
        self.turn_controller = PIDController(k_p=config.turn_kp,
                                             k_i=config.turn_ki,
                                             k_d=config.turn_kd,
                                             n=config.turn_n)
        self.speed_controller = PIDController(k_p=config.speed_kp,
                                              k_i=config.speed_ki,
                                              k_d=config.speed_kd,
                                              n=config.speed_n)
        self.speed_histogram = []
        
        # Redefine the depth channel for 4-channel input, if needed
        # May set default value, whether set the fourth channel
        # self.patch_visual_proj_for_4ch(self.qwen_2_5_VL)

    def forward(self):
        # 处理输入, 调用 Qwen2.5 VL, 输出 pred_waypoints 等
        pass

    def generate(self, input_ids, pixel_values, image_grid_thw):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        current_points = torch.zeros(batch_size, 1, 2, device=device)

        base_inputs_embeds = self._process_embeddings(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        for step in range(self.config.pred_len):
            if not self.config.use_cache or step == 0:
                point_embeds = self.point_embed_layer(current_points)
                base_inputs_embeds = base_inputs_embeds.to(point_embeds.device)
                full_inputs_embeds = torch.cat([base_inputs_embeds, point_embeds], dim=1)
                outputs = self.qwen_2_5_VL.model(inputs_embeds=full_inputs_embeds)
            else:
                point_embeds = self.point_embed_layer(next_point)
                outputs = self.qwen_2_5_VL.model(
                    inputs_embeds=point_embeds,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            if self.config.use_cache:
                past_key_values = outputs.past_key_values
            
            point_hidden_states = outputs.last_hidden_state[:, -1:, :]
            next_point = self.reg_head(point_hidden_states)

            current_points = current_points.to(next_point.device)
            current_points = torch.cat([current_points, next_point], dim=1)

        # return current_points[:, 1:, :]
        return current_points[:, 1:, [1, 0]]

    def carla_generate(self, rgb, command):

        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        rgb_np = rgb_np.clip(0, 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_np)

        command_to_text = ["void", "turn left", "turn right", "go straight", "follow lane", "change lane to left", "change lane to right"]
        command_text = command_to_text[torch.argmax(command).item()]
        
        full_command_text = (
            f"{'You want to ' + command_text + '. ' if command_text != 'void' else ''}"
            "You are a responsible driver, you need to follow the rules of the road and stay safe as efficiently as possible."
            "Every 0.5s, the coordinates are represented by [x, y], where x is the front and y is the left and right direction,"
            "and the trajectory of the future 4s is output in the format [x1, y1], [x2, y2],..., [x8, y8]]."
        )

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_pil},
                    {"type": "text", "text": full_command_text}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[rgb_pil],
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            pred_wp = self.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw
            )
        return pred_wp, None, None, None, None, None, None, None, None, None, command_text  # [pred_len, 2]


    def compute_loss(self, pred_wp, waypoint_label):
        # 计算 wp 损失
        pass

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            model_name_or_path, 
            *model_args, 
            is_finetuned=True, 
            device_map="auto",
            **kwargs
        )
        model.processor = AutoProcessor.from_pretrained(
            model_name_or_path, 
            padding_side="left", 
            use_fast=True
        )
        return model

    def save_pretrained(self, save_directory):
        self.config.qwen_config = self.qwen_2_5_VL.config.to_dict()
        self.config.reg_head_config = self.reg_head.config.to_dict()
        super().save_pretrained(save_directory)

    @classmethod
    def patch_visual_proj_for_4ch(cls, qwen_2_5_VL):
        """
        Replace the visual.patch_embed.proj with a 4-channel Conv3d,
        copy pretrained weights for the first 3 channels, and initialize the 4th channel.
        """
        # Redefine the depth channel for 4-channel input
        old_proj = qwen_2_5_VL.visual.patch_embed.proj
        state_dict = qwen_2_5_VL.state_dict()
        pretrained_weight = state_dict.get("visual.patch_embed.proj.weight")
        out_channels = old_proj.out_channels
        kernel_size = old_proj.kernel_size
        stride = old_proj.stride
        device = pretrained_weight.device if pretrained_weight is not None else old_proj.weight.device
        dtype = pretrained_weight.dtype if pretrained_weight is not None else old_proj.weight.dtype

        # Create new Conv3d
        new_proj = nn.Conv3d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        ).to(device=device, dtype=dtype)

        if pretrained_weight is not None and pretrained_weight.shape[1] >= 3:
            with torch.no_grad():
                # Copy the first 3 channels
                new_proj.weight[:, :3, :, :, :] = pretrained_weight[:, :3, :, :, :]
                # Xavier initialization for the 4th channel
                nn.init.xavier_uniform_(new_proj.weight[:, 3:4, :, :, :])
        else:
            # If no pretrained weight, just use default init
            nn.init.xavier_uniform_(new_proj.weight)

        # Replace the proj layer
        qwen_2_5_VL.visual.patch_embed.in_channels = 4
        qwen_2_5_VL.visual.patch_embed.proj = new_proj

        # Remove the old weight from state_dict to avoid loading conflict
        state_dict.pop("visual.patch_embed.proj.weight", None)
        qwen_2_5_VL.load_state_dict(state_dict, strict=False)

    def _process_embeddings(self, input_ids, pixel_values, image_grid_thw):
        """
        处理文本和图像输入的embedding (删除视频支持) 
        复制Qwen forward中的embedding处理逻辑 (核心部分) 
        
        Args:
            input_ids: 文本token IDs
            pixel_values: 图像像素值
            image_grid_thw: 图像网格信息
        
        Returns:
            torch.Tensor: 处理后的embedding [batch_size, seq_len, hidden_size]
        """
        
        # === 复制Qwen的embedding处理逻辑 (删除视频部分) ===
        if input_ids is None:
            raise ValueError("input_ids must be provided when inputs_embeds is None")
            
        inputs_embeds = self.qwen_2_5_VL.model.embed_tokens(input_ids)
        
        # 处理图像
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.qwen_2_5_VL.visual.dtype)
            image_embeds = self.qwen_2_5_VL.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.qwen_2_5_VL.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.qwen_2_5_VL.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        return inputs_embeds

    def control_pid(self, waypoints, velocity, tuned_aim_distance=False):
        """
        基于 waypoints 生成控制信号(仿照 LidarCenterNet)
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()

        # 计算期望速度(基于 waypoints 的距离和时间)
        one_second = int(self.config.carla_config.carla_fps // (self.config.carla_config.wp_dilation * self.config.carla_config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        
        import os
        self.make_histogram = int(os.environ.get('HISTOGRAM', 0))

        if self.make_histogram:
            self.speed_histogram.append(desired_speed * 3.6)

        # 刹车判断
        brake = ((desired_speed < self.config.carla_config.brake_speed) or ((speed / desired_speed) > self.config.carla_config.brake_ratio))

        # 油门控制
        delta = np.clip(desired_speed - speed, 0.0, self.config.carla_config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.carla_config.clip_throttle)
        throttle = throttle if not brake else 0.0

        # 瞄准距离
        if tuned_aim_distance:
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            if desired_speed < self.config.carla_config.aim_distance_threshold:
                aim_distance = self.config.carla_config.aim_distance_slow
            else:
                aim_distance = self.config.carla_config.aim_distance_fast

        # 选择瞄准 waypoint
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            angle = 0.0
        if brake:
            angle = 0.0

        # 转向控制
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

    def init_visualization(self):
        # Privileged map access for visualization
        if self.config.carla_config.debug:
            # pass
            # Only needed if you want to render the GT map by uncommenting some lines in visualize_model()
            from team_code.birds_eye_view.chauffeurnet import ObsManager  # pylint: disable=locally-disabled, import-outside-toplevel
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
            obs_config = {
                'width_in_pixels': self.config.carla_config.lidar_resolution_width * 4,
                'pixels_ev_to_bottom': self.config.carla_config.lidar_resolution_height / 2.0, # * 4,
                'pixels_per_meter': self.config.carla_config.pixels_per_meter / 2.0, # * 4,
                'history_idx': [-1],
                'scale_bbox': True,
                'scale_mask_col': 4.0,
                'map_folder': 'maps'  # 'maps_high_res'
            }
            self._vehicle = CarlaDataProvider.get_hero_actor()
            self.ss_bev_manager = ObsManager(obs_config, self.config.carla_config)
            self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=None)

    @torch.no_grad()
    def visualize_model(  # pylint: disable=locally-disabled, unused-argument
        self,
        save_path,
        step,
        rgb,
        lidar_bev,
        target_point,
        pred_wp,
        target_point_next=None,
        pred_semantic=None,
        pred_bev_semantic=None,
        pred_depth=None,
        pred_checkpoint=None,
        pred_speed=None,
        pred_target_speed_scalar=None,
        pred_bb=None,
        gt_wp=None,
        gt_checkpoints=None,
        gt_bbs=None,
        gt_speed=None,
        gt_bev_semantic=None,
        wp_selected=None,
        command_text=None):
        # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign, 4 emergency vehicle
        color_classes = [
                np.array([255, 165, 0]),
                np.array([0, 255, 0]),
                np.array([255, 0, 0]),
                np.array([250, 160, 160]),
                np.array([16, 133, 133])
        ]

        size_width = int((self.config.carla_config.max_y - self.config.carla_config.min_y) * self.config.carla_config.pixels_per_meter)
        size_height = int((self.config.carla_config.max_x - self.config.carla_config.min_x) * self.config.carla_config.pixels_per_meter)

        scale_factor = 4
        origin_x_ratio = self.config.carla_config.max_x / (
                self.config.carla_config.max_x -
                self.config.carla_config.min_x) if self.config.carla_config.crop_bev and self.config.carla_config.crop_bev_height_only_from_behind else 1
        origin = ((size_width * scale_factor) // 2, (origin_x_ratio * size_height * scale_factor) // 2)
        loc_pixels_per_meter = self.config.carla_config.pixels_per_meter * scale_factor

        ## add rgb image and lidar
        if self.config.carla_config.use_ground_plane:
            images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
        else:
            images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

        images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
        images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)
        images_lidar = cv2.resize(images_lidar,
                                  dsize=(images_lidar.shape[1] * scale_factor, images_lidar.shape[0] * scale_factor),
                                  interpolation=cv2.INTER_NEAREST)

        # # Uncomment and comment next block to render ground truth map instead of bev prediction.
        # # Render road over image
        # road = self.ss_bev_manager.get_road()
        # # Alpha blending the road over the LiDAR
        # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

        if pred_bev_semantic is not None:
            bev_semantic_indices = np.argmax(pred_bev_semantic[0].detach().cpu().numpy(), axis=0)
            converter = np.array(self.config.carla_config.bev_classes_list)
            converter[1][0:3] = 40
            bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
            alpha = np.ones_like(bev_semantic_indices) * 0.33
            alpha = alpha.astype(np.float32)
            alpha[bev_semantic_indices == 0] = 0.0
            alpha[bev_semantic_indices == 1] = 0.1

            alpha = cv2.resize(alpha,
                               dsize=(alpha.shape[1] * scale_factor, alpha.shape[0] * scale_factor),
                               interpolation=cv2.INTER_NEAREST)
            alpha = np.expand_dims(alpha, 2)
            bev_semantic_image = cv2.resize(bev_semantic_image,
                                            dsize=(bev_semantic_image.shape[1] * scale_factor,
                                                   bev_semantic_image.shape[0] * scale_factor),
                                            interpolation=cv2.INTER_NEAREST)

            images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

        if gt_bev_semantic is not None:
            bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
            converter = np.array(self.config.carla_config.bev_classes_list)
            converter[1][0:3] = 40
            bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
            alpha = np.ones_like(bev_semantic_indices) * 0.33
            alpha = alpha.astype(np.float32)
            alpha[bev_semantic_indices == 0] = 0.0
            alpha[bev_semantic_indices == 1] = 0.1

            alpha = cv2.resize(alpha,
                               dsize=(alpha.shape[1] * scale_factor, alpha.shape[0] * scale_factor),
                               interpolation=cv2.INTER_NEAREST)
            alpha = np.expand_dims(alpha, 2)
            bev_semantic_image = cv2.resize(bev_semantic_image,
                                            dsize=(bev_semantic_image.shape[1] * scale_factor,
                                                   bev_semantic_image.shape[0] * scale_factor),
                                            interpolation=cv2.INTER_NEAREST)
            images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

            images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

        # Draw wps
        # Red ground truth
        if gt_wp is not None:
            gt_wp_color = (255, 255, 0)
            for wp in gt_wp.detach().cpu().numpy()[0]:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

        # Orange ground truth checkpoint
        if gt_checkpoints is not None:
            for wp in gt_checkpoints.detach().cpu().numpy()[0]:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]  # this is where the minus comes from ^
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=8, lineType=cv2.LINE_AA, color=(0, 0, 0), thickness=-1)

        # Green predicted checkpoint
        if pred_checkpoint is not None:
            for wp in pred_checkpoint.detach().cpu().numpy()[0]:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                                     radius=8,
                                     lineType=cv2.LINE_AA,
                                     color=(0, 128, 255),
                                     thickness=-1)

        # Blue predicted wp
        if pred_wp is not None:
            pred_wps = pred_wp.detach().cpu().numpy()[0]
            num_wp = len(pred_wps)
            for idx, wp in enumerate(pred_wps):
                color_weight = 0.5 + 0.5 * float(idx) / num_wp
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                                     radius=8,
                                     lineType=cv2.LINE_AA,
                                     color=(0, 0, int(color_weight * 255)),
                                     thickness=-1)

        # Draw target points
        if self.config.carla_config.use_tp:
            x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
            y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

            # draw next tp too
            if self.config.carla_config.two_tp_input and target_point_next is not None:
                x_tpn = target_point_next[0][0] * loc_pixels_per_meter + origin[0]
                y_tpn = target_point_next[0][1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(x_tpn), int(y_tpn)),
                                     radius=12,
                                     lineType=cv2.LINE_AA,
                                     color=(255, 0, 0),
                                     thickness=-1)

        # draw ego
        sample_box = np.array([
                int(images_lidar.shape[0] / 2),
                int(origin_x_ratio * images_lidar.shape[1] / 2), self.config.carla_config.ego_extent_x * loc_pixels_per_meter,
                self.config.carla_config.ego_extent_y * loc_pixels_per_meter,
                np.deg2rad(90.0), 0.0
        ])
        images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

        if pred_bb is not None:
            for box in pred_bb:
                inv_brake = 1.0 - box[6]
                color_box = deepcopy(color_classes[int(box[7])])
                color_box[1] = color_box[1] * inv_brake
                box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, self.config.carla_config.min_x, self.config.carla_config.min_y)
                images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

        if gt_bbs is not None:
            gt_bbs = gt_bbs.detach().cpu().numpy()[0]
            real_boxes = gt_bbs.sum(axis=-1) != 0.
            gt_bbs = gt_bbs[real_boxes]
            for box in gt_bbs:
                box[:4] = box[:4] * scale_factor
                images_lidar = t_u.draw_box(images_lidar, box, color=(0, 255, 255), pixel_per_meter=loc_pixels_per_meter)

        images_lidar = np.rot90(images_lidar, k=1)
        images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
        rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()

        if wp_selected is not None:
            colors_name = ['blue', 'yellow']
            colors_idx = [(0, 0, 255), (255, 255, 0)]
            cv2.putText(images_lidar, 'Selected: ', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(images_lidar, f'{colors_name[wp_selected]}', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    colors_idx[wp_selected], 2, cv2.LINE_AA)

        if pred_speed is not None:
            pred_speed = pred_speed.detach().cpu().numpy()[0]
            t_u.draw_probability_boxes(images_lidar, pred_speed, self.config.carla_config.target_speeds)

        if gt_speed is not None:
            gt_speed_float = gt_speed[0].detach().cpu().item()
            cv2.putText(images_lidar, f'Speed: {gt_speed_float:.2f}', (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                                    cv2.LINE_AA)

        if pred_target_speed_scalar is not None:
            cv2.putText(images_lidar, f'Pred TS: {pred_target_speed_scalar:.2f}', (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 1, cv2.LINE_AA)
            
        if command_text is not None:
            cv2.putText(images_lidar, f'Command: {command_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        all_images = np.concatenate((rgb_image, images_lidar), axis=0)
        all_images = Image.fromarray(all_images.astype(np.uint8))

        store_path = str(str(save_path) + (f'/{step:04}.png'))
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        all_images.save(store_path)

    def create_optimizer_groups(self, weight_decay):
        """
        将模型参数分为两组:decay(应用权重衰减)和 no_decay(不应用)。
        仿照 LidarCenterNet 的逻辑。
        """
        # 分离参数为 decay 和 no_decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn  # 完整参数名

                if pn.endswith('bias'):
                    # 所有偏置不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and 'conv.' in pn:  # 卷积层衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and '.bn' in pn:  # BatchNorm 不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and '.ln' in pn:  # LayerNorm 不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and 'downsample.0.weight' in pn:  # Conv2D stride 2 衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and 'downsample.1.weight' in pn:  # BN 不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and '.attn' in pn:  # Attention 线性层衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and 'channel_to_' in pn:  # 通道变化卷积层衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and '.mlp' in pn:  # MLP 线性层衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and 'target_speed_network' in pn:  # 目标速度网络衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and 'join.' in pn and not '.norm' in pn:  # MLP 层衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and 'join.' in pn and '.norm' in pn:  # Norm 层不衰减
                    no_decay.add(fpn)
                elif pn.endswith('_ih') or pn.endswith('_hh'):
                    # 循环权重不衰减
                    no_decay.add(fpn)
                elif pn.endswith('_emb') or '_token' in pn:
                    no_decay.add(fpn)
                elif pn.endswith('_embed'):
                    no_decay.add(fpn)
                elif 'bias_ih_l0' in pn or 'bias_hh_l0' in pn:
                    no_decay.add(fpn)
                elif 'weight_ih_l0' in pn or 'weight_hh_l0' in pn:
                    decay.add(fpn)
                elif '_query' in pn or 'weight_hh_l0' in pn:
                    no_decay.add(fpn)
                elif 'valid_bev_pixels' in pn:
                    no_decay.add(fpn)

        # 验证所有参数都被分组
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (len(inter_params) == 0), f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f'parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!'

        # 创建优化器组
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(list(decay))],
                'weight_decay': weight_decay,
            },
            {
                'params': [param_dict[pn] for pn in sorted(list(no_decay))],
                'weight_decay': 0.0,
            },
        ]
        return optim_groups

    