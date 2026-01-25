import torch.nn as nn
import torch.utils.data
import math

import time

DEBUG_PRINT = True
USE_PHYSCENE = True

CLASS_ID_VALS = {
    'vanity': 0,
    'toilet': 1,
    'shower': 2,
    'tub': 3,
    'floor': 4,
    'wall': 5,
    'door': 6,
    'window': 7,
    'empty': 8,
}

CLASS_ID_NAMES = {
    0: 'vanity',
    1: 'toilet',
    2: 'shower',
    3: 'tub',
    4: 'floor',
    5: 'wall',
    6: 'door',
    7: 'window',
    8: 'empty',
}

ARCH_CLASS_IDS = 4, 5, 6, 7, 8 


class GaussianDiffusionPhyScene:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def _p_sample_baseline(self, denoise_fn, data, t, condition, condition_cross, noise_fn, clip_denoised=False, return_pred_xstart=False, 
                           shared_noise=None):
        """
        Baseline sampling (no PhyScene guidance) - uses standard diffusion p_sample logic.
        Accepts shared_noise for fair comparison with PhyScene path.
        """
        model_mean, model_variance, model_log_variance, pred_xstart = self.diffusion.p_mean_variance(
            denoise_fn, data=data, t=t, condition=condition, condition_cross=condition_cross, 
            clip_denoised=clip_denoised, return_pred_xstart=True)
        
        # Use shared noise if provided, otherwise generate new noise
        if shared_noise is not None:
            noise = shared_noise
        else:
            noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))
        
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def _p_sample_physcene_v3(self, denoise_fn, data, t, condition, condition_cross, noise_fn, clip_denoised=False, return_pred_xstart=False, 
                             num_partial=0, partial_room_min=None, partial_room_max=None, shared_noise=None):
        """
        Sample from the model with PhyScene guidance.
        Accepts shared_noise for fair comparison with baseline path.
        """
        model_mean, model_variance, model_log_variance, pred_xstart = self.diffusion.p_mean_variance(denoise_fn, data=data, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        # Use shared noise if provided, otherwise generate new noise
        if shared_noise is not None:
            noise = shared_noise
        else:
            noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))


        # 2. 拦截逻辑：在物理引导之前抹除建筑元素 (t < 20)
        #if t[0] == 0:
        #if t[0] < 200:
        if t[0] < 20:
            if DEBUG_PRINT:
                print(f"\n" + "="*80)
                print(f"DEBUG TOTAL model_mean [Step {int(t[0])}] before filtering")
                print(f"num_partial = {num_partial} | total_slots = {model_mean.shape[1]}")
                print("-" * 80)
                
                # 遍历第一个 Batch 的所有槽位
                for i in range(model_mean.shape[1]):
                    item = model_mean[0, i] # 取出第 i 个物体
                    
                    # 提取坐标 (前3位)
                    pos = item[0:3].tolist()
                    pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

                    size = item[3:6].tolist()
                    size_str = f"({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"
                    
                    angle = item[6:8].tolist()
                    angle_str = f"({angle[0]:.3f}, {angle[1]:.3f})"
                    
                    # 提取类别 Logits (7到16位)
                    class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
                    logits = item[class_range]
                    cid = torch.argmax(logits).item()
                    
                    # 提取 Objectness (由于 D=17, objectness 是最后一位，即索引 16)
                    obj_val = item[self.diffusion.bbox_dim + self.diffusion.class_dim - 1].item()
                    
                    # 判定标记
                    prefix = "[FIXED]" if i < num_partial else "[NEW]  "
                    # 根据源码：<= 0 为有效
                    status = "VALID" if obj_val <= 0 else "EMPTY" 
                    
                    # 打印详细信息
                    print(f"{prefix} Slot {i:02d} | Pos: {pos_str} | Size: {size_str} | Angle: {angle_str}  | ClassVal: {item[class_range][cid]:7.3f} | ClassID: {cid:<2} | Obj: {obj_val:7.3f} | Status: {status}")
                
                print("="*80 + "\n")

            B, total_slots, D = model_mean.shape
            
            # 只有当存在新生成的空间时才执行过滤
            assert num_partial < total_slots
            if True:
                # 获取新生成物体的切片
                new_gen_section = model_mean[:, num_partial:, :]
                # 参考预测的 clean 图像 (pred_xstart) 来识别类别
                new_gen_x0 = pred_xstart[:, num_partial:, :]
                
                # 使用 foreach 确保逻辑 100% 准确
                for b in range(B):
                    for n in range(new_gen_section.shape[1]):
                        # 1. 提取当前槽位的类别 ID
                        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
                        logits = new_gen_x0[b, n, class_range]
                        cid = torch.argmax(logits).item()
                        
                        # 2. 如果是建筑类，执行抹除
                        if cid in ARCH_CLASS_IDS:
                            # 抹除 BBox 坐标 (前 7 位)
                            #new_gen_section[b, n, 0 : self.bbox_dim] = 0.0
                            new_gen_section[b, n, 0] = -0.113
                            new_gen_section[b, n, 1] = -1.0
                            new_gen_section[b, n, 2] = 0.173

                            # 这样 descale 出来的物理尺寸才是 Min_Size (接近 0)
                            new_gen_section[b, n, 3:6] = -1.0
                            
                            # C. 抹除角度 (索引 6:8): 既然 angle_dim=2, 设为 sin=0, cos=1
                            new_gen_section[b, n, 6] = 1.0 # cos
                            new_gen_section[b, n, 7] = 0.0 # sin
                                        
                            # 抹除类别 Logits (设为极小值)
                            new_gen_section[b, n, class_range] = -1.0
                            
                            # 核心：设置 Objectness 为正数使其变为 EMPTY
                            # 对应 D=17, 索引是 16
                            o_idx = self.diffusion.bbox_dim + self.diffusion.class_dim - 1 
                            new_gen_section[b, n, o_idx] = 1.0 
                            
                # 写回修改后的均值
                model_mean[:, num_partial:, :] = new_gen_section

            if DEBUG_PRINT:
                print(f"DEBUG TOTAL model_mean [Step {int(t[0])}] after filtering")
                print(f"num_partial = {num_partial} | total_slots = {model_mean.shape[1]}")
                print("-" * 80)
                
                # 遍历第一个 Batch 的所有槽位
                for i in range(model_mean.shape[1]):
                    item = model_mean[0, i] # 取出第 i 个物体
                    
                    # 提取坐标 (前3位)
                    pos = item[0:3].tolist()
                    pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

                    size = item[3:6].tolist()
                    size_str = f"({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"
                    
                    angle = item[6:8].tolist()
                    angle_str = f"({angle[0]:.3f}, {angle[1]:.3f})"

                    # 提取类别 Logits (7到16位)
                    class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
                    logits = item[class_range]
                    cid = torch.argmax(logits).item()
                    
                    # 提取 Objectness (由于 D=17, objectness 是最后一位，即索引 16)
                    obj_val = item[self.diffusion.bbox_dim + self.diffusion.class_dim - 1].item()
                    
                    # 判定标记
                    prefix = "[FIXED]" if i < num_partial else "[NEW]  "
                    # 根据源码：<= 0 为有效
                    status = "VALID" if obj_val <= 0 else "EMPTY" 
                    
                    # 打印详细信息
                    print(f"{prefix} Slot {i:02d} | Pos: {pos_str} | Size: {size_str} | Angle: {angle_str} | ClassVal: {item[class_range][cid]:7.3f} | ClassID: {cid:<2} | Obj: {obj_val:7.3f} | Status: {status}")
                
                print("="*80 + "\n")

                
        #if True:
        #if t[0] < 100:
        #if t[0] == 0:
        # if t[0] < 10:  # Physics guidance enabled for dual-path comparison
        if False:
            print("guidance on timestep "+str(int(t[0])))

            with torch.enable_grad():
                # Crucial: detach and require_grad to create a new trackable starting point
                # Compute gradient based on the trackable tensor
                #max_tries = 10
                #max_tries = 1
                #max_tries = 10 if t[0] == 0 else 1
                max_tries = 100 if t[0] == 0 else 1
                for trial in range(max_tries): 
                    model_mean_for_grad = model_mean.detach().requires_grad_(True)
                    # use_soft_snap = True
                    # use_hard_snap = t[0] == 0 and (max_tries-trial)<3
                    #use_hard_snap = True
                    #use_hard_snap = False
                    scale = 0.002
                    #scale = 0.1
                    grad_clip = 0.01
                    #grad_clip = 0.05
                    gradient, loss = self._get_simple_iou_gradient_v3(
                        model_mean_for_grad, 
                        room_min=partial_room_min, 
                        room_max=partial_room_max,
                        num_partial=num_partial, # 关键新增输入
                        scale=scale,
                        grad_clip=grad_clip
                    )
                    if gradient is None:
                        break
                    #if num_partial > 0:
                    #    # 前 num_partial 个物体的所有梯度清零，确保它们不动
                    #    gradient[:, :num_partial, :] = 0
                    # Add the steering signal back to the original mean
                    print(f"[{trial+1}/{max_tries}]APPLY guidance to reduce overlap: loss = {loss}")
                    model_mean = model_mean + gradient

            if DEBUG_PRINT:
                print(f"\n" + "="*80)
                print(f"DEBUG TOTAL model_mean [Step {int(t[0])}] after gradient")
                print(f"num_partial = {num_partial} | total_slots = {model_mean.shape[1]}")
                print("-" * 80)
                
                # 遍历第一个 Batch 的所有槽位
                for i in range(model_mean.shape[1]):
                    item = model_mean[0, i] # 取出第 i 个物体
                    
                    # 提取坐标 (前3位)
                    pos = item[0:3].tolist()
                    pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

                    size = item[3:6].tolist()
                    size_str = f"({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"
                    
                    angle = item[6:8].tolist()
                    angle_str = f"({angle[0]:.3f}, {angle[1]:.3f})"

                    # 提取类别 Logits (7到16位)
                    class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
                    logits = item[class_range]
                    cid = torch.argmax(logits).item()
                    
                    # 提取 Objectness (由于 D=17, objectness 是最后一位，即索引 16)
                    obj_val = item[self.diffusion.bbox_dim + self.diffusion.class_dim - 1].item()
                    
                    # 判定标记
                    prefix = "[FIXED]" if i < num_partial else "[NEW]  "
                    # 根据源码：<= 0 为有效
                    status = "VALID" if obj_val <= 0 else "EMPTY" 
                    
                    # 打印详细信息
                    print(f"{prefix} Slot {i:02d} | Pos: {pos_str} | Size: {size_str} | Angle: {angle_str} | ClassVal: {item[class_range][cid]:7.3f} | ClassID: {cid:<2} | Obj: {obj_val:7.3f} | Status: {status}")
                
                print("="*80 + "\n")
        # else:
        #     print("skip guidance on timestep "+str(int(t[0])))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop_complete(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, partial_boxes=None,
                      dual_path_compare=True):
        """
        Complete samples based on partial samples.
        
        Args:
            dual_path_compare: If True, runs two paths (baseline and PhyScene) with shared noise
                               and returns (img_t_baseline, img_t_physcene).
                               If False, only runs PhyScene path and returns img_t.
        
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """
        if DEBUG_PRINT:
                num_partial = partial_boxes.shape[1]
                print(f"\n" + "="*80)
                print(f"DEBUG TOTAL partial_boxes [Step {0}] input")
                print(f"num_partial = {num_partial} | total_slots = {partial_boxes.shape[1]}")
                print("-" * 80)
                
                # 遍历第一个 Batch 的所有槽位
                for i in range(partial_boxes.shape[1]):
                    item = partial_boxes[0, i] # 取出第 i 个物体
                    
                    # 提取坐标 (前3位)
                    pos = item[0:3].tolist()
                    pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

                    size = item[3:6].tolist()
                    size_str = f"({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"

                    angle = item[6:8].tolist()
                    angle_str = f"({angle[0]:.3f}, {angle[1]:.3f})"
                    
                    # 提取类别 Logits (7到16位)
                    class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
                    logits = item[class_range]
                    cid = torch.argmax(logits).item()
                    
                    # 提取 Objectness (由于 D=17, objectness 是最后一位，即索引 16)
                    obj_val = item[self.diffusion.bbox_dim + self.diffusion.class_dim - 1].item()
                    
                    # 判定标记
                    prefix = "[FIXED]" if i < num_partial else "[NEW]  "
                    # 根据源码：<= 0 为有效
                    status = "VALID" if obj_val <= 0 else "EMPTY" 
                    
                    # 打印详细信息
                    print(f"{prefix} Slot {i:02d} | Pos: {pos_str} | Size: {size_str} | Angle: {angle_str}| ClassVal: {item[class_range][cid]:7.3f} | ClassID: {cid:<2} | Obj: {obj_val:7.3f} | Status: {status}")
                
                print("="*80 + "\n")

        assert isinstance(shape, (tuple, list))
        num_partial = partial_boxes.shape[1]
        
        # Generate SHARED initial noise for fair comparison
        initial_noise = noise_fn(size=shape, dtype=torch.float, device=device)
        
        if dual_path_compare:
            # Two paths: baseline (no guidance) and PhyScene (with guidance)
            img_t_baseline = initial_noise.clone()
            img_t_physcene = initial_noise.clone()
            print("[DUAL PATH] Running both baseline and PhyScene paths with shared noise")
        else:
            img_t = initial_noise
        
        if USE_PHYSCENE:
            B = partial_boxes.shape[0]
            partial_world = self.diffusion.descale_to_origin(
                partial_boxes[:, :, 0:3], 
                self.diffusion._centroids_min.to(device), 
                self.diffusion._centroids_max.to(device)
            )  # [B, N, 3]
            
            # 1. 获取有效性掩码 (Objectness)
            o_idx = self.diffusion.bbox_dim + self.diffusion.class_dim - 1
            is_valid_mask = partial_boxes[:, :, o_idx] <= 0 # [B, N]
            
            # 2. 获取类别掩码 (只选取 Wall, Class ID = 5)
            class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
            class_logits = partial_boxes[:, :, class_range] # [B, N, class_dim]
            pred_class_ids = torch.argmax(class_logits, dim=-1) # [B, N]
            is_wall_mask = (pred_class_ids == CLASS_ID_VALS['wall']) # [B, N]
            
            # 3. 合并掩码：必须是有效物体且类别是墙
            wall_filter_mask = is_valid_mask & is_wall_mask  # [B, N]
            
            # 4. 计算每个 batch 的房间 AABB (per-batch bounds)
            # Use masked min/max to compute per-scene bounds
            partial_room_min_list = []
            partial_room_max_list = []
            
            for b in range(B):
                wall_mask_b = wall_filter_mask[b]  # [N]
                valid_mask_b = is_valid_mask[b]    # [N]
                
                if wall_mask_b.any():
                    # Use wall points for this scene
                    wall_points_b = partial_world[b][wall_mask_b]  # [num_walls, 3]
                    room_min_b = wall_points_b.min(dim=0)[0]  # [3]
                    room_max_b = wall_points_b.max(dim=0)[0]  # [3]
                elif valid_mask_b.any():
                    # Fallback: use all valid objects
                    valid_points_b = partial_world[b][valid_mask_b]  # [num_valid, 3]
                    room_min_b = valid_points_b.min(dim=0)[0]  # [3]
                    room_max_b = valid_points_b.max(dim=0)[0]  # [3]
                    print(f"WARNING: Batch {b} has no wall segments! Falling back to all valid objects.")
                else:
                    # Emergency fallback: use entire space
                    room_min_b = torch.tensor([-5.0, -5.0, -5.0], device=device)
                    room_max_b = torch.tensor([5.0, 5.0, 5.0], device=device)
                    print(f"WARNING: Batch {b} has no valid objects! Using default bounds.")
                
                partial_room_min_list.append(room_min_b)
                partial_room_max_list.append(room_max_b)
            
            partial_room_min = torch.stack(partial_room_min_list, dim=0)  # [B, 3]
            partial_room_max = torch.stack(partial_room_max_list, dim=0)  # [B, 3]
            
            total_walls = wall_filter_mask.sum().item()
            print(f"DEBUG: Room AABB calculated per-batch using {total_walls} total wall segments across {B} scenes.")

        for t in reversed(range(0, self.diffusion.num_timesteps if not keep_running else len(self.diffusion.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)

            # Generate SHARED noise for partial boxes re-noising (used by both paths)
            shared_noise_partial = noise_fn(size=partial_boxes.shape, dtype=torch.float, device=device)
            partial_boxes_t = self.diffusion.q_sample(x_start=partial_boxes, t=t_, noise=shared_noise_partial)

            # Generate SHARED noise for sampling step (used by both paths)
            shared_noise_sample = noise_fn(size=shape, dtype=torch.float, device=device)

            if dual_path_compare:
                # ========== PATH 1: Baseline (no PhyScene guidance) ==========
                img_t_baseline_combined = torch.cat([partial_boxes_t, img_t_baseline[:, num_partial:, :]], dim=1).contiguous()
                img_t_baseline = self._p_sample_baseline(
                    denoise_fn=denoise_fn, data=img_t_baseline_combined, t=t_, 
                    condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                    clip_denoised=clip_denoised, return_pred_xstart=False,
                    shared_noise=shared_noise_sample
                )
                
                # ========== PATH 2: PhyScene (with guidance) ==========
                img_t_physcene_combined = torch.cat([partial_boxes_t, img_t_physcene[:, num_partial:, :]], dim=1).contiguous()
                if USE_PHYSCENE:
                    img_t_physcene = self._p_sample_physcene_v3(
                        denoise_fn=denoise_fn, data=img_t_physcene_combined, t=t_, 
                        condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                        clip_denoised=clip_denoised, return_pred_xstart=False,
                        num_partial=num_partial, partial_room_min=partial_room_min, partial_room_max=partial_room_max,
                        shared_noise=shared_noise_sample
                    )
                else:
                    img_t_physcene = self._p_sample_baseline(
                        denoise_fn=denoise_fn, data=img_t_physcene_combined, t=t_, 
                        condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                        clip_denoised=clip_denoised, return_pred_xstart=False,
                        shared_noise=shared_noise_sample
                    )
                
                # At t=0, replace partial boxes with clean originals
                if t == 0:
                    print('last:', t, self.diffusion.num_timesteps, len(self.diffusion.betas))
                    img_t_baseline = torch.cat([partial_boxes, img_t_baseline[:, num_partial:, :]], dim=1).contiguous()
                    img_t_physcene = torch.cat([partial_boxes, img_t_physcene[:, num_partial:, :]], dim=1).contiguous()
            else:
                # Original single-path logic
                img_t = torch.cat([partial_boxes_t, img_t[:, num_partial:, :]], dim=1).contiguous()

                if USE_PHYSCENE:
                    img_t = self._p_sample_physcene_v3(
                        denoise_fn=denoise_fn, data=img_t, t=t_, 
                        condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                        clip_denoised=clip_denoised, return_pred_xstart=False,
                        num_partial=num_partial, partial_room_min=partial_room_min, partial_room_max=partial_room_max,
                        shared_noise=shared_noise_sample
                    )
                else:
                    img_t = self.diffusion.p_sample(
                        denoise_fn=denoise_fn, data=img_t, t=t_, 
                        condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                        clip_denoised=clip_denoised, return_pred_xstart=False
                    )
                    
                if t == 0:
                    print('last:', t, self.diffusion.num_timesteps, len(self.diffusion.betas))
                    img_t = torch.cat([partial_boxes, img_t[:, num_partial:, :]], dim=1).contiguous()

        if dual_path_compare:
            assert img_t_baseline.shape == shape
            assert img_t_physcene.shape == shape
            return img_t_baseline, img_t_physcene
        else:
            assert img_t.shape == shape
            return img_t

    def _get_simple_iou_gradient_v3(self, x, room_min, room_max, num_partial=0,
                               scale=0.002, grad_clip=0.01, padding=0.1):
        device = x.device
        B, N, D = x.shape
        
        # 1. 生成有效性掩码 (针对 D=17 逻辑: <= 0 为 VALID)
        o_idx = self.diffusion.bbox_dim + self.diffusion.class_dim - 1
        assert o_idx == 16
        valid_mask = (x[:, :, o_idx] <= 0).float() # [B, N]

        # --- 新增：识别哪些物体是 floor ---
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        assert self.diffusion.bbox_dim == 8
        assert self.diffusion.bbox_dim + self.diffusion.class_dim == 17
        # 注意：这里我们使用输入 x 的类别 logits 来判定，
        # 因为在 t=0 时，x 的类别分布已经相对确定
        class_logits = x[:, :, class_range] # [B, N, class_dim]
        pred_class_ids = torch.argmax(class_logits, dim=-1) # [B, N]

        is_wall_mask = (pred_class_ids == 5).float()
        is_floor_mask = (pred_class_ids == 4).float()
        is_door_mask = (pred_class_ids == 6).float() # 识别门
        is_not_floor_mask = (1.0 - is_floor_mask)
        # 家具掩码 (ID 0-3: vanity, toilet, shower, tub)
        is_furniture_mask = (pred_class_ids <= 3).float()

        # 2. 转换世界坐标
        #loss_collision = 0

        # atan2 参数顺序为 (y, x) -> (sin, cos)
        angles = torch.atan2(x[:, :, 7], x[:, :, 6])

        pos = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
        raw_size = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))

        # --- 核心改进：双重保险 (清空 EMPTY 物体的物理体积) ---
        # 即使 Slot 49 等 EMPTY 物体的原始值是 0.0 (物理 6 米)，乘以后变为 0
        pos = pos * valid_mask.unsqueeze(-1)
        raw_size = raw_size * valid_mask.unsqueeze(-1)

        half_size = (torch.clamp(raw_size, min=1e-4) / 2.0) + padding
        mins, maxs = pos - half_size, pos + half_size

        loss_collision = torch.zeros(B, device=device)  # Per-batch loss
        use_loss_collision = True
        if use_loss_collision:
            # --- A. 基础碰撞损失 (Collision Loss) ---
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)  # [B, N, N]
            
            mask_valid_pair = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, N, N]
            mask_self = torch.eye(N, device=device).unsqueeze(0)  # [1, N, N]
            is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
            mask_moveable = (is_moveable.unsqueeze(2).bool() | is_moveable.unsqueeze(1).bool()).float()

            # 碰撞掩码：有效对 + 排除自碰 + 至少一个可动 + 双向排除地板
            # Fix: symmetric floor mask - exclude floor on BOTH sides of the pair
            is_not_floor_mask_2d = is_not_floor_mask.unsqueeze(2) * is_not_floor_mask.unsqueeze(1)  # [B, N, N]
            final_collision_mask = mask_valid_pair * (1 - mask_self) * mask_moveable * is_not_floor_mask_2d
            loss_collision = (intersection_vol * final_collision_mask).sum(dim=[1, 2])  # [B] per-scene loss

        # --- B. 门前净空损失 (Furniture vs Door Clearance) ---
        #loss_clearance = torch.tensor(0.0, device=device)
        #use_loss_clearance = True # 开启门口净空检测
        #if use_loss_clearance:
        #    clearance_depth = 0.8 
        #    # 门的法线 (由 Cos, Sin 构成)
        #    door_normals = torch.stack([x[:,:,6], torch.zeros_like(angles), x[:,:,7]], dim=-1)
        #    
        #    # 幽灵盒中心：门中心沿法线偏移
        #    ghost_center = pos + door_normals * (raw_size[:, :, 0:1] / 2.0 + clearance_depth / 2.0)
        #    
        #    # 幽灵盒旋转 AABB
        #    cos_a, sin_a = x[:,:,6].abs().unsqueeze(-1), x[:,:,7].abs().unsqueeze(-1)
        #    g_ex = (clearance_depth / 2.0) * cos_a + (raw_size[:, :, 2:3] / 2.0) * sin_a
        #    g_ez = (clearance_depth / 2.0) * sin_a + (raw_size[:, :, 2:3] / 2.0) * cos_a
        #    ghost_half_size = torch.cat([g_ex, raw_size[:, :, 1:2]/2.0, g_ez], dim=-1)
        #    
        #    g_mins, g_maxs = ghost_center - ghost_half_size, ghost_center + ghost_half_size
        #    
        #    inter_mins_c = torch.max(mins.unsqueeze(2), g_mins.unsqueeze(1))
        #    inter_maxs_c = torch.min(maxs.unsqueeze(2), g_maxs.unsqueeze(1))
        #    intersection_vol_c = torch.clamp(inter_maxs_c - inter_mins_c, min=0.0).prod(dim=-1)
        #    
        #    # 掩码：有效家具 vs 有效门
        #    clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
        #    loss_clearance = (intersection_vol_c * clearance_mask).sum()

        # --- B. 2D 门前净空损失 (XZ-Plane Only) ---
        loss_clearance = torch.zeros(B, device=device)  # Per-batch loss
        use_loss_clearance = False 
        
        if use_loss_clearance:
            clearance_depth = 0.8  # 门前后各 0.8m 的净空
            
            # 1. 2D 基础数据
            d_pos_2d = pos[:, :, [0, 2]]  # [B, N, 2] (X, Z轴)
            d_horiz_sizes = raw_size[:, :, [0, 2]]
            
            # 幽灵盒中心直接设为门中心，简化逻辑
            # 这里定义 ghost_center_2d 确保后续 Debug 可访问
            ghost_center_2d = d_pos_2d 
            
            # 2. 动态识别门的最薄轴（厚度）
            d_thick_axis = torch.argmin(d_horiz_sizes, dim=-1)
            cos_a = x[:, :, 6].abs().unsqueeze(-1)
            sin_a = x[:, :, 7].abs().unsqueeze(-1)
            
            d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
            d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)
            
            # 3. 计算 2D 幽灵盒的 AABB 投影半径
            # 总厚度 = 门厚 + 前后 1.6m
            total_g_depth = d_thick + clearance_depth * 2.0
            
            # AABB 投影包络
            g_ex = (total_g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
            g_ez = (total_g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
            g_half_size_2d = torch.cat([g_ex, g_ez], dim=-1)
            
            # 4. 计算 2D 碰撞 (XZ 平面)
            f_mins_2d = pos[:, :, [0, 2]] - half_size[:, :, [0, 2]]
            f_maxs_2d = pos[:, :, [0, 2]] + half_size[:, :, [0, 2]]
            
            g_mins_2d = ghost_center_2d - g_half_size_2d
            g_maxs_2d = ghost_center_2d + g_half_size_2d

            inter_mins_2d = torch.max(f_mins_2d.unsqueeze(2), g_mins_2d.unsqueeze(1))
            inter_maxs_2d = torch.min(f_maxs_2d.unsqueeze(2), g_maxs_2d.unsqueeze(1))
            
            overlap_2d = torch.clamp(inter_maxs_2d - inter_mins_2d, min=0.0)
            intersection_area = overlap_2d[..., 0] * overlap_2d[..., 1] # [B, Nf, Nd]
            
            clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
            loss_clearance = (intersection_area * clearance_mask).sum(dim=[1, 2])  # [B] per-scene loss

            # --- Debug 逻辑放在 if 内部，防止 NameError ---
            if DEBUG_PRINT and loss_clearance.item() == 0:
                d_idx = torch.where(is_door_mask[0] > 0.5)[0]
                f_idx = torch.where(is_furniture_mask[0] > 0.5)[0]
                if len(d_idx) > 0 and len(f_idx) > 0:
                    di, fi = d_idx[0], f_idx[0]
                    # 计算两个矩形在 XZ 平面上的中心距离
                    dist_2d = torch.norm(pos[0, fi, [0, 2]] - ghost_center_2d[0, di])
                    print(f"[Debug Clearance 2D] Door {di.item()} vs Furn {fi.item()} | Dist: {dist_2d.item():.3f}m")
                    print(f"    - Overlap X: {overlap_2d[0, fi, di, 0].item():.4f}")
                    print(f"    - Overlap Z: {overlap_2d[0, fi, di, 1].item():.4f}")

        # 4. 房间边界惩罚 (只对 NEW & VALID 物体)
        #loss_boundary = 0
        #if room_min is not None and room_max is not None:
        assert room_min is not None and room_max is not None
        loss_boundary = torch.zeros(B, device=device)  # Per-batch loss
        boundary_mask = valid_mask * is_moveable * is_not_floor_mask
        use_loss_boundary = False
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            # room_min/room_max are now [B, 3], need to unsqueeze to [B, 1, 3] for broadcasting
            room_min_expanded = room_min.to(device).unsqueeze(1)  # [B, 1, 3]
            room_max_expanded = room_max.to(device).unsqueeze(1)  # [B, 1, 3]
            dist_min = torch.clamp(room_min_expanded - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max_expanded, min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum(dim=1)  # [B] per-scene loss

        loss_attach = torch.zeros(B, device=device)  # Per-batch loss
        snap_offset = torch.zeros_like(pos) 
        
        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        use_soft_snap = False
        use_hard_snap = False
        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0 
            for b in range(B):
                # 提取有效墙体和可移动家具索引
                walls_idx = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
                furs_idx = torch.where((is_furniture_mask[b] * is_moveable[b] * valid_mask[b]) > 0.5)[0]
                
                if len(walls_idx) > 0 and len(furs_idx) > 0:
                    w_pos, w_angle = pos[b, walls_idx], angles[b, walls_idx]
                    f_pos, f_angle = pos[b, furs_idx], angles[b, furs_idx]
                    
                    # 1. 动态识别墙体法线 (Method B)
                    # 识别 X(0) 或 Z(2) 哪个是厚度轴
                    w_horiz_sizes = raw_size[b, walls_idx][:, [0, 2]] # [Nw, 2]
                    thick_axis_idx = torch.argmin(w_horiz_sizes, dim=-1) # [Nw] (0:X, 1:Z)
                    
                    # 构建法线：如果 X 是厚度则为 [cos, 0, sin]；如果 Z 是厚度则为 [-sin, 0, cos]
                    n_if_x_thick = torch.stack([torch.cos(w_angle), torch.zeros_like(w_angle), torch.sin(w_angle)], dim=-1)
                    n_if_z_thick = torch.stack([-torch.sin(w_angle), torch.zeros_like(w_angle), torch.cos(w_angle)], dim=-1)
                    
                    is_z_thick = thick_axis_idx.unsqueeze(-1).float() 
                    normals = (1.0 - is_z_thick) * n_if_x_thick + is_z_thick * n_if_z_thick # [Nw, 3]
                    
                    # 2. 计算家具中心相对于墙体法线的投影间隙
                    rel_pos = f_pos.unsqueeze(1) - w_pos.unsqueeze(0) # [Nf, Nw, 3]
                    c2c_dists = (rel_pos * normals.unsqueeze(0)).sum(dim=-1) # [Nf, Nw]
                    
                    # 墙体真实半厚度 (最小值)
                    w_half_t = w_horiz_sizes.min(dim=-1)[0] / 2.0 # [Nw]
                    # 家具在法线方向的投影半厚度
                    f_thicks = (true_half_size[b, furs_idx].unsqueeze(1) * normals.unsqueeze(0).abs()).sum(dim=-1) # [Nf, Nw]
                    
                    # 3. 计算所有可能的间隙并寻找"物理距离表面最近"的墙
                    all_gaps = torch.abs(c2c_dists) - (f_thicks + w_half_t.unsqueeze(0)) # [Nf, Nw]
                    
                    # 关键改进：使用绝对值寻找最近墙，防止错误的负投影干扰
                    dist_to_surface = torch.abs(all_gaps)
                    _, nearest_wall_ids = dist_to_surface.min(dim=1)
                    # 提取每个家具对应的最近墙的带符号间隙
                    min_gaps = all_gaps.gather(1, nearest_wall_ids.unsqueeze(-1)).squeeze(-1) # [Nf]

                    # --- A. Soft Snap (双向弹簧逻辑) ---
                    if use_soft_snap:
                        # 移除 clamp。Gap > 0 (远) 产生拉力；Gap < 0 (穿) 产生推力
                        loss_attach[b] = loss_attach[b] + min_gaps.pow(2).sum()  # Per-scene loss
                        
                        if DEBUG_PRINT:
                            for idx, f_i in enumerate(furs_idx):
                                g = min_gaps[idx].item()
                                #if abs(g) > 0.05:
                                if abs(g) > 0.0001:
                                    status = "PENETRATING" if g < 0 else "TOO FAR"
                                    print(f"[Debug Gap] Batch {b} Slot {f_i.item()} {status}: Gap={g:.4f}m")

                    # --- B. Hard Snap (精确校正) ---
                    if use_hard_snap:
                        # 只有在非常接近墙时才执行 Hard Snap
                        snap_mask = dist_to_surface.min(dim=1)[0] < 0.15 
                        #snap_mask = dist_to_surface.min(dim=1)[0] < 1
                        for idx, can_snap in enumerate(snap_mask):
                            if can_snap:
                                fi = furs_idx[idx]
                                wi_local = nearest_wall_ids[idx]
                                n = normals[wi_local]
                                
                                # 1. 位移校正
                                d_val = c2c_dists[idx, wi_local]
                                ft, wt = f_thicks[idx, wi_local], w_half_t[wi_local]
                                # 计算并应用物理偏移
                                s_gap = d_val - torch.sign(d_val) * (ft + wt)
                                raw_off = -s_gap * n
                                
                                # 边界 Clamp (use per-batch room bounds)
                                safe_min = room_min[b].to(device) + half_size[b, fi]
                                safe_max = room_max[b].to(device) - half_size[b, fi]
                                clamped_p = torch.max(torch.min(pos[b, fi] + raw_off, safe_max), safe_min)
                                snap_offset[b, fi] = clamped_p - pos[b, fi]
        
                                # 2. 角度校正
                                # 如果墙体是 Z 轴厚度，相比 w_angle 有 90 度偏置
                                angle_offset = (math.pi / 2.0) * thick_axis_idx[wi_local].float()
                                # 目标角度：朝向房间内 (+pi)
                                target_a = w_angle[wi_local] + angle_offset + math.pi
                                
                                # 直接作用于 D=17 的 sin(7) / cos(6) 空间
                                angle_corr_cos[b, fi] = torch.cos(target_a) - x[b, fi, 6]
                                angle_corr_sin[b, fi] = torch.sin(target_a) - x[b, fi, 7]

                                if DEBUG_PRINT and snap_offset[b, fi].abs().sum() > 1e-4:
                                    print(f"[Hard Snap] Slot {fi} Snap to Wall {walls_idx[wi_local].item()} | Axis: {'Z' if angle_offset > 0 else 'X'}")

        #total_loss = loss_attach*1.0
        # All losses are now [B] tensors (per-scene), compute weighted sum
        total_loss_per_scene = loss_boundary*5.0 + loss_collision*1.0 + loss_attach*5.0 + loss_clearance*10.0  # [B]
        # Use .sum() so each scene gets its FULL gradient (no 1/B scaling)
        # Gradient for scene b only comes from total_loss_per_scene[b] - they are already independent
        total_loss = total_loss_per_scene.sum()  # Scalar for backward, each scene contributes fully
        
        if DEBUG_PRINT:
            # Print per-scene losses
            print(f"[Physics Loss] Sum Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.sum().item():.6f} | Clear: {loss_clearance.sum().item():.6f} | "
                  f"Bound: {loss_boundary.sum().item():.6f} | Snap: {loss_attach.sum().item():.6f}")
            for b in range(B):
                print(f"  [Scene {b}] Total: {total_loss_per_scene[b].item():.6f} | "
                      f"Coll: {loss_collision[b].item():.6f} | Snap: {loss_attach[b].item():.6f}")

        #if total_loss <= 1e-7:
        if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
            if DEBUG_PRINT:
                print("Physical loss too small, early return")
            return None, total_loss

        total_loss.backward()
        
        # 5. 组装梯度：确保物理掩码在所有分支下生效
        if x.grad is None: return None, total_loss
        if DEBUG_PRINT:
            # 找到 grad 最大的物体，看是否有能量产生
            max_grad_val = x.grad[:, :, 0:3].abs().max().item()
            print(f"[Debug Backward] Raw Gradient Max Magnitude: {max_grad_val:.6f}")
        
        grad_final = torch.zeros_like(x.grad)
        
        # 处理位移梯度 (0:3)
        combined_pos_grad = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            snap_pos_grad = snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)
            combined_pos_grad = combined_pos_grad + snap_pos_grad * 0.5 
        grad_final[:, :, 0:3] = combined_pos_grad * boundary_mask.unsqueeze(-1)

        # 处理旋转梯度 (6:cos, 7:sin)
        # 注意：x.grad[:,:,6:8] 已经包含了 Soft Snap 产生的梯度
        grad_final[:, :, 6] = x.grad[:, :, 6] * scale
        grad_final[:, :, 7] = x.grad[:, :, 7] * scale
        
        if use_hard_snap:
            grad_final[:, :, 6] = (grad_final[:, :, 6] + angle_corr_cos * 0.2) * boundary_mask
            grad_final[:, :, 7] = (grad_final[:, :, 7] + angle_corr_sin * 0.2) * boundary_mask
        else:
            grad_final[:, :, 6:8] = grad_final[:, :, 6:8] * boundary_mask.unsqueeze(-1)

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

