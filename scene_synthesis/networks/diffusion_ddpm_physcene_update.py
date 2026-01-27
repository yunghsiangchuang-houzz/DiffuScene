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
        Sample from the model
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

        objectness = model_mean[:,:,self.diffusion.bbox_dim+self.diffusion.class_dim-1:self.diffusion.bbox_dim+self.diffusion.class_dim]<0

        # 2. 拦截逻辑：在物理引导之前抹除建筑元素 (t < 20)
        #if t[0] < 200:
        if t[0] < 20:
        #if t[0] == 0:
        #if t[0] < 100:
            #if DEBUG_PRINT:
            if False:
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

            # 只有当存在新生成的空间时才执行过滤
            remove_count = 0
            if True:
                new_model_mean, remove_count = self._apply_arch_item_removal(model_mean, pred_xstart, num_partial=num_partial)
                if remove_count > 0:
                    # 写回修改后的均值
                    model_mean = new_model_mean
                if DEBUG_PRINT:
                    print(f"DEBUG removed {remove_count} arch items")

            if DEBUG_PRINT and remove_count > 0:
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

        #if False:
        if t[0] == 20 - 1:
            print("Apply hard physical projection at timestamp "+str(int(t[0])))
            with torch.no_grad():
                model_mean = self._apply_physical_wall_attach(model_mean, num_partial=num_partial)

        #if True:
        #if t[0] == 0:
        #if t[0] < 100:
        guidance_count = 10
        #if t[0] < guidance_count:
        if False:
            #if False:
            if t[0] == 0:
                print("Apply hard physical projection first time timestep "+str(int(t[0])))
                with torch.no_grad():
                    model_mean = self._apply_physical_wall_attach(model_mean, num_partial=num_partial)

            print("guidance on timestep "+str(int(t[0])))
            device = model_mean.device

            use_damping_gradient = False
            if use_damping_gradient:
                velocity = torch.zeros_like(model_mean)

            with torch.enable_grad():
                #t_linear_decay = max(0.1, t[0] / guidance_count)
                # If guidance_count is big, we can use a cosine decay, or use the same way as the paper.
                # TODO: Test that
                t_linear_decay = 1

                # Crucial: detach and require_grad to create a new trackable starting point
                # Compute gradient based on the trackable tensor
                #max_tries = 10
                #max_tries = 1
                #max_tries = 10 if t[0] == 0 else 1
                #max_tries = 20 if t[0] == 0 else 1
                max_tries = 20 if t[0] == 0 else 1
                #max_tries = 100 if t[0] == 0 else 1
                #max_tries = 500 if t[0] == 0 else 1
                #max_tries = 1000 if t[0] == 0 else 1
                #max_tries = 10000 if t[0] == 0 else 1

                grad_clip = 0.01
                #grad_clip = 0.05

                #base_scale = 0.001
                max_scale = 0.002 * t_linear_decay
                min_scale = 0.0001 * t_linear_decay

                max_grad_clip = 0.01 * t_linear_decay
                min_grad_clip = 0.001 * t_linear_decay

                original_raw_size = self.diffusion.descale_to_origin(
                    model_mean[:, :, 3:6],
                    self.diffusion._sizes_min.to(device),
                    self.diffusion._sizes_max.to(device)
                ).detach() # detach 确保它是一个常数锚点，不产生梯度

                for trial in range(max_tries):
                    model_mean_for_grad = model_mean.detach().requires_grad_(True)

                    progress = trial / max_tries
                    #use_soft_snap = True
                    use_hard_snap = False
                    if t[0] > 0:
                        use_soft_snap = False
                        use_scaling = False
                    else:
                        use_soft_snap = True
                        use_scaling = progress > 0.3 and progress < 0.7

                    update_pos_gradient = True
                    update_size_gradient = use_scaling

                    #decay_factor = max(0.1, 1.0 - (trial / max_tries))
                    #scale = base_scale * decay_factor
                    smooth_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    scale = min_scale + (max_scale - min_scale) * smooth_factor
                    grad_clip = min_grad_clip + (max_grad_clip - min_grad_clip) * smooth_factor


                    gradient, loss = self._get_simple_iou_gradient_v8(
                        model_mean_for_grad,
                        room_min=partial_room_min,
                        room_max=partial_room_max,
                        num_partial=num_partial, # 关键新增输入
                        original_raw_size=original_raw_size, # 传入固定的锚点
                        use_soft_snap=use_soft_snap,
                        use_hard_snap=use_hard_snap,
                        scale=scale,
                        grad_clip=grad_clip,
                        use_scaling=use_scaling,
                        update_pos_gradient=update_pos_gradient,
                        update_size_gradient=update_size_gradient,
                    )
                    if gradient is None:
                        break
                    #if num_partial > 0:
                    #    # 前 num_partial 个物体的所有梯度清零，确保它们不动
                    #    gradient[:, :num_partial, :] = 0
                    # Add the steering signal back to the original mean
                    if use_damping_gradient:
                        #gamma = 0.8 if progress < 0.8 else 0.4
                        gamma = 0.8  # 动量系数 (0.5-0.9 之间，值越大平滑效果越强)
                        velocity = gamma * velocity + (1.0 - gamma) * gradient
                        print(f"[{trial+1}/{max_tries}]APPLY guidance: loss = {loss:.6f}")
                        # --- 3. 使用平滑后的 velocity 更新 model_mean ---
                        model_mean = model_mean + velocity
                    else:
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

            #if False:
            if t[0] == 0:
                print("Apply hard physical projection second time timestep "+str(int(t[0])))
                with torch.no_grad():
                    model_mean = self._apply_physical_wall_attach(model_mean, num_partial=num_partial)

                if DEBUG_PRINT:
                    print(f"\n" + "="*80)
                    print(f"DEBUG TOTAL model_mean [Step {int(t[0])}] after physical rule")
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


        else:
            print("skip guidance on timestep "+str(int(t[0])))

        if t[0] == 0:
            new_model_mean, remaining_issues, modified = self._apply_physical_changes(model_mean, partial_room_min, partial_room_max, num_partial=num_partial)
            if modified:
                model_mean = new_model_mean

            # always attach to the wall
            # FIXME: it seems not always work?
            with torch.no_grad():
                model_mean = self._apply_physical_wall_attach(model_mean, num_partial=num_partial)

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
        return self._p_sample_loop_complete_v1(denoise_fn, shape, device, condition, condition_cross,
                                               noise_fn=noise_fn,clip_denoised=clip_denoised, keep_running=keep_running,partial_boxes=partial_boxes,
                                               dual_path_compare=dual_path_compare)

    def _p_sample_loop_complete_v1(self, denoise_fn, shape, device, condition, condition_cross,
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
            partial_world = self.diffusion.descale_to_origin(
                partial_boxes[:, :, 0:3],
                self.diffusion._centroids_min.to(device),
                self.diffusion._centroids_max.to(device)
            )

            # 1. 获取有效性掩码 (Objectness)
            o_idx = self.diffusion.bbox_dim + self.diffusion.class_dim - 1
            is_valid_mask = partial_boxes[:, :, o_idx] <= 0 # [B, N]

            # 2. 获取类别掩码 (只选取 Wall, Class ID = 5)
            class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
            class_logits = partial_boxes[:, :, class_range] # [B, N, class_dim]
            pred_class_ids = torch.argmax(class_logits, dim=-1) # [B, N]
            is_wall_mask = (pred_class_ids == CLASS_ID_VALS['wall']) # 找到所有 Wall 的索引

            # 3. 合并掩码：必须是有效物体且类别是墙
            # 形状依然是 [B, N]
            wall_filter_mask = is_valid_mask & is_wall_mask

            # 4. 提取墙体坐标点
            valid_wall_points = partial_world[wall_filter_mask] # [TotalWallObjects, 3]

            # 安全检查：如果场景中存在墙体，则按墙体计算；否则退回到使用所有有效物体（防止报错）
            if valid_wall_points.shape[0] > 0:
                partial_room_min = valid_wall_points.min(dim=0)[0] # [3]
                partial_room_max = valid_wall_points.max(dim=0)[0] # [3]
                print(f"DEBUG: Room AABB calculated using {valid_wall_points.shape[0]} wall segments.")
            else:
                # 降级逻辑：如果没有墙，则使用所有有效物体（如 floor）计算
                valid_partial_points = partial_world[is_valid_mask]
                assert valid_partial_points.shape[0] > 0
                partial_room_min = valid_partial_points.min(dim=0)[0]
                partial_room_max = valid_partial_points.max(dim=0)[0]
                print("WARNING: No wall segments found! Falling back to all valid objects for AABB.")

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
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False):
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

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True
        if use_loss_collision:
            # --- A. 基础碰撞损失 (Collision Loss) ---
            loss_collision = torch.tensor(0.0, device=device)
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            mask_valid_pair = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
            mask_self = torch.eye(N, device=device).unsqueeze(0)
            is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
            mask_moveable = (is_moveable.unsqueeze(2).bool() | is_moveable.unsqueeze(1).bool()).float()

            # 碰撞掩码：有效对 + 排除自碰 + 至少一个可动 + 排除地板
            final_collision_mask = mask_valid_pair * (1 - mask_self) * mask_moveable * is_not_floor_mask.unsqueeze(2)
            loss_collision = (intersection_vol * final_collision_mask).sum()

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
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

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
            loss_clearance = (intersection_area * clearance_mask).sum()

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
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            boundary_mask = valid_mask * is_moveable * is_not_floor_mask
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

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

                    # 3. 计算所有可能的间隙并寻找“物理距离表面最近”的墙
                    all_gaps = torch.abs(c2c_dists) - (f_thicks + w_half_t.unsqueeze(0)) # [Nf, Nw]

                    # 关键改进：使用绝对值寻找最近墙，防止错误的负投影干扰
                    dist_to_surface = torch.abs(all_gaps)
                    _, nearest_wall_ids = dist_to_surface.min(dim=1)
                    # 提取每个家具对应的最近墙的带符号间隙
                    min_gaps = all_gaps.gather(1, nearest_wall_ids.unsqueeze(-1)).squeeze(-1) # [Nf]

                    # --- A. Soft Snap (双向弹簧逻辑) ---
                    if use_soft_snap:
                        # 移除 clamp。Gap > 0 (远) 产生拉力；Gap < 0 (穿) 产生推力
                        loss_attach = loss_attach + min_gaps.pow(2).sum()

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

                                # 边界 Clamp
                                safe_min = room_min.to(device) + half_size[b, fi]
                                safe_max = room_max.to(device) - half_size[b, fi]
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
        total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*5.0 + loss_clearance*10.0
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f}")

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


    def _get_simple_iou_gradient_v4(self, x, room_min, room_max, num_partial=0,
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False):
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

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True
        if use_loss_collision:
            # --- A. 基础碰撞损失 (Collision Loss) ---
            loss_collision = torch.tensor(0.0, device=device)
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            mask_valid_pair = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
            mask_self = torch.eye(N, device=device).unsqueeze(0)
            is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
            mask_moveable = (is_moveable.unsqueeze(2).bool() | is_moveable.unsqueeze(1).bool()).float()

            # 碰撞掩码：有效对 + 排除自碰 + 至少一个可动 + 排除地板
            final_collision_mask = mask_valid_pair * (1 - mask_self) * mask_moveable * is_not_floor_mask.unsqueeze(2)
            loss_collision = (intersection_vol * final_collision_mask).sum()

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
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

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
            loss_clearance = (intersection_area * clearance_mask).sum()

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
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            boundary_mask = valid_mask * is_moveable * is_not_floor_mask
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0
            for b in range(B):
                walls_idx = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
                furs_idx = torch.where((is_furniture_mask[b] * is_moveable[b] * valid_mask[b]) > 0.5)[0]

                if len(walls_idx) > 0 and len(furs_idx) > 0:
                    # 获取墙体法线
                    w_pos, w_angle = pos[b, walls_idx], angles[b, walls_idx]
                    thick_axis = torch.argmin(raw_size[b, walls_idx][:, [0, 2]], dim=-1)
                    n_x = torch.stack([torch.cos(w_angle), torch.zeros_like(w_angle), torch.sin(w_angle)], dim=-1)
                    n_z = torch.stack([-torch.sin(w_angle), torch.zeros_like(w_angle), torch.cos(w_angle)], dim=-1)
                    normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

                    # --- 核心修复：内部归一化处理 ---
                    f_cos, f_sin = x[b, furs_idx, 6], x[b, furs_idx, 7]
                    f_mag = torch.sqrt(f_cos**2 + f_sin**2 + 1e-8)
                    f_cos_n = f_cos / f_mag
                    f_sin_n = f_sin / f_mag

                    # 使用归一化后的分量构建局部坐标系
                    # local_X: 家具的 X 轴方向向量
                    # local_Z: 家具的 Z 轴方向向量
                    local_X = torch.stack([f_cos_n, torch.zeros_like(f_cos_n), f_sin_n], dim=-1)
                    local_Z = torch.stack([-f_sin_n, torch.zeros_like(f_sin_n), f_cos_n], dim=-1)

                    for i, f_i in enumerate(furs_idx):
                        cid = pred_class_ids[b, f_i].item()
                        f_sz = raw_size[b, f_i, [0, 2]]
                        is_x_longer = f_sz[0] > f_sz[1]

                        # 选择背部轴
                        if cid == 1: # Toilet: 窄边靠墙 -> 对应长轴
                            back_axis = local_X[i] if is_x_longer else local_Z[i]
                            back_is_X = is_x_longer
                        else:        # Others: 长边靠墙 -> 对应短轴
                            back_axis = local_Z[i] if is_x_longer else local_X[i]
                            back_is_X = not is_x_longer

                        # 寻找最近墙逻辑
                        rel_pos = pos[b, f_i] - w_pos
                        c2c_d = (rel_pos * normals).sum(dim=-1)
                        f_thick_proj = (true_half_size[b, f_i].unsqueeze(0) * normals.abs()).sum(dim=-1)
                        gaps = torch.abs(c2c_d) - (f_thick_proj + (raw_size[b, walls_idx].min(dim=-1)[0]/2.0))

                        nearest_wi = torch.abs(gaps).argmin(); n_w = normals[nearest_wi]

                        # 4. Soft Snap: 此时点积绝对值永远 <= 1，Loss 恒正
                        if use_soft_snap:
                            loss_attach = loss_attach + gaps[nearest_wi].pow(2)
                            dot = (back_axis * n_w).sum()
                            # 即使不分正反，1 - dot^2 也永远 >= 0
                            loss_attach = loss_attach + (1.0 - torch.pow(dot, 2)) * 0.5

                        if use_hard_snap and torch.abs(gaps[nearest_wi]) < 0.15:
                            # 修正：使用正确的变量名 f_thick_proj
                            s_gap = c2c_d[nearest_wi] - torch.sign(c2c_d[nearest_wi])*(f_thick_proj[nearest_wi] + raw_size[b, walls_idx[nearest_wi]].min()/2.0)

                            # 应用位移校正
                            snap_p = pos[b, f_i] - s_gap * n_w
                            # 边界约束
                            snap_p = torch.max(torch.min(snap_p, room_max.to(device)-half_size[b, f_i]), room_min.to(device)+half_size[b, f_i])
                            snap_offset[b, f_i] = snap_p - pos[b, f_i]

                            # 目标角度：让 back_axis 指向墙面
                            v_to_wall = -torch.sign(c2c_d[nearest_wi]) * n_w
                            target_a_back = torch.atan2(v_to_wall[2], v_to_wall[0])

                            # 如果 back_axis 是 Z，则主角度 (local_X) 需要偏移 90 度
                            final_target_a = target_a_back if back_is_X else (target_a_back - math.pi/2)

                            angle_corr_cos[b, f_i] = torch.cos(final_target_a) - x[b, f_i, 6]
                            angle_corr_sin[b, f_i] = torch.sin(final_target_a) - x[b, f_i, 7]

                            if DEBUG_PRINT:
                                print(f"[Hard Snap] Slot")

        #total_loss = loss_attach*1.0
        total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*1.0 + loss_clearance*10.0
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f}")

        #if total_loss <= 1e-7:
        #if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
        #    if DEBUG_PRINT:
        #        print("Physical loss too small, early return")
        #    return None, total_loss

        total_loss.backward()

        # 5. 组装梯度：确保物理掩码在所有分支下生效
        if x.grad is None: return None, total_loss
        if DEBUG_PRINT:
            # 找到 grad 最大的物体，看是否有能量产生
            max_grad_val = x.grad[:, :, 0:3].abs().max().item()
            print(f"[Debug Backward] Raw Gradient Max Magnitude: {max_grad_val:.6f}")

        grad_final = torch.zeros_like(x.grad)
        # 位移梯度
        pos_grad = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            # 增加位移修正权重
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            pos_grad = pos_grad + (snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)) * 0.3

        grad_final[:, :, 0:3] = pos_grad * boundary_mask.unsqueeze(-1)
        # 旋转梯度 (结合了 Soft 对齐力和 Hard 校正力)
        grad_final[:, :, 6] = (x.grad[:, :, 6] * scale + angle_corr_cos * 0.1) * boundary_mask
        grad_final[:, :, 7] = (x.grad[:, :, 7] * scale + angle_corr_sin * 0.1) * boundary_mask

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

    def _get_simple_iou_gradient_v5(self, x, room_min, room_max, num_partial=0,
                               original_raw_size=None, # 传入 trial=0 时的原始世界尺寸
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False,
                               use_scaling=True):

        device = x.device
        B, N, D = x.shape

        CLASS_MIN_SIZES = {
            0: torch.tensor([0.6, 0.7, 0.4], device=device), # 最小宽度 0.6m, 深度 0.4m
            2: torch.tensor([0.8, 1.8, 0.8], device=device), # 淋浴房最小 0.8x0.8
            3: torch.tensor([1.2, 0.4, 0.7], device=device)  # 浴缸最小长度 1.2m
        }

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

        is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
        boundary_mask = valid_mask * is_moveable * is_not_floor_mask
        can_scale_mask = is_furniture_mask * (pred_class_ids != 1) * is_moveable

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

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True
        if use_loss_collision:
            # 1. 计算相交体积 [B, N, N]
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            # 2. 基础掩码
            mask_valid_pair = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1) # 双方必须有效
            mask_self = torch.eye(N, device=device).unsqueeze(0)               # 排除自身

            # 3. 定义“可移动家具”掩码 [B, N]
            # 家具 ID 为 0-3 (Vanity, Toilet, Shower, Tub)，且必须是可移动的 (is_moveable)
            is_mov_furn_mask = is_furniture_mask * is_moveable

            # 4. 定义“非地板”掩码 [B, N]
            # 这里的 is_not_floor_mask 已经排除了 ID 4

            # 5. 核心逻辑：(一方是动家具 AND 另一方非地板) OR (另一方是动家具 AND 一方非地板)
            # 这样写确保了“家具 vs 墙”、“家具 vs 家具”、“家具 vs 门”都会被计算
            # 且排除了“墙 vs 墙”这种无意义的计算（墙不会动）
            mask_coll_target = (
                (is_mov_furn_mask.unsqueeze(2) * is_not_floor_mask.unsqueeze(1)) |
                (is_not_floor_mask.unsqueeze(2) * is_mov_furn_mask.unsqueeze(1))
            ).float()

            # 6. 最终合成掩码
            # 注意：使用 (1 - mask_self) 排除自身，防止一个物体把自己算进去
            final_mask = mask_valid_pair * (1.0 - mask_self) * mask_coll_target

            # 7. 计算总损失
            # 我们除以 2 是因为矩阵 [i, j] 和 [j, i] 是重复的对称碰撞
            loss_collision = (intersection_vol * final_mask).sum() / 2.0

        # --- B. 2D 门前净空损失 (XZ-Plane Only) ---
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

        if use_loss_clearance:
            clearance_depth = 0.8

            cos_a, sin_a = x[:, :, 6].abs().unsqueeze(-1), x[:, :, 7].abs().unsqueeze(-1)
            d_horiz_sizes = raw_size[:, :, [0, 2]]
            d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
            d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

            g_depth = d_thick + clearance_depth * 2.0
            g_ex = (g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
            g_ez = (g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
            g_mins_2d = pos[:, :, [0, 2]] - torch.cat([g_ex, g_ez], dim=-1)
            g_maxs_2d = pos[:, :, [0, 2]] + torch.cat([g_ex, g_ez], dim=-1)

            f_mins_2d, f_maxs_2d = pos[:, :, [0, 2]] - half_size[:, :, [0, 2]], pos[:, :, [0, 2]] + half_size[:, :, [0, 2]]
            inter_mins_2d = torch.max(f_mins_2d.unsqueeze(2), g_mins_2d.unsqueeze(1))
            inter_maxs_2d = torch.min(f_maxs_2d.unsqueeze(2), g_maxs_2d.unsqueeze(1))
            intersection_area = torch.clamp(inter_maxs_2d - inter_mins_2d, min=0.0).prod(dim=-1)

            clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
            loss_clearance = (intersection_area * clearance_mask).sum()

        # 4. 房间边界惩罚 (只对 NEW & VALID 物体)
        #loss_boundary = 0
        #if room_min is not None and room_max is not None:
        assert room_min is not None and room_max is not None
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0
            for b in range(B):
                w_idx, f_idx = torch.where((is_wall_mask[b]*valid_mask[b]) > 0.5)[0], torch.where((is_furniture_mask[b]*is_moveable[b]*valid_mask[b]) > 0.5)[0]
                if len(w_idx) > 0 and len(f_idx) > 0:
                    # ... [计算 normals 和 local 轴的逻辑保持不变] ...
                    # (为了节省篇幅，假设 normals, local_X, local_Z 已算出)

                    for i, fi in enumerate(f_idx):
                        # 锁定语义识别 (建议使用 original_raw_size 防止 90 度乱转)
                        orig_f_sz = original_raw_size[b, fi, [0, 2]]
                        is_x_longer = orig_f_sz[0] > orig_f_sz[1]
                        cid = pred_class_ids[b, fi].item()
                        if cid == 1: back_axis, back_is_X = (local_X[i] if is_x_longer else local_Z[i]), is_x_longer
                        else: back_axis, back_is_X = (local_Z[i] if is_x_longer else local_X[i]), not is_x_longer

                        rel_p = pos[b, fi] - pos[b, w_idx]
                        c2c_d = (rel_p * normals).sum(dim=-1)
                        f_thick_p = (true_half_size[b, fi].unsqueeze(0) * normals.abs()).sum(dim=-1)
                        gaps = torch.abs(c2c_d) - (f_thick_p + (raw_size[b, w_idx].min(dim=-1)[0]/2.0))
                        ni = torch.abs(gaps).argmin(); n_w = normals[ni]
                        v_to_wall = -torch.sign(c2c_d[ni]) * n_w

                        # --- 核心改进：Soft Snap 只计算距离 Loss ---
                        # 增加一个线性项让吸附更彻底：pow(2) 负责远距离，abs 负责近距离接触
                        loss_attach += gaps[ni].pow(2) + torch.abs(gaps[ni]) * 0.05

                        # 只有 Hard Snap 才允许计算角度修正
                        if use_hard_snap and torch.abs(gaps[ni]) < 0.15:
                            s_gap = c2c_d[ni] - torch.sign(c2c_d[ni]) * (f_thick_p[ni] + raw_size[b, w_idx[ni]].min()/2.0)
                            snap_offset[b, fi] = -s_gap * n_w

                            # 角度修正逻辑仅保留在 Hard Snap 中
                            target_a_back = torch.atan2(v_to_wall[2], v_to_wall[0])
                            final_t_a = target_a_back if back_is_X else (target_a_back - math.pi/2)
                            angle_corr_cos[b, fi], angle_corr_sin[b, fi] = torch.cos(final_t_a) - x[b, fi, 6], torch.sin(final_t_a) - x[b, fi, 7]

                            if DEBUG_PRINT:
                                print(f"[Hard Snap] Slot")

        # --- C. V5 统一尺寸优化逻辑 (通用平面版) ---
        loss_size_total = torch.tensor(0.0, device=device)
        if use_scaling and original_raw_size is not None:
            for b in range(B):
                f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
                for fi in f_idx:
                    cid = pred_class_ids[b, fi].item()
                    curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

                    # 1. 轴无关的底线防御 (Hinge Loss)
                    if cid in CLASS_MIN_SIZES:
                        m_val = CLASS_MIN_SIZES[cid].to(device)
                        # 高度 (Y轴)
                        loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0
                        # 平面 (XZ轴) 排序对比，自动适配旋转
                        curr_h, _ = torch.sort(curr_s[[0, 2]]) # [短边, 长边]
                        min_h, _ = torch.sort(m_val[[0, 2]])   # [最小深, 最小宽]
                        loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * 500.0

                    # 2. 尺寸回归项 (阻力项：确保无碰撞时不缩放，且只在平面动)
                    # X(1.0), Y(200.0 锁死高度), Z(1.0)
                    dyn_weights = torch.tensor([1.0, 200.0, 1.0], device=device)
                    loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 10.0

        #total_loss = loss_attach*1.0
        #total_loss = loss_size_total*1.0
        #total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*1.0 + loss_clearance*10.0 + loss_size_total*1.0
        total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*10.0 + loss_clearance*10.0 + loss_size_total*0.1
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f} | "
                  f"Size: {loss_size_total.item():.6f}")

        #if total_loss <= 1e-7:
        if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
            if DEBUG_PRINT:
                print("Physical loss too small, early return")
            return None, total_loss

        total_loss.backward()

        # 4. 组装梯度
        grad_final = torch.zeros_like(x.grad)
        pos_g = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            pos_g += (snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)) * 0.3

        grad_final[:, :, 0:3] = pos_g * boundary_mask.unsqueeze(-1)
        if use_scaling:
            grad_final[:, :, 3:6] = x.grad[:, :, 3:6] * scale * 0.5 * can_scale_mask.unsqueeze(-1)

        grad_final[:, :, 6] = (x.grad[:, :, 6] * scale + angle_corr_cos * 0.1) * boundary_mask
        grad_final[:, :, 7] = (x.grad[:, :, 7] * scale + angle_corr_sin * 0.1) * boundary_mask

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

    def _apply_hard_physical_projection_v1(self, x, num_partial=0):
        """
        物理硬投影 v2:
        1. 解决马桶轴向偏移问题（强制垂直）。
        2. 提高贴墙精度 (1mm 阈值)。
        3. 仅报告产生实际物理变动的家具。
        """
        device = x.device
        B, N, D = x.shape

        # 1. 基础信息提取
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)

        o_idx = 16
        valid_mask = (x[:, :, o_idx] <= 0).float()
        is_wall_mask = (pred_class_ids == 5).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_toilet_mask = (pred_class_ids == 1).float()

        # 2. 坐标转换
        pos = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
        raw_size = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
        curr_angles = torch.atan2(x[:, :, 7], x[:, :, 6])

        new_pos = pos.clone()
        new_cos = x[:, :, 6].clone()
        new_sin = x[:, :, 7].clone()

        moved_report = []

        for b in range(B):
            w_indices = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = torch.where((is_furniture_mask[b] * valid_mask[b]) > 0.5)[0]
            # 只处理新生成的家具
            f_indices = f_indices[f_indices >= num_partial]

            if len(w_indices) == 0 or len(f_indices) == 0: continue

            # --- 计算墙体法线 ---
            w_sizes = raw_size[b, w_indices]
            thick_axis = torch.argmin(w_sizes[:, [0, 2]], dim=-1)
            w_angles = torch.atan2(x[b, w_indices, 7], x[b, w_indices, 6])
            # 构建墙体局部坐标系
            n_x = torch.stack([torch.cos(w_angles), torch.zeros_like(w_angles), torch.sin(w_angles)], dim=-1)
            n_z = torch.stack([-torch.sin(w_angles), torch.zeros_like(w_angles), torch.cos(w_angles)], dim=-1)
            w_normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

            for fi in f_indices:
                f_pos = pos[b, fi]
                f_sz = raw_size[b, fi]
                is_toilet = is_toilet_mask[b, fi] > 0.5

                # 计算到所有墙的 Gap
                rel_p = f_pos - pos[b, w_indices]
                dist_to_walls = (rel_p * w_normals).sum(dim=-1)
                f_half_thick = (f_sz.unsqueeze(0) * w_normals.abs()).sum(dim=-1) / 2.0
                w_half_thick = w_sizes.min(dim=-1)[0] / 2.0
                gaps = torch.abs(dist_to_walls) - (f_half_thick + w_half_thick)

                ni = torch.abs(gaps).argmin()
                target_n = w_normals[ni]
                actual_gap = gaps[ni]

                # --- A. 位置对齐 (1mm 阈值) ---
                pos_changed = False
                if torch.abs(actual_gap) > 0.001:  # 只要缝隙大于 1 毫米就移动
                    push_dir = -torch.sign(dist_to_walls[ni])
                    new_pos[b, fi] += push_dir * actual_gap * target_n
                    pos_changed = True

                # --- B. 马桶旋转对齐 (增加 90 度偏置修正) ---
                rot_changed = False
                if is_toilet:
                    v_to_room = -torch.sign(dist_to_walls[ni]) * target_n
                    target_angle = torch.atan2(v_to_room[2], v_to_room[0])

                    # 关键修复：既然之前的 FinalAngle = TargetAngle 是水平贴墙
                    # 那么说明你的模型正面定义需要旋转 90 度才能垂直。
                    # 我们尝试在这里统一加上 1.5708 (90 deg)
                    final_a = target_angle - 1.5708

                    # 检查角度偏差是否大于 1 度
                    angle_diff = torch.abs(curr_angles[b, fi] - final_a)
                    # 处理角度回环问题
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)).abs()

                    if angle_diff > 0.017: # 约 1 度
                        new_cos[b, fi] = torch.cos(final_a)
                        new_sin[b, fi] = torch.sin(final_a)
                        rot_changed = True

                if pos_changed or rot_changed:
                    # 记录具体变动
                    moved_report.append({
                        "id": fi.item(),
                        "cat": pred_class_ids[b, fi].item(),
                        "gap": actual_gap.item(),
                        "pos_mv": pos_changed,
                        "rot_mv": rot_changed
                    })

        # 3. 数据 rescale 并回填
        x_new = x.clone()
        c_min, c_max = self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)
        x_new[:, :, 0:3] = (new_pos - c_min) / (c_max - c_min) * 2.0 - 1.0
        x_new[:, :, 6] = new_cos
        x_new[:, :, 7] = new_sin

        # 4. 输出变动清单
        if DEBUG_PRINT and len(moved_report) > 0:
            print(f"\n[Hard Projection Activity] Found {len(moved_report)} items needing adjustment:")
            for item in moved_report:
                action = []
                if item["pos_mv"]: action.append(f"AttachedToWall(Gap:{item['gap']*100:.1f}cm)")
                if item["rot_mv"]: action.append("RotatedToPerpendicular")
                print(f"  - Slot {item['id']} (Cat:{item['cat']}): {' & '.join(action)}")

        return x_new

    def _apply_hard_physical_projection_v2(self, x, num_partial=0):
        """
        物理硬投影 v3 (Proactive Door-Aware):
        1. 遍历墙体候选，排除会导致家具与门(Door)重叠的投影目标。
        2. 仅在不堵门的情况下执行贴墙。
        """
        device = x.device
        B, N, D = x.shape

        # 1. 基础信息提取
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)

        o_idx = 16
        valid_mask = (x[:, :, o_idx] <= 0).float()
        is_wall_mask = (pred_class_ids == 5).float()
        is_door_mask = (pred_class_ids == 6).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_toilet_mask = (pred_class_ids == 1).float()

        # 2. 坐标转换
        pos = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
        raw_size = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
        curr_angles = torch.atan2(x[:, :, 7], x[:, :, 6])

        new_pos = pos.clone()
        new_cos = x[:, :, 6].clone()
        new_sin = x[:, :, 7].clone()

        moved_report = []

        for b in range(B):
            w_indices = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
            d_indices = torch.where((is_door_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = torch.where((is_furniture_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = f_indices[f_indices >= num_partial]

            if len(w_indices) == 0 or len(f_indices) == 0: continue

            if DEBUG_PRINT and len(d_indices) > 0:
                print(f"\n[Projection Debug] Batch {b} | Found {len(d_indices)} Doors")
                for di_idx, di in enumerate(d_indices):
                    d_sz = raw_size[b, di]
                    print(f"  - Door Slot {di.item()}: Size={d_sz.tolist()}, Pos={pos[b, di].tolist()}")

            # 提前准备门的 AABB (增加 5cm 的安全缓冲区)
            if len(d_indices) > 0:
                d_half = raw_size[b, d_indices] / 2.0
                d_mins = pos[b, d_indices] - d_half - 0.05
                d_maxs = pos[b, d_indices] + d_half + 0.05

            # 计算墙体法线
            w_sizes = raw_size[b, w_indices]
            thick_axis = torch.argmin(w_sizes[:, [0, 2]], dim=-1)
            w_angles = torch.atan2(x[b, w_indices, 7], x[b, w_indices, 6])
            n_x = torch.stack([torch.cos(w_angles), torch.zeros_like(w_angles), torch.sin(w_angles)], dim=-1)
            n_z = torch.stack([-torch.sin(w_angles), torch.zeros_like(w_angles), torch.cos(w_angles)], dim=-1)
            w_normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

            for fi in f_indices:
                f_pos = pos[b, fi]
                f_sz = raw_size[b, fi]
                is_toilet = is_toilet_mask[b, fi] > 0.5

                # 计算到所有墙的 Gap 并排序
                rel_p = f_pos - pos[b, w_indices]
                dist_to_walls = (rel_p * w_normals).sum(dim=-1)
                f_half_thick = (f_sz.unsqueeze(0) * w_normals.abs()).sum(dim=-1) / 2.0
                w_half_thick = w_sizes.min(dim=-1)[0] / 2.0
                gaps = torch.abs(dist_to_walls) - (f_half_thick + w_half_thick)

                # 获取候选墙的排序 (从小到大)
                sorted_wall_cand = torch.argsort(torch.abs(gaps))

                best_ni = None
                if DEBUG_PRINT: print(f"  > Furniture {fi.item()} (Cat:{pred_class_ids[b,fi]}): Testing walls...")
                for cand_i in sorted_wall_cand:
                    ni = cand_i.item()
                    target_n = w_normals[ni]
                    actual_gap = gaps[ni]

                    # 模拟投影后的位置
                    push_dir = -torch.sign(dist_to_walls[ni])
                    temp_pos = f_pos + push_dir * actual_gap * target_n

                    # 校验是否堵门
                    is_blocking = False
                    if len(d_indices) > 0:
                        f_half = f_sz / 2.0
                        f_min, f_max = temp_pos - f_half, temp_pos + f_half

                        # AABB 碰撞检测: 只要与任何一扇门重叠，就标记为堵门
                        # 仅检查 X 和 Z 平面
                        for di in range(len(d_indices)):
                            overlap_x = (f_min[0] < d_maxs[di, 0]) and (f_max[0] > d_mins[di, 0])
                            overlap_z = (f_min[2] < d_maxs[di, 2]) and (f_max[2] > d_mins[di, 2])
                            if overlap_x and overlap_z:
                                is_blocking = True
                                if DEBUG_PRINT: print(f"    - Wall {ni} REJECTED: Blocks Door {d_indices[di].item()}")
                                break

                    if not is_blocking:
                        best_ni = ni
                        if DEBUG_PRINT: print(f"    - Wall {ni} ACCEPTED (Gap: {gaps[ni].item()*100:.1f}cm)")
                        break

                # 如果找到了不堵门的墙，执行更新
                if best_ni is not None:
                    ni = best_ni
                    target_n = w_normals[ni]
                    actual_gap = gaps[ni]

                    pos_changed = False
                    if torch.abs(actual_gap) > 0.001:
                        push_dir = -torch.sign(dist_to_walls[ni])
                        new_pos[b, fi] += push_dir * actual_gap * target_n
                        pos_changed = True

                    # --- B. 马桶旋转对齐 (保持原逻辑) ---
                    # 注意这里 dist_to_walls 使用的是当前选定的 ni
                    rot_changed = False
                    if is_toilet:
                        v_to_room = -torch.sign(dist_to_walls[ni]) * target_n
                        target_angle = torch.atan2(v_to_room[2], v_to_room[0])
                        final_a = target_angle - 1.5708
                        angle_diff = torch.atan2(torch.sin(curr_angles[b, fi] - final_a), torch.cos(curr_angles[b, fi] - final_a)).abs()
                        if angle_diff > 0.017:
                            new_cos[b, fi], new_sin[b, fi] = torch.cos(final_a), torch.sin(final_a)
                            rot_changed = True

                    if pos_changed or rot_changed:
                        moved_report.append({"id": fi.item(), "cat": pred_class_ids[b, fi].item(), "gap": actual_gap.item(), "pos_mv": pos_changed, "rot_mv": rot_changed})

        # 3. 数据 rescale 并回填
        x_new = x.clone()
        c_min, c_max = self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)
        x_new[:, :, 0:3] = (new_pos - c_min) / (c_max - c_min) * 2.0 - 1.0
        x_new[:, :, 6] = new_cos
        x_new[:, :, 7] = new_sin

        # 4. 输出变动清单
        if DEBUG_PRINT and len(moved_report) > 0:
            print(f"\n[Hard Projection Activity] Found {len(moved_report)} items needing adjustment:")
            for item in moved_report:
                action = []
                if item["pos_mv"]: action.append(f"AttachedToWall(Gap:{item['gap']*100:.1f}cm)")
                if item["rot_mv"]: action.append("RotatedToPerpendicular")
                print(f"  - Slot {item['id']} (Cat:{item['cat']}): {' & '.join(action)}")

        return x_new

    def _apply_physical_wall_attach(self, x, num_partial=0):
        """
        物理硬投影 v4 (Ghost Clearance 版):
        使用 1.6m 厚度的门禁幽灵盒 (Ghost Box) 来进行碰撞拦截。
        """
        device = x.device
        B, N, D = x.shape

        # 1. 基础信息提取
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)

        o_idx = 16
        valid_mask = (x[:, :, o_idx] <= 0).float()
        is_wall_mask = (pred_class_ids == 5).float()
        is_door_mask = (pred_class_ids == 6).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_toilet_mask = (pred_class_ids == 1).float()

        # 2. 坐标转换
        pos = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
        raw_size = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
        curr_angles = torch.atan2(x[:, :, 7], x[:, :, 6])

        new_pos = pos.clone()
        new_cos = x[:, :, 6].clone()
        new_sin = x[:, :, 7].clone()

        moved_report = []

        for b in range(B):
            w_indices = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
            d_indices = torch.where((is_door_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = torch.where((is_furniture_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = f_indices[f_indices >= num_partial]

            if len(w_indices) == 0 or len(f_indices) == 0: continue

            # --- 1. 预计算该 Batch 下所有门的 Ghost Box AABB ---
            ghost_mins_2d = []
            ghost_maxs_2d = []
            if len(d_indices) > 0:
                clearance_depth = 0.8
                d_pos_2d = pos[b, d_indices][:, [0, 2]]
                d_horiz_sizes = raw_size[b, d_indices][:, [0, 2]]

                # 旋转感知
                cos_a = x[b, d_indices, 6].abs().unsqueeze(-1)
                sin_a = x[b, d_indices, 7].abs().unsqueeze(-1)

                d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
                d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

                # 总厚度 = 门厚 + 前后各 0.8m
                total_g_depth = d_thick + clearance_depth * 2.0

                # 计算幽灵盒的 AABB 包络半径
                g_ex = (total_g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
                g_ez = (total_g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
                g_half_size_2d = torch.cat([g_ex, g_ez], dim=-1)

                ghost_mins_2d = d_pos_2d - g_half_size_2d
                ghost_maxs_2d = d_pos_2d + g_half_size_2d

            if DEBUG_PRINT and len(d_indices) > 0:
                print(f"\n[Debug GhostBoxes] Batch {b}:")
                for i, di in enumerate(d_indices):
                    width = (ghost_maxs_2d[i, 0] - ghost_mins_2d[i, 0]).item()
                    depth = (ghost_maxs_2d[i, 1] - ghost_mins_2d[i, 1]).item()
                    print(f"  - Door {di.item()} Ghost AABB: W={width:.2f}m, D={depth:.2f}m")

            # 计算墙体法线 (保持 v2 逻辑)
            w_sizes = raw_size[b, w_indices]
            thick_axis = torch.argmin(w_sizes[:, [0, 2]], dim=-1)
            w_angles = torch.atan2(x[b, w_indices, 7], x[b, w_indices, 6])
            n_x = torch.stack([torch.cos(w_angles), torch.zeros_like(w_angles), torch.sin(w_angles)], dim=-1)
            n_z = torch.stack([-torch.sin(w_angles), torch.zeros_like(w_angles), torch.cos(w_angles)], dim=-1)
            w_normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

            for fi in f_indices:
                f_pos, f_sz = pos[b, fi], raw_size[b, fi]
                is_toilet = is_toilet_mask[b, fi] > 0.5

                # 墙体排序
                rel_p = f_pos - pos[b, w_indices]
                dist_to_walls = (rel_p * w_normals).sum(dim=-1)

                #f_half_thick = (f_sz.unsqueeze(0) * w_normals.abs()).sum(dim=-1) / 2.0
                #w_half_thick = w_sizes.min(dim=-1)[0] / 2.0
                #gaps = torch.abs(dist_to_walls) - (f_half_thick + w_half_thick)
                #sorted_wall_cand = torch.argsort(torch.abs(gaps))

                # --- 修正后的投影逻辑 ---
                # 获取家具旋转的绝对分量
                f_cos = torch.cos(curr_angles[b, fi]).abs()
                f_sin = torch.sin(curr_angles[b, fi]).abs()

                # 计算家具在世界 X 和 Z 轴上的投影半径 (AABB)
                # 即使旋转 0.1 度，它也会正确组合 size[0] 和 size[2]
                f_aabb_half_w = (f_cos * f_sz[0] + f_sin * f_sz[2]) / 2.0
                f_aabb_half_d = (f_sin * f_sz[0] + f_cos * f_sz[2]) / 2.0

                # 根据墙体法线选择对应的投影轴
                # w_n_abs 告诉我们墙是朝向 X 还是 Z
                w_n_abs = w_normals.abs()
                f_half_thick = w_n_abs[:, 0] * f_aabb_half_w + w_n_abs[:, 2] * f_aabb_half_d
                # ---------------------------------------------------

                # 4. 计算墙体半厚度并得出间隙 (Gap)
                w_half_thick = w_sizes.min(dim=-1)[0] / 2.0
                gaps = torch.abs(dist_to_walls) - (f_half_thick + w_half_thick)

                # 5. 按间隙绝对值排序，寻找最近的墙
                sorted_wall_cand = torch.argsort(torch.abs(gaps))

                if DEBUG_PRINT:
                    print(f"\n[Debug Furniture {fi.item()} (Cat:{pred_class_ids[b,fi].item()})]")
                    print(f"  - Current Angle: {torch.rad2deg(curr_angles[b,fi]).item():.1f}°")
                    print(f"  - Physical Size: {f_sz[0].item():.2f}x{f_sz[2].item():.2f}")

                best_ni = None
                for cand_rank, cand_i in enumerate(sorted_wall_cand):
                    ni = cand_i.item()
                    # 模拟投影
                    push_dir = -torch.sign(dist_to_walls[ni])
                    temp_pos = f_pos + push_dir * gaps[ni] * w_normals[ni]

                    # --- 2. 核心：使用 Ghost Box 进行堵门校验 ---
                    is_blocking = False
                    blocker_id = -1
                    if len(d_indices) > 0:
                        # 当前家具的 2D AABB
                        f_min_2d = temp_pos[[0, 2]] - (f_sz[[0, 2]] / 2.0)
                        f_max_2d = temp_pos[[0, 2]] + (f_sz[[0, 2]] / 2.0)

                        # 与所有幽灵盒比对
                        for di in range(len(d_indices)):
                            overlap_x = (f_min_2d[0] < ghost_maxs_2d[di, 0]) and (f_max_2d[0] > ghost_mins_2d[di, 0])
                            overlap_z = (f_min_2d[1] < ghost_maxs_2d[di, 1]) and (f_max_2d[1] > ghost_mins_2d[di, 1])

                            if overlap_x and overlap_z:
                                is_blocking = True
                                blocker_id = d_indices[di].item()
                                if DEBUG_PRINT:
                                    print(f"    - Wall {ni} REJECTED: Slot {fi.item()} hits Door {d_indices[di].item()} Ghost Box")
                                break

                    if DEBUG_PRINT:
                        obesity = f_half_thick[ni].item() - (f_sz[[0,2]].min().item() / 2.0)
                        status = "REJECTED (Blocked)" if is_blocking else "ACCEPTED"
                        print(f"  - Wall {ni} (Rank {cand_rank}): {status}")
                        print(f"    - Gap to Wall: {gaps[ni].item()*100:.1f}cm")
                        print(f"    - Virtual Obesity: {obesity*100:.1f}cm (AABB error)")
                        if is_blocking:
                            print(f"    - Blocked by GhostBox of Door {blocker_id}")

                    if not is_blocking:
                        best_ni = ni
                        break

                if best_ni is None and DEBUG_PRINT:
                    print(f"  !!! WARNING: Furniture {fi.item()} could not attach to ANY wall. Floating.")

                # --- 3. 执行更新 (保持 v2 逻辑) ---
                if best_ni is not None:
                    ni = best_ni
                    target_n = w_normals[ni]
                    actual_gap = gaps[ni]

                    pos_changed = False
                    if torch.abs(actual_gap) > 0.001:
                        push_dir = -torch.sign(dist_to_walls[ni])
                        new_pos[b, fi] += push_dir * actual_gap * target_n
                        pos_changed = True

                    rot_changed = False
                    if is_toilet:
                        v_to_room = -torch.sign(dist_to_walls[ni]) * target_n
                        target_angle = torch.atan2(v_to_room[2], v_to_room[0])
                        final_a = target_angle - 1.5708
                        angle_diff = torch.atan2(torch.sin(curr_angles[b, fi] - final_a), torch.cos(curr_angles[b, fi] - final_a)).abs()
                        if angle_diff > 0.017:
                            new_cos[b, fi], new_sin[b, fi] = torch.cos(final_a), torch.sin(final_a)
                            rot_changed = True

                    if pos_changed or rot_changed:
                        moved_report.append({"id": fi.item(), "cat": pred_class_ids[b, fi].item(), "gap": actual_gap.item(), "pos_mv": pos_changed, "rot_mv": rot_changed})

        # --- 4. Rescale 并回填 ---
        x_new = x.clone()
        c_min, c_max = self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)
        x_new[:, :, 0:3] = (new_pos - c_min) / (c_max - c_min) * 2.0 - 1.0
        x_new[:, :, 6] = new_cos
        x_new[:, :, 7] = new_sin

        # 4. 输出变动清单
        if DEBUG_PRINT and len(moved_report) > 0:
            print(f"\n[Hard Projection Activity] Found {len(moved_report)} items needing adjustment:")
            for item in moved_report:
                action = []
                if item["pos_mv"]: action.append(f"AttachedToWall(Gap:{item['gap']*100:.1f}cm)")
                if item["rot_mv"]: action.append("RotatedToPerpendicular")
                print(f"  - Slot {item['id']} (Cat:{item['cat']}): {' & '.join(action)}")

        return x_new

    def _get_simple_iou_gradient_v6(self, x, room_min, room_max, num_partial=0,
                               original_raw_size=None, # 传入 trial=0 时的原始世界尺寸
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False,
                               use_scaling=True):

        device = x.device
        B, N, D = x.shape

        CLASS_MIN_SIZES = {
            0: torch.tensor([0.6, 0.7, 0.4], device=device), # 最小宽度 0.6m, 深度 0.4m
            2: torch.tensor([0.8, 1.8, 0.8], device=device), # 淋浴房最小 0.8x0.8
            3: torch.tensor([1.2, 0.4, 0.7], device=device)  # 浴缸最小长度 1.2m
        }

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

        is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
        #boundary_mask = valid_mask * is_moveable * is_not_floor_mask
        boundary_mask = valid_mask * is_moveable * is_furniture_mask
        can_scale_mask = is_furniture_mask * (pred_class_ids != 1) * is_moveable

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

        # 预先定义好 Boolean Mask 以便进行逻辑运算
        is_mov_furn_bool = (is_furniture_mask * is_moveable).bool()  # 可移动的家具 (ID 0-3)
        is_not_floor_bool = is_not_floor_mask.bool()                # 所有非地板物体 (ID != 4)

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True

        if use_loss_collision:
            # 1. 计算所有物体对之间的 AABB 相交体积 [B, N, N]
            # mins, maxs 的 shape 为 [B, N, 3]
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))

            # 计算相交部分的边长，小于 0 的部分 clamp 为 0
            # product dim=-1 得到体积 (Width * Height * Depth)
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            # 2. 基础过滤掩码
            mask_valid_pair = (valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)).bool() # 双方都必须有效
            mask_self = torch.eye(N, device=device).unsqueeze(0).bool()                 # 排除物体自身 [1, N, N]

            # 3. 核心物理逻辑掩码：
            # 条件 1：至少其中一方是“可移动家具” (否则墙撞墙不需要计算梯度)
            # 条件 2：双方都必须“不是地板” (彻底解决家具踩在地板上的数值爆炸问题)
            mask_coll_target = (
                (is_mov_furn_bool.unsqueeze(2) | is_mov_furn_bool.unsqueeze(1)) &
                (is_not_floor_bool.unsqueeze(2) & is_not_floor_bool.unsqueeze(1))
            )

            # 4. 合并所有掩码并转回 Float
            # 最终掩码：双方有效 AND 非自身 AND 满足物理碰撞条件
            final_mask = (mask_valid_pair & (~mask_self) & mask_coll_target).float()

            # 5. 计算总损失
            # 除以 2.0 是因为矩阵 [i, j] 和 [j, i] 对称计算了两次体积
            loss_collision = (intersection_vol * final_mask).sum() / 2.0

        # --- B. 2D 门前净空损失 (XZ-Plane Only) ---
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

        if use_loss_clearance:
            clearance_depth = 0.8

            cos_a, sin_a = x[:, :, 6].abs().unsqueeze(-1), x[:, :, 7].abs().unsqueeze(-1)
            d_horiz_sizes = raw_size[:, :, [0, 2]]
            d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
            d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

            g_depth = d_thick + clearance_depth * 2.0
            g_ex = (g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
            g_ez = (g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
            g_mins_2d = pos[:, :, [0, 2]] - torch.cat([g_ex, g_ez], dim=-1)
            g_maxs_2d = pos[:, :, [0, 2]] + torch.cat([g_ex, g_ez], dim=-1)

            f_mins_2d, f_maxs_2d = pos[:, :, [0, 2]] - half_size[:, :, [0, 2]], pos[:, :, [0, 2]] + half_size[:, :, [0, 2]]
            inter_mins_2d = torch.max(f_mins_2d.unsqueeze(2), g_mins_2d.unsqueeze(1))
            inter_maxs_2d = torch.min(f_maxs_2d.unsqueeze(2), g_maxs_2d.unsqueeze(1))
            intersection_area = torch.clamp(inter_maxs_2d - inter_mins_2d, min=0.0).prod(dim=-1)

            clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
            loss_clearance = (intersection_area * clearance_mask).sum()

        # 4. 房间边界惩罚 (只对 NEW & VALID 物体)
        #loss_boundary = 0
        #if room_min is not None and room_max is not None:
        assert room_min is not None and room_max is not None
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0
            for b in range(B):
                w_idx, f_idx = torch.where((is_wall_mask[b]*valid_mask[b]) > 0.5)[0], torch.where((is_furniture_mask[b]*is_moveable[b]*valid_mask[b]) > 0.5)[0]
                if len(w_idx) > 0 and len(f_idx) > 0:
                    # 计算墙体法线 (保持不变)
                    thick_axis = torch.argmin(raw_size[b, w_idx][:, [0, 2]], dim=-1)
                    n_x = torch.stack([torch.cos(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.sin(angles[b, w_idx])], dim=-1)
                    n_z = torch.stack([-torch.sin(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.cos(angles[b, w_idx])], dim=-1)
                    normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

                    for i, fi in enumerate(f_idx):
                        # --- 关键改进 1：锁定语义方向，防止 90 度翻转 ---
                        # 判定长短轴永远参考原始尺寸，不参考正在被压扁的当前尺寸
                        orig_f_sz = original_raw_size[b, fi, [0, 2]]
                        is_x_longer = orig_f_sz[0] > orig_f_sz[1]
                        cid = pred_class_ids[b, fi].item()

                        # 寻找最近墙
                        rel_p = pos[b, fi] - pos[b, w_idx]
                        c2c_d = (rel_p * normals).sum(dim=-1)
                        f_thick_p = (true_half_size[b, fi].unsqueeze(0) * normals.abs()).sum(dim=-1)
                        gaps = torch.abs(c2c_d) - (f_thick_p + (raw_size[b, w_idx].min(dim=-1)[0]/2.0))
                        ni = torch.abs(gaps).argmin(); n_w = normals[ni]

                        # --- 关键改进 2：Soft Snap 仅计算距离 Loss ---
                        if use_soft_snap:
                            # 删除了所有的点积 (Dot Product) 计算
                            # 这样 backward 时，旋转位 (6:8) 的梯度就为 0
                            loss_attach += gaps[ni].pow(2)

                        # --- 关键改进 3：Hard Snap 才允许旋转干预 ---
                        if use_hard_snap and torch.abs(gaps[ni]) < 0.15:
                            s_gap = c2c_d[ni] - torch.sign(c2c_d[ni]) * (f_thick_p[ni] + raw_size[b, w_idx[ni]].min()/2.0)
                            snap_offset[b, fi] = -s_gap * n_w

                            # 只有在 Hard Snap 阶段，我们才去构建局部坐标系并计算目标角度
                            f_mag = torch.sqrt(x[b, fi, 6]**2 + x[b, fi, 7]**2 + 1e-8)
                            f_cos_n, f_sin_n = x[b, fi, 6]/f_mag, x[b, fi, 7]/f_mag
                            l_X = torch.tensor([f_cos_n, 0.0, f_sin_n], device=device)
                            l_Z = torch.tensor([-f_sin_n, 0.0, f_cos_n], device=device)

                            if cid == 1: back_is_X = is_x_longer
                            else: back_is_X = not is_x_longer

                            v_to_wall = -torch.sign(c2c_d[ni]) * n_w
                            target_a_back = torch.atan2(v_to_wall[2], v_to_wall[0])
                            final_t_a = target_a_back if back_is_X else (target_a_back - math.pi/2)

                            angle_corr_cos[b, fi] = torch.cos(final_t_a) - x[b, fi, 6]
                            angle_corr_sin[b, fi] = torch.sin(final_t_a) - x[b, fi, 7]

                            if DEBUG_PRINT:
                                print(f"[Hard Snap] Slot")

        # --- C. V5 统一尺寸优化逻辑 (通用平面版) ---
        #loss_size_total = torch.tensor(0.0, device=device)
        #if use_scaling and original_raw_size is not None:
        #    for b in range(B):
        #        f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
        #        for fi in f_idx:
        #            cid = pred_class_ids[b, fi].item()
        #            curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

        #            # 1. 轴无关的底线防御 (Hinge Loss)
        #            if cid in CLASS_MIN_SIZES:
        #                m_val = CLASS_MIN_SIZES[cid].to(device)
        #                # 高度 (Y轴)
        #                loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0
        #                # 平面 (XZ轴) 排序对比，自动适配旋转
        #                curr_h, _ = torch.sort(curr_s[[0, 2]]) # [短边, 长边]
        #                min_h, _ = torch.sort(m_val[[0, 2]])   # [最小深, 最小宽]
        #                loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * 500.0

        #            # 2. 尺寸回归项 (阻力项：确保无碰撞时不缩放，且只在平面动)
        #            # X(1.0), Y(200.0 锁死高度), Z(1.0)
        #            dyn_weights = torch.tensor([1.0, 200.0, 1.0], device=device)
        #            loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 10.0
        loss_size_total = torch.tensor(0.0, device=device)
        if use_scaling and original_raw_size is not None:
            for b in range(B):
                # 修改点 1: 允许所有家具（包括马桶）缩放
                f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
                for fi in f_idx:
                    cid = pred_class_ids[b, fi].item()
                    curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

                    if cid in CLASS_MIN_SIZES:
                        m_val = CLASS_MIN_SIZES[cid].to(device)
                        # 高度防御
                        loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 1000.0 # 提高防御级别
                        # 平面对称防御
                        curr_h, _ = torch.sort(curr_s[[0, 2]])
                        min_h, _ = torch.sort(m_val[[0, 2]])
                        loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * 1000.0

                    # 修改点 2: 减小回归项权重，让家具不再那么“顽固”
                    dyn_weights = torch.tensor([1.0, 500.0, 1.0], device=device) # 依然锁死高度，但放开平面
                    loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 5.0

        #total_loss = loss_attach*1.0
        #total_loss = loss_size_total*1.0
        #total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*1.0 + loss_clearance*1.0 + loss_size_total*1.0

        #total_loss = loss_boundary*10.0 + loss_collision*1.0 + loss_attach*5.0 + loss_clearance*10.0 + loss_size_total*0.1
        #total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*0.05
        total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*2
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f} | "
                  f"Size: {loss_size_total.item():.6f}")

        #if total_loss <= 1e-7:
        if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
            if DEBUG_PRINT:
                print("Physical loss too small, early return")
            return None, total_loss

        total_loss.backward()

        grad_final = torch.zeros_like(x.grad)
        pos_g = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            pos_g += (snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)) * 0.8

        grad_final[:, :, 0:3] = pos_g * boundary_mask.unsqueeze(-1)
        if use_scaling:
            # 这里的 2.0 是关键，它能让家具更快地缩放以响应碰撞
            grad_final[:, :, 3:6] = x.grad[:, :, 3:6] * scale * 2.0 * can_scale_mask.unsqueeze(-1)

        # 现在，如果 use_hard_snap 为 False，angle_corr 为 0，x.grad 也为 0，家具就不会转了
        grad_final[:, :, 6] = (x.grad[:, :, 6] * scale + angle_corr_cos * 0.1) * boundary_mask
        grad_final[:, :, 7] = (x.grad[:, :, 7] * scale + angle_corr_sin * 0.1) * boundary_mask

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

    def _get_simple_iou_gradient_v7(self, x, room_min, room_max, num_partial=0,
                               original_raw_size=None, # 传入 trial=0 时的原始世界尺寸
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False,
                               use_scaling=True):

        device = x.device
        B, N, D = x.shape

        CLASS_MIN_SIZES = {
            0: torch.tensor([0.6, 0.7, 0.4], device=device), # 最小宽度 0.6m, 深度 0.4m
            2: torch.tensor([0.8, 1.8, 0.8], device=device), # 淋浴房最小 0.8x0.8
            3: torch.tensor([1.2, 0.4, 0.7], device=device)  # 浴缸最小长度 1.2m
        }

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

        is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
        #boundary_mask = valid_mask * is_moveable * is_not_floor_mask
        boundary_mask = valid_mask * is_moveable * is_furniture_mask
        can_scale_mask = is_furniture_mask * (pred_class_ids != 1) * is_moveable

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

        # 预先定义好 Boolean Mask 以便进行逻辑运算
        is_mov_furn_bool = (is_furniture_mask * is_moveable).bool()  # 可移动的家具 (ID 0-3)
        is_not_floor_bool = is_not_floor_mask.bool()                # 所有非地板物体 (ID != 4)

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True

        if use_loss_collision:
            # 1. 计算所有物体对之间的 AABB 相交体积 [B, N, N]
            # mins, maxs 的 shape 为 [B, N, 3]
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))

            # 计算相交部分的边长，小于 0 的部分 clamp 为 0
            # product dim=-1 得到体积 (Width * Height * Depth)
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            # 2. 基础过滤掩码
            mask_valid_pair = (valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)).bool() # 双方都必须有效
            mask_self = torch.eye(N, device=device).unsqueeze(0).bool()                 # 排除物体自身 [1, N, N]

            # 3. 核心物理逻辑掩码：
            # 条件 1：至少其中一方是“可移动家具” (否则墙撞墙不需要计算梯度)
            # 条件 2：双方都必须“不是地板” (彻底解决家具踩在地板上的数值爆炸问题)
            mask_coll_target = (
                (is_mov_furn_bool.unsqueeze(2) | is_mov_furn_bool.unsqueeze(1)) &
                (is_not_floor_bool.unsqueeze(2) & is_not_floor_bool.unsqueeze(1))
            )

            # 4. 合并所有掩码并转回 Float
            # 最终掩码：双方有效 AND 非自身 AND 满足物理碰撞条件
            final_mask = (mask_valid_pair & (~mask_self) & mask_coll_target).float()

            # 5. 计算总损失
            # 除以 2.0 是因为矩阵 [i, j] 和 [j, i] 对称计算了两次体积
            loss_collision = (intersection_vol * final_mask).sum() / 2.0

        # --- B. 2D 门前净空损失 (XZ-Plane Only) ---
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

        if use_loss_clearance:
            clearance_depth = 0.8

            cos_a, sin_a = x[:, :, 6].abs().unsqueeze(-1), x[:, :, 7].abs().unsqueeze(-1)
            d_horiz_sizes = raw_size[:, :, [0, 2]]
            d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
            d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

            g_depth = d_thick + clearance_depth * 2.0
            g_ex = (g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
            g_ez = (g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
            g_mins_2d = pos[:, :, [0, 2]] - torch.cat([g_ex, g_ez], dim=-1)
            g_maxs_2d = pos[:, :, [0, 2]] + torch.cat([g_ex, g_ez], dim=-1)

            f_mins_2d, f_maxs_2d = pos[:, :, [0, 2]] - half_size[:, :, [0, 2]], pos[:, :, [0, 2]] + half_size[:, :, [0, 2]]
            inter_mins_2d = torch.max(f_mins_2d.unsqueeze(2), g_mins_2d.unsqueeze(1))
            inter_maxs_2d = torch.min(f_maxs_2d.unsqueeze(2), g_maxs_2d.unsqueeze(1))
            intersection_area = torch.clamp(inter_maxs_2d - inter_mins_2d, min=0.0).prod(dim=-1)

            clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
            loss_clearance = (intersection_area * clearance_mask).sum()

        # 4. 房间边界惩罚 (只对 NEW & VALID 物体)
        #loss_boundary = 0
        #if room_min is not None and room_max is not None:
        assert room_min is not None and room_max is not None
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0
            for b in range(B):
                w_idx, f_idx = torch.where((is_wall_mask[b]*valid_mask[b]) > 0.5)[0], torch.where((is_furniture_mask[b]*is_moveable[b]*valid_mask[b]) > 0.5)[0]
                if len(w_idx) > 0 and len(f_idx) > 0:
                    # 计算墙体法线 (保持不变)
                    thick_axis = torch.argmin(raw_size[b, w_idx][:, [0, 2]], dim=-1)
                    n_x = torch.stack([torch.cos(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.sin(angles[b, w_idx])], dim=-1)
                    n_z = torch.stack([-torch.sin(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.cos(angles[b, w_idx])], dim=-1)
                    normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

                    for i, fi in enumerate(f_idx):
                        # --- 关键改进 1：锁定语义方向，防止 90 度翻转 ---
                        # 判定长短轴永远参考原始尺寸，不参考正在被压扁的当前尺寸
                        orig_f_sz = original_raw_size[b, fi, [0, 2]]
                        is_x_longer = orig_f_sz[0] > orig_f_sz[1]
                        cid = pred_class_ids[b, fi].item()

                        # 寻找最近墙
                        rel_p = pos[b, fi] - pos[b, w_idx]
                        c2c_d = (rel_p * normals).sum(dim=-1)
                        f_thick_p = (true_half_size[b, fi].unsqueeze(0) * normals.abs()).sum(dim=-1)
                        gaps = torch.abs(c2c_d) - (f_thick_p + (raw_size[b, w_idx].min(dim=-1)[0]/2.0))
                        ni = torch.abs(gaps).argmin(); n_w = normals[ni]

                        # --- 关键改进 2：Soft Snap 仅计算距离 Loss ---
                        if use_soft_snap:
                            # 删除了所有的点积 (Dot Product) 计算
                            # 这样 backward 时，旋转位 (6:8) 的梯度就为 0
                            loss_attach += gaps[ni].pow(2)

                        # --- 关键改进 3：Hard Snap 才允许旋转干预 ---
                        if use_hard_snap and torch.abs(gaps[ni]) < 0.15:
                            s_gap = c2c_d[ni] - torch.sign(c2c_d[ni]) * (f_thick_p[ni] + raw_size[b, w_idx[ni]].min()/2.0)
                            snap_offset[b, fi] = -s_gap * n_w

                            # 只有在 Hard Snap 阶段，我们才去构建局部坐标系并计算目标角度
                            f_mag = torch.sqrt(x[b, fi, 6]**2 + x[b, fi, 7]**2 + 1e-8)
                            f_cos_n, f_sin_n = x[b, fi, 6]/f_mag, x[b, fi, 7]/f_mag
                            l_X = torch.tensor([f_cos_n, 0.0, f_sin_n], device=device)
                            l_Z = torch.tensor([-f_sin_n, 0.0, f_cos_n], device=device)

                            if cid == 1: back_is_X = is_x_longer
                            else: back_is_X = not is_x_longer

                            v_to_wall = -torch.sign(c2c_d[ni]) * n_w
                            target_a_back = torch.atan2(v_to_wall[2], v_to_wall[0])
                            final_t_a = target_a_back if back_is_X else (target_a_back - math.pi/2)

                            angle_corr_cos[b, fi] = torch.cos(final_t_a) - x[b, fi, 6]
                            angle_corr_sin[b, fi] = torch.sin(final_t_a) - x[b, fi, 7]

                            if DEBUG_PRINT:
                                print(f"[Hard Snap] Slot")

        # --- C. V5 统一尺寸优化逻辑 (通用平面版) ---
        #loss_size_total = torch.tensor(0.0, device=device)
        #if use_scaling and original_raw_size is not None:
        #    for b in range(B):
        #        f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
        #        for fi in f_idx:
        #            cid = pred_class_ids[b, fi].item()
        #            curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

        #            # 1. 轴无关的底线防御 (Hinge Loss)
        #            if cid in CLASS_MIN_SIZES:
        #                m_val = CLASS_MIN_SIZES[cid].to(device)
        #                # 高度 (Y轴)
        #                loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0
        #                # 平面 (XZ轴) 排序对比，自动适配旋转
        #                curr_h, _ = torch.sort(curr_s[[0, 2]]) # [短边, 长边]
        #                min_h, _ = torch.sort(m_val[[0, 2]])   # [最小深, 最小宽]
        #                loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * 500.0

        #            # 2. 尺寸回归项 (阻力项：确保无碰撞时不缩放，且只在平面动)
        #            # X(1.0), Y(200.0 锁死高度), Z(1.0)
        #            dyn_weights = torch.tensor([1.0, 200.0, 1.0], device=device)
        #            loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 10.0
        loss_size_total = torch.tensor(0.0, device=device)
        if use_scaling and original_raw_size is not None:
            for b in range(B):
                f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
                for fi in f_idx:
                    cid = pred_class_ids[b, fi].item()
                    curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

                    if cid in CLASS_MIN_SIZES:
                        m_val = CLASS_MIN_SIZES[cid].to(device)

                        # 1. 降低内部系数 (从 1000 降到 100)，让“弹簧”变软
                        inner_k = 100.0

                        # 高度防御 (Y轴通常不缩放，可以保持硬一点)
                        loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0

                        # 平面对称防御 (XZ轴是震荡重灾区，必须变软)
                        curr_h, _ = torch.sort(curr_s[[0, 2]])
                        min_h, _ = torch.sort(m_val[[0, 2]])
                        loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * inner_k

                    # 2. 回归项：给平面缩放留出余地，但高度依然锁死
                    dyn_weights = torch.tensor([1.0, 500.0, 1.0], device=device)
                    loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 5.0

        #total_loss = loss_attach*1.0
        #total_loss = loss_size_total*1.0
        #total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*1.0 + loss_clearance*1.0 + loss_size_total*1.0

        #total_loss = loss_boundary*10.0 + loss_collision*1.0 + loss_attach*5.0 + loss_clearance*10.0 + loss_size_total*0.1
        #total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*0.05
        total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*1
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f} | "
                  f"Size: {loss_size_total.item():.6f}")

        #if total_loss <= 1e-7:
        if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
            if DEBUG_PRINT:
                print("Physical loss too small, early return")
            return None, total_loss

        total_loss.backward()

        grad_final = torch.zeros_like(x.grad)
        pos_g = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            pos_g += (snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)) * 0.8

        grad_final[:, :, 0:3] = pos_g * boundary_mask.unsqueeze(-1)
        if use_scaling:
            # 这里的 2.0 是关键，它能让家具更快地缩放以响应碰撞
            grad_final[:, :, 3:6] = x.grad[:, :, 3:6] * scale * 2.0 * can_scale_mask.unsqueeze(-1)

        # 现在，如果 use_hard_snap 为 False，angle_corr 为 0，x.grad 也为 0，家具就不会转了
        grad_final[:, :, 6] = (x.grad[:, :, 6] * scale + angle_corr_cos * 0.1) * boundary_mask
        grad_final[:, :, 7] = (x.grad[:, :, 7] * scale + angle_corr_sin * 0.1) * boundary_mask

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

    def _get_simple_iou_gradient_v8(self, x, room_min, room_max, num_partial=0,
                               original_raw_size=None, # 传入 trial=0 时的原始世界尺寸
                               scale=0.002, grad_clip=0.01, padding=0.1,
                               use_soft_snap=True, use_hard_snap=False,
                               use_scaling=True,
                               update_pos_gradient=True, update_size_gradient=True, update_rot_gradient=True):

        device = x.device
        B, N, D = x.shape

        CLASS_MIN_SIZES = {
            0: torch.tensor([0.6, 0.7, 0.4], device=device), # 最小宽度 0.6m, 深度 0.4m
            2: torch.tensor([0.8, 1.8, 0.8], device=device), # 淋浴房最小 0.8x0.8
            3: torch.tensor([1.2, 0.4, 0.7], device=device)  # 浴缸最小长度 1.2m
        }

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

        is_moveable = torch.ones_like(valid_mask); is_moveable[:, :num_partial] = 0
        #boundary_mask = valid_mask * is_moveable * is_not_floor_mask
        boundary_mask = valid_mask * is_moveable * is_furniture_mask
        can_scale_mask = is_furniture_mask * (pred_class_ids != 1) * is_moveable

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

        # 预先定义好 Boolean Mask 以便进行逻辑运算
        is_mov_furn_bool = (is_furniture_mask * is_moveable).bool()  # 可移动的家具 (ID 0-3)
        is_not_floor_bool = is_not_floor_mask.bool()                # 所有非地板物体 (ID != 4)

        loss_collision = torch.tensor(0.0, device=device)
        use_loss_collision = True

        if use_loss_collision:
            # 1. 计算所有物体对之间的 AABB 相交体积 [B, N, N]
            # mins, maxs 的 shape 为 [B, N, 3]
            inter_mins = torch.max(mins.unsqueeze(2), mins.unsqueeze(1))
            inter_maxs = torch.min(maxs.unsqueeze(2), maxs.unsqueeze(1))

            # 计算相交部分的边长，小于 0 的部分 clamp 为 0
            # product dim=-1 得到体积 (Width * Height * Depth)
            intersection_vol = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            # 2. 基础过滤掩码
            mask_valid_pair = (valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)).bool() # 双方都必须有效
            mask_self = torch.eye(N, device=device).unsqueeze(0).bool()                 # 排除物体自身 [1, N, N]

            # 3. 核心物理逻辑掩码：
            # 条件 1：至少其中一方是“可移动家具” (否则墙撞墙不需要计算梯度)
            # 条件 2：双方都必须“不是地板” (彻底解决家具踩在地板上的数值爆炸问题)
            mask_coll_target = (
                (is_mov_furn_bool.unsqueeze(2) | is_mov_furn_bool.unsqueeze(1)) &
                (is_not_floor_bool.unsqueeze(2) & is_not_floor_bool.unsqueeze(1))
            )

            # 4. 合并所有掩码并转回 Float
            # 最终掩码：双方有效 AND 非自身 AND 满足物理碰撞条件
            final_mask = (mask_valid_pair & (~mask_self) & mask_coll_target).float()

            # 5. 计算总损失
            # 除以 2.0 是因为矩阵 [i, j] 和 [j, i] 对称计算了两次体积
            loss_collision = (intersection_vol * final_mask).sum() / 2.0

        # --- B. 2D 门前净空损失 (XZ-Plane Only) ---
        loss_clearance = torch.tensor(0.0, device=device)
        use_loss_clearance = True

        if use_loss_clearance:
            clearance_depth = 0.8

            cos_a, sin_a = x[:, :, 6].abs().unsqueeze(-1), x[:, :, 7].abs().unsqueeze(-1)
            d_horiz_sizes = raw_size[:, :, [0, 2]]
            d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
            d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

            g_depth = d_thick + clearance_depth * 2.0
            g_ex = (g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
            g_ez = (g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
            g_mins_2d = pos[:, :, [0, 2]] - torch.cat([g_ex, g_ez], dim=-1)
            g_maxs_2d = pos[:, :, [0, 2]] + torch.cat([g_ex, g_ez], dim=-1)

            f_mins_2d, f_maxs_2d = pos[:, :, [0, 2]] - half_size[:, :, [0, 2]], pos[:, :, [0, 2]] + half_size[:, :, [0, 2]]
            inter_mins_2d = torch.max(f_mins_2d.unsqueeze(2), g_mins_2d.unsqueeze(1))
            inter_maxs_2d = torch.min(f_maxs_2d.unsqueeze(2), g_maxs_2d.unsqueeze(1))
            intersection_area = torch.clamp(inter_maxs_2d - inter_mins_2d, min=0.0).prod(dim=-1)

            clearance_mask = (is_furniture_mask * valid_mask * is_moveable).unsqueeze(2) * (is_door_mask * valid_mask).unsqueeze(1)
            loss_clearance = (intersection_area * clearance_mask).sum()

        # 4. 房间边界惩罚 (只对 NEW & VALID 物体)
        #loss_boundary = 0
        #if room_min is not None and room_max is not None:
        assert room_min is not None and room_max is not None
        loss_boundary = torch.tensor(0.0, device=device)
        use_loss_boundary = True
        if use_loss_boundary:
            # --- C. 房间边界惩罚 ---
            dist_min = torch.clamp(room_min.to(device) - (pos - half_size), min=0.0)
            dist_max = torch.clamp((pos + half_size) - room_max.to(device), min=0.0)
            loss_boundary = ((dist_min**2 + dist_max**2).sum(dim=-1) * boundary_mask).sum()

        loss_attach = torch.tensor(0.0, device=device)
        snap_offset = torch.zeros_like(pos)

        angle_corr_cos = torch.zeros_like(x[:, :, 6])
        angle_corr_sin = torch.zeros_like(x[:, :, 7])

        if use_soft_snap or use_hard_snap:
            true_half_size = raw_size / 2.0
            for b in range(B):
                w_idx, f_idx = torch.where((is_wall_mask[b]*valid_mask[b]) > 0.5)[0], torch.where((is_furniture_mask[b]*is_moveable[b]*valid_mask[b]) > 0.5)[0]
                if len(w_idx) > 0 and len(f_idx) > 0:
                    # 计算墙体法线 (保持不变)
                    thick_axis = torch.argmin(raw_size[b, w_idx][:, [0, 2]], dim=-1)
                    n_x = torch.stack([torch.cos(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.sin(angles[b, w_idx])], dim=-1)
                    n_z = torch.stack([-torch.sin(angles[b, w_idx]), torch.zeros_like(angles[b, w_idx]), torch.cos(angles[b, w_idx])], dim=-1)
                    normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

                    for i, fi in enumerate(f_idx):
                        # --- 关键改进 1：锁定语义方向，防止 90 度翻转 ---
                        # 判定长短轴永远参考原始尺寸，不参考正在被压扁的当前尺寸
                        orig_f_sz = original_raw_size[b, fi, [0, 2]]
                        is_x_longer = orig_f_sz[0] > orig_f_sz[1]
                        cid = pred_class_ids[b, fi].item()

                        # 寻找最近墙
                        rel_p = pos[b, fi] - pos[b, w_idx]
                        c2c_d = (rel_p * normals).sum(dim=-1)
                        f_thick_p = (true_half_size[b, fi].unsqueeze(0) * normals.abs()).sum(dim=-1)
                        gaps = torch.abs(c2c_d) - (f_thick_p + (raw_size[b, w_idx].min(dim=-1)[0]/2.0))
                        ni = torch.abs(gaps).argmin(); n_w = normals[ni]

                        # --- 关键改进 2：Soft Snap 仅计算距离 Loss ---
                        if use_soft_snap:
                            # 删除了所有的点积 (Dot Product) 计算
                            # 这样 backward 时，旋转位 (6:8) 的梯度就为 0
                            loss_attach += gaps[ni].pow(2)

                        # --- 关键改进 3：Hard Snap 才允许旋转干预 ---
                        if use_hard_snap and torch.abs(gaps[ni]) < 0.15:
                            s_gap = c2c_d[ni] - torch.sign(c2c_d[ni]) * (f_thick_p[ni] + raw_size[b, w_idx[ni]].min()/2.0)
                            snap_offset[b, fi] = -s_gap * n_w

                            # 只有在 Hard Snap 阶段，我们才去构建局部坐标系并计算目标角度
                            f_mag = torch.sqrt(x[b, fi, 6]**2 + x[b, fi, 7]**2 + 1e-8)
                            f_cos_n, f_sin_n = x[b, fi, 6]/f_mag, x[b, fi, 7]/f_mag
                            l_X = torch.tensor([f_cos_n, 0.0, f_sin_n], device=device)
                            l_Z = torch.tensor([-f_sin_n, 0.0, f_cos_n], device=device)

                            if cid == 1: back_is_X = is_x_longer
                            else: back_is_X = not is_x_longer

                            v_to_wall = -torch.sign(c2c_d[ni]) * n_w
                            target_a_back = torch.atan2(v_to_wall[2], v_to_wall[0])
                            final_t_a = target_a_back if back_is_X else (target_a_back - math.pi/2)

                            angle_corr_cos[b, fi] = torch.cos(final_t_a) - x[b, fi, 6]
                            angle_corr_sin[b, fi] = torch.sin(final_t_a) - x[b, fi, 7]

                            if DEBUG_PRINT:
                                print(f"[Hard Snap] Slot")

        # --- C. V5 统一尺寸优化逻辑 (通用平面版) ---
        #loss_size_total = torch.tensor(0.0, device=device)
        #if use_scaling and original_raw_size is not None:
        #    for b in range(B):
        #        f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
        #        for fi in f_idx:
        #            cid = pred_class_ids[b, fi].item()
        #            curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

        #            # 1. 轴无关的底线防御 (Hinge Loss)
        #            if cid in CLASS_MIN_SIZES:
        #                m_val = CLASS_MIN_SIZES[cid].to(device)
        #                # 高度 (Y轴)
        #                loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0
        #                # 平面 (XZ轴) 排序对比，自动适配旋转
        #                curr_h, _ = torch.sort(curr_s[[0, 2]]) # [短边, 长边]
        #                min_h, _ = torch.sort(m_val[[0, 2]])   # [最小深, 最小宽]
        #                loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * 500.0

        #            # 2. 尺寸回归项 (阻力项：确保无碰撞时不缩放，且只在平面动)
        #            # X(1.0), Y(200.0 锁死高度), Z(1.0)
        #            dyn_weights = torch.tensor([1.0, 200.0, 1.0], device=device)
        #            loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 10.0
        loss_size_total = torch.tensor(0.0, device=device)
        if use_scaling and original_raw_size is not None:
            for b in range(B):
                f_idx = torch.where(can_scale_mask[b] > 0.5)[0]
                for fi in f_idx:
                    cid = pred_class_ids[b, fi].item()
                    curr_s, orig_s = raw_size[b, fi], original_raw_size[b, fi]

                    if cid in CLASS_MIN_SIZES:
                        m_val = CLASS_MIN_SIZES[cid].to(device)

                        # 1. 降低内部系数 (从 1000 降到 100)，让“弹簧”变软
                        inner_k = 100.0

                        # 高度防御 (Y轴通常不缩放，可以保持硬一点)
                        loss_size_total += torch.clamp(m_val[1] - curr_s[1], min=0.0).pow(2) * 500.0

                        # 平面对称防御 (XZ轴是震荡重灾区，必须变软)
                        curr_h, _ = torch.sort(curr_s[[0, 2]])
                        min_h, _ = torch.sort(m_val[[0, 2]])
                        loss_size_total += torch.clamp(min_h - curr_h, min=0.0).pow(2).sum() * inner_k

                    # 2. 回归项：给平面缩放留出余地，但高度依然锁死
                    dyn_weights = torch.tensor([1.0, 500.0, 1.0], device=device)
                    loss_size_total += ((curr_s - orig_s).pow(2) * dyn_weights).sum() * 5.0

        #total_loss = loss_attach*1.0
        #total_loss = loss_size_total*1.0
        #total_loss = loss_boundary*1.0 + loss_collision*1.0 + loss_attach*1.0 + loss_clearance*1.0 + loss_size_total*1.0

        #total_loss = loss_boundary*10.0 + loss_collision*1.0 + loss_attach*5.0 + loss_clearance*10.0 + loss_size_total*0.1
        #total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*0.05
        total_loss = loss_boundary*5.0 + loss_collision*8.0 + loss_attach*5.0 + loss_clearance*80.0 + loss_size_total*1
        if DEBUG_PRINT:
            # 提取数值，注意处理 loss_boundary 可能是 0 的情况
            print(f"[Physics Loss] Total: {total_loss.item():.6f} | "
                  f"Coll: {loss_collision.item():.6f} | Clear: {loss_clearance.item():.6f} | "
                  f"Bound: {loss_boundary.item():.6f} | Snap: {loss_attach.item():.6f} | "
                  f"Size: {loss_size_total.item():.6f}")

        #if total_loss <= 1e-7:
        if total_loss <= 1e-7 and snap_offset.abs().sum() < 1e-7:
            if DEBUG_PRINT:
                print("Physical loss too small, early return")
            return None, total_loss

        total_loss.backward()

        grad_final = torch.zeros_like(x.grad)
        pos_g = x.grad[:, :, 0:3] * scale
        if use_hard_snap:
            w_range = (self.diffusion._centroids_max - self.diffusion._centroids_min).to(device)
            pos_g += (snap_offset / (w_range / 2.0).unsqueeze(0).unsqueeze(0)) * 0.8

        if update_pos_gradient:
            grad_final[:, :, 0:3] = pos_g * boundary_mask.unsqueeze(-1)

        if use_scaling and update_size_gradient:
            # 这里的 2.0 是关键，它能让家具更快地缩放以响应碰撞
            #grad_final[:, :, 3:6] = x.grad[:, :, 3:6] * scale * 2.0 * can_scale_mask.unsqueeze(-1)
            # 增加 Langevin 噪声：只在尺寸开启初期加入，防止缩放死锁
            noise = torch.randn_like(x.grad[:, :, 3:6]) * scale * 0.1
            grad_final[:, :, 3:6] = (x.grad[:, :, 3:6] * scale * 2.0 + noise) * can_scale_mask.unsqueeze(-1)

        # 现在，如果 use_hard_snap 为 False，angle_corr 为 0，x.grad 也为 0，家具就不会转了
        if use_hard_snap and update_rot_gradient:
            grad_final[:, :, 6] = (x.grad[:, :, 6] * scale + angle_corr_cos * 0.1) * boundary_mask
            grad_final[:, :, 7] = (x.grad[:, :, 7] * scale + angle_corr_sin * 0.1) * boundary_mask

        return -torch.clamp(grad_final, -grad_clip*2, grad_clip*2), total_loss.item()

    def _apply_arch_item_removal(self, model_mean, pred_xstart, num_partial=0):
        B, total_slots, D = model_mean.shape
        assert num_partial < total_slots

        remove_count = 0
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
                if cid in ARCH_CLASS_IDS and cid != CLASS_ID_VALS['empty']:
                    remove_count += 1
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

        if remove_count > 0:
            # 写回修改后的均值
            model_mean[:, num_partial:, :] = new_gen_section
        return model_mean, remove_count

    def _apply_hard_physical_snap(self, x, num_partial=0):
        """
        物理硬投影 v4 (Ghost Clearance 版):
        使用 1.6m 厚度的门禁幽灵盒 (Ghost Box) 来进行碰撞拦截。
        """
        device = x.device
        B, N, D = x.shape

        # 1. 基础信息提取
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)

        o_idx = 16
        valid_mask = (x[:, :, o_idx] <= 0).float()
        is_wall_mask = (pred_class_ids == 5).float()
        is_door_mask = (pred_class_ids == 6).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_toilet_mask = (pred_class_ids == 1).float()

        # 2. 坐标转换
        pos = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
        raw_size = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
        curr_angles = torch.atan2(x[:, :, 7], x[:, :, 6])

        new_pos = pos.clone()
        new_cos = x[:, :, 6].clone()
        new_sin = x[:, :, 7].clone()

        moved_report = []

        for b in range(B):
            w_indices = torch.where((is_wall_mask[b] * valid_mask[b]) > 0.5)[0]
            d_indices = torch.where((is_door_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = torch.where((is_furniture_mask[b] * valid_mask[b]) > 0.5)[0]
            f_indices = f_indices[f_indices >= num_partial]

            if len(w_indices) == 0 or len(f_indices) == 0: continue

            # --- 1. 预计算该 Batch 下所有门的 Ghost Box AABB ---
            ghost_mins_2d = []
            ghost_maxs_2d = []
            if len(d_indices) > 0:
                clearance_depth = 0.8
                d_pos_2d = pos[b, d_indices][:, [0, 2]]
                d_horiz_sizes = raw_size[b, d_indices][:, [0, 2]]

                # 旋转感知
                cos_a = x[b, d_indices, 6].abs().unsqueeze(-1)
                sin_a = x[b, d_indices, 7].abs().unsqueeze(-1)

                d_width = d_horiz_sizes.max(dim=-1)[0].unsqueeze(-1)
                d_thick = d_horiz_sizes.min(dim=-1)[0].unsqueeze(-1)

                # 总厚度 = 门厚 + 前后各 0.8m
                total_g_depth = d_thick + clearance_depth * 2.0

                # 计算幽灵盒的 AABB 包络半径
                g_ex = (total_g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
                g_ez = (total_g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a
                g_half_size_2d = torch.cat([g_ex, g_ez], dim=-1)

                ghost_mins_2d = d_pos_2d - g_half_size_2d
                ghost_maxs_2d = d_pos_2d + g_half_size_2d

            # 计算墙体法线 (保持 v2 逻辑)
            w_sizes = raw_size[b, w_indices]
            thick_axis = torch.argmin(w_sizes[:, [0, 2]], dim=-1)
            w_angles = torch.atan2(x[b, w_indices, 7], x[b, w_indices, 6])
            n_x = torch.stack([torch.cos(w_angles), torch.zeros_like(w_angles), torch.sin(w_angles)], dim=-1)
            n_z = torch.stack([-torch.sin(w_angles), torch.zeros_like(w_angles), torch.cos(w_angles)], dim=-1)
            w_normals = (1.0 - thick_axis.unsqueeze(-1).float()) * n_x + thick_axis.unsqueeze(-1).float() * n_z

            for fi in f_indices:
                f_pos, f_sz = pos[b, fi], raw_size[b, fi]
                is_toilet = is_toilet_mask[b, fi] > 0.5

                # 墙体排序
                rel_p = f_pos - pos[b, w_indices]
                dist_to_walls = (rel_p * w_normals).sum(dim=-1)
                f_half_thick = (f_sz.unsqueeze(0) * w_normals.abs()).sum(dim=-1) / 2.0
                w_half_thick = w_sizes.min(dim=-1)[0] / 2.0
                gaps = torch.abs(dist_to_walls) - (f_half_thick + w_half_thick)
                sorted_wall_cand = torch.argsort(torch.abs(gaps))

                best_ni = None
                for cand_i in sorted_wall_cand:
                    ni = cand_i.item()
                    # 模拟投影
                    push_dir = -torch.sign(dist_to_walls[ni])
                    temp_pos = f_pos + push_dir * gaps[ni] * w_normals[ni]

                    # --- 2. 核心：使用 Ghost Box 进行堵门校验 ---
                    is_blocking = False
                    if len(d_indices) > 0:
                        # 当前家具的 2D AABB
                        f_min_2d = temp_pos[[0, 2]] - (f_sz[[0, 2]] / 2.0)
                        f_max_2d = temp_pos[[0, 2]] + (f_sz[[0, 2]] / 2.0)

                        # 与所有幽灵盒比对
                        for di in range(len(d_indices)):
                            overlap_x = (f_min_2d[0] < ghost_maxs_2d[di, 0]) and (f_max_2d[0] > ghost_mins_2d[di, 0])
                            overlap_z = (f_min_2d[1] < ghost_maxs_2d[di, 1]) and (f_max_2d[1] > ghost_mins_2d[di, 1])

                            if overlap_x and overlap_z:
                                is_blocking = True
                                if DEBUG_PRINT:
                                    print(f"    - Wall {ni} REJECTED: Slot {fi.item()} hits Door {d_indices[di].item()} Ghost Box")
                                break

                    if not is_blocking:
                        best_ni = ni
                        break

                # --- 3. 执行更新 (保持 v2 逻辑) ---
                if best_ni is not None:
                    ni = best_ni
                    target_n = w_normals[ni]
                    actual_gap = gaps[ni]

                    pos_changed = False
                    if torch.abs(actual_gap) > 0.001:
                        push_dir = -torch.sign(dist_to_walls[ni])
                        new_pos[b, fi] += push_dir * actual_gap * target_n
                        pos_changed = True

                    rot_changed = False
                    if is_toilet:
                        v_to_room = -torch.sign(dist_to_walls[ni]) * target_n
                        target_angle = torch.atan2(v_to_room[2], v_to_room[0])
                        final_a = target_angle - 1.5708
                        angle_diff = torch.atan2(torch.sin(curr_angles[b, fi] - final_a), torch.cos(curr_angles[b, fi] - final_a)).abs()
                        if angle_diff > 0.017:
                            new_cos[b, fi], new_sin[b, fi] = torch.cos(final_a), torch.sin(final_a)
                            rot_changed = True

                    if pos_changed or rot_changed:
                        moved_report.append({"id": fi.item(), "cat": pred_class_ids[b, fi].item(), "gap": actual_gap.item(), "pos_mv": pos_changed, "rot_mv": rot_changed})

        # --- 4. Rescale 并回填 ---
        x_new = x.clone()
        c_min, c_max = self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)
        x_new[:, :, 0:3] = (new_pos - c_min) / (c_max - c_min) * 2.0 - 1.0
        x_new[:, :, 6] = new_cos
        x_new[:, :, 7] = new_sin

        # 4. 输出变动清单
        if DEBUG_PRINT and len(moved_report) > 0:
            print(f"\n[Hard Projection Activity] Found {len(moved_report)} items needing adjustment:")
            for item in moved_report:
                action = []
                if item["pos_mv"]: action.append(f"AttachedToWall(Gap:{item['gap']*100:.1f}cm)")
                if item["rot_mv"]: action.append("RotatedToPerpendicular")
                print(f"  - Slot {item['id']} (Cat:{item['cat']}): {' & '.join(action)}")

        return x_new

    def _precompute_ghost_boxes(self, x, valid_mask, pred_class_ids, device):
        device = x.device
        B, N, D = x.shape
        ghost_mins_list, ghost_maxs_list = [], []
        with torch.no_grad():
            raw_sz_all = self.diffusion.descale_to_origin(x[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
            pos_all = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
            is_door_mask = (pred_class_ids == 6).float()

            for b in range(B):
                d_idx = torch.where((is_door_mask[b] * valid_mask[b]) > 0.5)[0]
                if len(d_idx) > 0:
                    d_p = pos_all[b, d_idx][:, [0, 2]]      # [Nd, 2]
                    d_s = raw_sz_all[b, d_idx][:, [0, 2]]   # [Nd, 2]

                    # 1. 找到厚度轴 (0=X, 1=Z)
                    thick_axis = torch.argmin(d_s, dim=-1) # [Nd]

                    d_w = d_s.max(dim=-1)[0] # 门宽
                    d_t = d_s.min(dim=-1)[0] # 门厚

                    # 2. 定义本地半长 (本地坐标系)
                    half_w = d_w / 2.0
                    half_g_t = (d_t + 1.6) / 2.0 # 增加 1.6m 净空

                    # 3. 旋转分量
                    cos_a = x[b, d_idx, 6].abs()
                    sin_a = x[b, d_idx, 7].abs()

                    # 4. 根据厚度轴向进行世界坐标投影 (AABB Envelope)
                    # 如果 thick_axis == 1 (Z是厚度):
                    #    g_ex = half_w * cos + half_g_t * sin
                    #    g_ez = half_w * sin + half_g_t * cos
                    # 如果 thick_axis == 0 (X是厚度): 反之

                    mask_z = (thick_axis == 1).float()
                    mask_x = (thick_axis == 0).float()

                    g_ex = mask_z * (half_w * cos_a + half_g_t * sin_a) + \
                           mask_x * (half_g_t * cos_a + half_w * sin_a)

                    g_ez = mask_z * (half_w * sin_a + half_g_t * cos_a) + \
                           mask_x * (half_g_t * sin_a + half_w * cos_a)

                    g_half = torch.stack([g_ex, g_ez], dim=-1) # [Nd, 2]

                    ghost_mins_list.append(d_p - g_half)
                    ghost_maxs_list.append(d_p + g_half)
                else:
                    ghost_mins_list.append(None)
                    ghost_maxs_list.append(None)

        return ghost_mins_list, ghost_maxs_list

    def _apply_physical_relax(self, x, room_min, room_max, num_partial=0):
        """
        Step 2: Relax (能量最小化平滑 - 修正统计版)
        """
        device = x.device
        B, N, D = x.shape

        # --- 1. 基础信息提取 ---
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)
        valid_mask = (x[:, :, 16] <= 0).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_not_floor_mask = (pred_class_ids != 4).float()

        # 记录初始位置作为正则化锚点
        pos_start = self.diffusion.descale_to_origin(
            x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)
        ).detach()

        x_relaxed = x.clone().detach().requires_grad_(True)
        # 只有 index >= num_partial 的家具允许移动
        move_mask = is_furniture_mask.clone()
        move_mask[:, :num_partial] = 0

        lr = 0.002
        #lr = 0.005
        #lr = 0.02

        #grad_clip = 0.005
        #grad_clip = 0.01
        grad_clip = 0.05

        eps = 1e-4  # 面积判定阈值

        # --- 2. 预计算 Door Ghost Boxes (法线感知修正版) ---
        ghost_mins_list, ghost_maxs_list = self._precompute_ghost_boxes(x, valid_mask, pred_class_ids, device)

        print(f"\n[Relax Phase] Active Items: {int(move_mask.sum())} | Start Optimizing...")

        best_loss = float('inf')
        best_total_issues = float('inf')
        best_num_coll = 0
        best_num_oob = 0
        best_num_ghost = 0
        best_x = x_relaxed.clone().detach()
        best_iter = 0

        modified = False

        # --- 3. 优化迭代循环 ---
        max_iters = 200
        for cur_iter in range(max_iters):
            if x_relaxed.grad is not None: x_relaxed.grad.zero_()

            curr_pos = self.diffusion.descale_to_origin(x_relaxed[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
            curr_sz = self.diffusion.descale_to_origin(x_relaxed[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
            f_mins, f_maxs = curr_pos[:, :, [0, 2]] - curr_sz[:, :, [0, 2]] / 2.0, curr_pos[:, :, [0, 2]] + curr_sz[:, :, [0, 2]] / 2.0

            # A. Boundary Loss & Count (只计家具)
            dist_min = torch.clamp(room_min[[0, 2]] - f_mins, min=0.0)
            dist_max = torch.clamp(f_maxs - room_max[[0, 2]], min=0.0)
            # Loss 只施加在能动的物体上
            loss_bound = (dist_min.pow(2) + dist_max.pow(2)).sum(dim=-1)
            loss_bound = (loss_bound * move_mask).sum() * 50.0
            # 计数只计家具
            oob_mask = ((dist_min > 0.01).any(dim=-1) | (dist_max > 0.01).any(dim=-1)) * is_furniture_mask * valid_mask
            num_oob = oob_mask.sum().item()

            # B. Collision Loss & Count (只计涉及家具的碰撞)
            inter_mins = torch.max(f_mins.unsqueeze(2), f_mins.unsqueeze(1))
            inter_maxs = torch.min(f_maxs.unsqueeze(2), f_maxs.unsqueeze(1))
            overlap_area = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)

            # 基础掩码：非自身，非地板，有效
            base_coll_mask = (valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)) * \
                             (1.0 - torch.eye(N, device=device).unsqueeze(0)) * \
                             (is_not_floor_mask.unsqueeze(2) * is_not_floor_mask.unsqueeze(1))

            # 碰撞必须涉及家具 (家具 vs 任何有效非地板物体)
            is_f = is_furniture_mask.unsqueeze(2).bool() | is_furniture_mask.unsqueeze(1).bool()
            final_coll_mask = base_coll_mask * is_f.float()

            # Loss 只施加在“至少有一个能动”的碰撞对上
            is_m = move_mask.unsqueeze(2).bool() | move_mask.unsqueeze(1).bool()
            active_coll_mask = final_coll_mask * is_m.float()

            loss_coll = (overlap_area * active_coll_mask).sum() * 100.0
            num_coll = ((overlap_area * final_coll_mask) > eps).sum().item() / 2

            # C. Ghost Box Loss & Count
            loss_ghost, num_ghost = 0.0, 0
            for b in range(B):
                if ghost_mins_list[b] is not None:
                    g_min, g_max = ghost_mins_list[b], ghost_maxs_list[b]
                    gi_mins = torch.max(f_mins[b].unsqueeze(1), g_min.unsqueeze(0))
                    gi_maxs = torch.min(f_maxs[b].unsqueeze(1), g_max.unsqueeze(0))
                    gi_area = torch.clamp(gi_maxs - gi_mins, min=0.0).prod(dim=-1)
                    # 只有“家具撞门”且“家具能动”才产生 Loss
                    g_mask = is_furniture_mask[b].unsqueeze(1) * valid_mask[b].unsqueeze(1)
                    num_ghost += ((gi_area * g_mask) > eps).sum().item()
                    loss_ghost += (gi_area * g_mask * move_mask[b].unsqueeze(1)).sum()
            loss_ghost = loss_ghost * 100.0

            # D. Regularization
            loss_reg = ((curr_pos - pos_start).pow(2).sum(dim=-1) * move_mask).sum() * 10.0

            total_loss = loss_bound + loss_coll + loss_ghost + loss_reg
            if cur_iter % 20 == 0 or cur_iter == max_iters - 1:
                print(f"  Relax iter {cur_iter} | Loss: {total_loss.item():.4f} | OOB: {int(num_oob)} | Coll: {int(num_coll)} | Ghost: {int(num_ghost)}")

            total_issues = num_coll + num_oob + num_ghost
            if total_issues <= best_total_issues and total_loss < best_loss:
                best_loss = total_loss.item()
                best_x = x_relaxed.clone().detach()
                best_num_coll = num_coll
                best_num_oob = num_oob
                best_total_issues = total_issues
                best_iter = cur_iter

            if total_loss < 1e-7 and num_coll == 0 and num_oob == 0 and num_ghost == 0:
                break

            modified = True
            total_loss.backward()

            with torch.no_grad():
                grad_pos = torch.clamp(x_relaxed.grad[:, :, 0:3], -grad_clip, grad_clip)
                x_relaxed[:, :, 0:3] -= lr * grad_pos * move_mask.unsqueeze(-1)
                x_relaxed.grad.zero_()

        print(f"Phy relax modified:{modified}  Best Iter {best_iter} | Loss: {best_loss:.4f} | OOB: {int(best_num_oob)} | Coll: {int(best_num_coll)} | Ghost: {int(best_num_ghost)}")
        return best_x, best_total_issues, modified

    def _apply_physical_squeeze(self, x, room_min, room_max, num_partial=0):
        """
        Step 3: Squeeze (尺寸压缩 - 完整约束版)
        """
        device = x.device
        B, N, D = x.shape

        # --- 1. 基础信息提取 (同前) ---
        class_range = slice(self.diffusion.bbox_dim, self.diffusion.bbox_dim + self.diffusion.class_dim)
        pred_class_ids = torch.argmax(x[:, :, class_range], dim=-1)
        valid_mask = (x[:, :, 16] <= 0).float()
        is_squeeze_cat = ((pred_class_ids == 0) | (pred_class_ids == 2) | (pred_class_ids == 3)).float()
        is_furniture_mask = (pred_class_ids <= 3).float()
        is_not_floor_mask = (pred_class_ids != 4).float()

        # 最小尺寸阈值定义
        class_min_dict = {
            0: torch.tensor([0.6, 0.4], device=device), # Vanity
            2: torch.tensor([0.8, 0.8], device=device), # Shower
            3: torch.tensor([1.2, 0.7], device=device)  # Tub
        }
        min_sz_limit = torch.zeros((B, N, 2), device=device)
        for cat_id, sz in class_min_dict.items():
            mask = (pred_class_ids == cat_id).float()
            min_sz_limit += mask.unsqueeze(-1) * sz.unsqueeze(0).unsqueeze(0)
        min_sz_sorted, _ = torch.sort(min_sz_limit, dim=-1)

        # 锚点记录
        pos_start = self.diffusion.descale_to_origin(x[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device)).detach()

        x_squeezed = x.clone().detach().requires_grad_(True)
        squeeze_mask = is_squeeze_cat.clone()
        squeeze_mask[:, :num_partial] = 0
        squeeze_mask = squeeze_mask.unsqueeze(-1) # [B, N, 1]

        #lr = 0.005
        lr = 0.002
        #grad_clip = 0.02
        grad_clip = 0.05
        eps = 1e-4

        # --- 2. 预计算 Door Ghost Boxes (必须在这里跑一次) ---
        ghost_mins_list, ghost_maxs_list = self._precompute_ghost_boxes(x, valid_mask, pred_class_ids, device)

        print(f"\n[Squeeze Phase] Active Items: {int(squeeze_mask.sum())} | Entry constraints included.")

        best_loss = float('inf')
        best_x = x_squeezed.clone().detach()

        best_total_issues = float('inf')
        best_num_coll = 0
        best_num_oob = 0
        best_num_ghost = 0
        best_iter = 0

        modified = False

        # --- 3. 优化迭代循环 ---
        max_iters = 1000
        for cur_iter in range(max_iters):
            if x_squeezed.grad is not None: x_squeezed.grad.zero_()

            curr_pos = self.diffusion.descale_to_origin(x_squeezed[:, :, 0:3], self.diffusion._centroids_min.to(device), self.diffusion._centroids_max.to(device))
            curr_sz = self.diffusion.descale_to_origin(x_squeezed[:, :, 3:6], self.diffusion._sizes_min.to(device), self.diffusion._sizes_max.to(device))
            f_mins, f_maxs = curr_pos[:, :, [0, 2]] - curr_sz[:, :, [0, 2]] / 2.0, curr_pos[:, :, [0, 2]] + curr_sz[:, :, [0, 2]] / 2.0

            # A. Boundary Loss
            dist_min = torch.clamp(room_min[[0, 2]] - f_mins, min=0.0)
            dist_max = torch.clamp(f_maxs - room_max[[0, 2]], min=0.0)
            loss_bound = (dist_min.pow(2) + dist_max.pow(2)).sum(dim=-1)
            loss_bound = (loss_bound * squeeze_mask.squeeze(-1)).sum() * 100.0

            # B. Collision Loss
            inter_mins = torch.max(f_mins.unsqueeze(2), f_mins.unsqueeze(1))
            inter_maxs = torch.min(f_maxs.unsqueeze(2), f_maxs.unsqueeze(1))
            overlap_area = torch.clamp(inter_maxs - inter_mins, min=0.0).prod(dim=-1)
            base_coll_mask = (valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)) * (1.0 - torch.eye(N, device=device).unsqueeze(0)) * (is_not_floor_mask.unsqueeze(2) * is_not_floor_mask.unsqueeze(1))
            f_inv = (is_furniture_mask.unsqueeze(2).bool() | is_furniture_mask.unsqueeze(1).bool()).float()
            # 只要对撞中有一个在 squeeze_mask 里，就计入 Loss
            act_coll_m = base_coll_mask * f_inv * ((squeeze_mask.unsqueeze(2) == 1) | (squeeze_mask.unsqueeze(1) == 1)).float()
            loss_coll = (overlap_area * act_coll_m).sum() * 200.0

            # C. Ghost Box Loss (这里补上了！)
            loss_ghost, num_ghost = 0.0, 0
            for b in range(B):
                if ghost_mins_list[b] is not None:
                    g_min, g_max = ghost_mins_list[b], ghost_maxs_list[b]
                    gi_mins = torch.max(f_mins[b].unsqueeze(1), g_min.unsqueeze(0))
                    gi_maxs = torch.min(f_maxs[b].unsqueeze(1), g_max.unsqueeze(0))
                    gi_area = torch.clamp(gi_maxs - gi_mins, min=0.0).prod(dim=-1)
                    # 掩码：是家具且有效
                    g_mask = is_furniture_mask[b].unsqueeze(1) * valid_mask[b].unsqueeze(1)
                    # 只有可动的家具产生 Loss
                    loss_ghost += (gi_area * g_mask * squeeze_mask[b].unsqueeze(1)).sum()
                    num_ghost += ((gi_area * g_mask) > eps).sum().item()
            loss_ghost = loss_ghost * 200.0 # 提高权重，门禁非常重要

            # D. Size Hinge Loss (最小尺寸约束)
            curr_sz_xz = curr_sz[:, :, [0, 2]]
            curr_sz_sorted, _ = torch.sort(curr_sz_xz, dim=-1)
            size_violation = torch.clamp(min_sz_sorted - curr_sz_sorted, min=0.0)
            loss_min_size = (size_violation.pow(2).sum(dim=-1) * squeeze_mask.squeeze(-1)).sum() * 1000.0

            # E. Regularization
            loss_reg = ((curr_pos - pos_start).pow(2).sum(dim=-1) * squeeze_mask.squeeze(-1)).sum() * 5.0

            total_loss = loss_bound + loss_coll + loss_ghost + loss_min_size + loss_reg

            # 综合计数
            num_oob = (((dist_min > 0.01).any(dim=-1) | (dist_max > 0.01).any(dim=-1)) * valid_mask * is_furniture_mask).sum().item()
            num_coll = ((overlap_area * base_coll_mask * f_inv) > eps).sum().item() / 2
            total_issues = num_oob + num_coll + num_ghost

            if total_issues <= best_total_issues and total_loss < best_loss:
                best_loss = total_loss.item()
                best_x = x_squeezed.clone().detach()
                best_total_issues = total_issues
                best_num_oob = num_oob
                best_num_coll = num_coll
                best_num_ghost = num_ghost
                best_iter = cur_iter

            if total_issues == 0 and total_loss < 1e-6: break
            modified = True

            total_loss.backward()

            with torch.no_grad():
                # 更新 0:6 (Pos + Size)
                grad = torch.clamp(x_squeezed.grad[:, :, 0:6], -grad_clip, grad_clip)
                x_squeezed[:, :, 0:6] -= lr * grad * squeeze_mask
                x_squeezed.grad.zero_()

            if cur_iter % 20 == 0 or cur_iter == max_iters - 1:
                print(f"  Squeeze iter {cur_iter} | Loss: {total_loss.item():.4f} | OOB: {int(num_oob)} | Coll: {int(num_coll)} | Ghost: {int(num_ghost)}")

        print(f"Phy squeeze modified:{modified}  Best Iter {best_iter} | Loss: {best_loss:.4f} | OOB: {int(best_num_oob)} | Coll: {int(best_num_coll)} | Ghost: {int(best_num_ghost)}")
        return best_x, best_total_issues, modified

    def _apply_physical_changes(self, model_mean, room_min, room_max, num_partial=0):
        remaining_issues = 0
        total_modified = 0
        max_trials = 3
        for trial in range(max_trials):
            relax_modified = False
            squeeze_modified = False
            with torch.enable_grad():
                if True:
                    print(f"[{trial+1}/{max_trials}] Apply physical relax")
                    new_model_mean, total_issues, relax_modified = self._apply_physical_relax(model_mean, room_min, room_max, num_partial=num_partial)
                    if relax_modified:
                        total_modified += 1
                        model_mean = new_model_mean
                        remaining_issues = total_issues
                if remaining_issues > 0:
                    print(f"[{trial+1}/{max_trials}] Apply physical squeeze to fix remaining_issues {remaining_issues}")
                    new_model_mean, total_slots, squeeze_modified = self._apply_physical_squeeze(model_mean, room_min, room_max, num_partial=num_partial)
                    if squeeze_modified:
                        total_modified += 1
                        model_mean = new_model_mean
                        remaining_issues = total_issues
            if squeeze_modified:
                with torch.no_grad():
                    model_mean = self._apply_physical_wall_attach(model_mean, num_partial=num_partial)
                    total_modified += 1
            else:
                break
        return model_mean, remaining_issues, True
