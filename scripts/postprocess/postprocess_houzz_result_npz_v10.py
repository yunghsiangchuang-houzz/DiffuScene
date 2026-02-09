import argparse
import os
import numpy as np
import math
from tqdm import tqdm

# ================= 1. 配置与标签 =================
CLASS_LABELS = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window", "empty"]
FURNITURE_CLASSES = {"vanity", "toilet", "shower", "tub"}
ARCHITECTURE_CLASSES = {"wall", "door", "window"}  # 用于bounds计算

# 门clearance距离（米）
DOOR_CLEARANCE = 0.8  # 门两侧各留0.8米的clearance

# 家具最小尺寸（米）- (长, 宽)
MIN_FURNITURE_SIZES = {
    'vanity': (0.6, 0.4),
    'shower': (0.8, 0.8),
    'tub': (1.2, 0.6),
}

# 可以squeeze的家具类型
SQUEEZABLE_FURNITURE = {'vanity', 'shower', 'tub'}

# 家具在bounds内的最小比例（低于此值将被删除）
MIN_INSIDE_RATIO = 0.3  # 至少30%在bounds内

# 严重collision的阈值（overlap比例）
SEVERE_COLLISION_THRESHOLDS = {
    'furniture_wall': 0.3,       # 家具vs墙：30%重叠算严重
    'furniture_clearance': 0.5,  # 家具vs门clearance：50%重叠算严重
    'furniture_furniture': 0.2,  # 家具vs家具：40%重叠算严重
}

# Squeeze时添加的额外gap（米）
SQUEEZE_GAPS = {
    'furniture_wall': 0.05,       # 5cm - 墙要严格一些
    'furniture_clearance': 0.02,  # 2cm - clearance可以稍微宽松
    'furniture_furniture': 0.03,  # 3cm - 家具之间适中
}

# 家具删除优先级（数字越小越先删除）
FURNITURE_REMOVAL_PRIORITY = {
    'shower': 1,
    'tub': 1,
    'vanity': 2,
    'toilet': 3,  # 最后才删除
}

# ================= 2. Collision Reduction 辅助函数 =================

def calculate_aabb_2d(pos, size, angle):
    """计算2D AABB"""
    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))
    half_w = (cos_a * size[0] + sin_a * size[2]) / 2.0
    half_d = (sin_a * size[0] + cos_a * size[2]) / 2.0
    return (pos[0] - half_w, pos[0] + half_w, pos[2] - half_d, pos[2] + half_d)

def aabb_overlap_area(aabb1, aabb2):
    """计算AABB重叠面积"""
    min_x1, max_x1, min_z1, max_z1 = aabb1
    min_x2, max_x2, min_z2, max_z2 = aabb2

    overlap_min_x = max(min_x1, min_x2)
    overlap_max_x = min(max_x1, max_x2)
    overlap_min_z = max(min_z1, min_z2)
    overlap_max_z = min(max_z1, max_z2)

    if overlap_max_x <= overlap_min_x or overlap_max_z <= overlap_min_z:
        return 0.0
    return (overlap_max_x - overlap_min_x) * (overlap_max_z - overlap_min_z)

def calculate_door_clearance_boxes(data):
    """
    计算所有门的clearance boxes
    返回: list of door clearance AABB
    """
    translations = data['translations']
    sizes = data['sizes']
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    clearance_boxes = []

    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue
        label_name = CLASS_LABELS[cls_indices[i]]
        if label_name != "door":
            continue

        pos = translations[i]
        size = sizes[i]
        ang = get_angle(angles[i])

        # 识别门的长边（宽度）和短边（厚度）
        door_width = max(size[0], size[2])  # 长边
        door_thick = min(size[0], size[2])  # 短边

        # 计算clearance box
        # clearance沿着门的厚度方向延伸，两侧各DOOR_CLEARANCE
        total_clearance_depth = door_thick + 2 * DOOR_CLEARANCE

        # 考虑旋转后的AABB
        cos_a = abs(math.cos(ang))
        sin_a = abs(math.sin(ang))

        # 根据门的朝向计算clearance box的半尺寸
        if size[0] < size[2]:
            # X是厚度轴，Z是宽度轴
            # clearance在X方向延伸
            half_x = total_clearance_depth / 2.0
            half_z = door_width / 2.0
        else:
            # Z是厚度轴，X是宽度轴
            # clearance在Z方向延伸
            half_x = door_width / 2.0
            half_z = total_clearance_depth / 2.0

        # 考虑旋转
        aabb_half_w = (cos_a * half_x + sin_a * half_z)
        aabb_half_d = (sin_a * half_x + cos_a * half_z)

        clearance_aabb = (
            pos[0] - aabb_half_w,
            pos[0] + aabb_half_w,
            pos[2] - aabb_half_d,
            pos[2] + aabb_half_d
        )

        clearance_boxes.append({
            'door_index': i,
            'aabb': clearance_aabb
        })

    return clearance_boxes

def calculate_bounds(data):
    """基于architecture items计算房间边界"""
    translations = data['translations']
    sizes = data['sizes']
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    min_x = float('inf')
    max_x = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')
    found_arch = False

    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue
        label_name = CLASS_LABELS[cls_indices[i]]
        if label_name not in ARCHITECTURE_CLASSES:
            continue

        found_arch = True
        pos = translations[i]
        size = sizes[i]
        ang = get_angle(angles[i])
        aabb = calculate_aabb_2d(pos, size, ang)

        min_x = min(min_x, aabb[0])
        max_x = max(max_x, aabb[1])
        min_z = min(min_z, aabb[2])
        max_z = max(max_z, aabb[3])

    if not found_arch:
        return (-5.0, 5.0, -5.0, 5.0)

    return (min_x, max_x, min_z, max_z)

def detect_collisions(data, bounds):
    """检测所有collision，包括门clearance"""
    translations = data['translations']
    sizes = data['sizes']
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    collisions = []
    furniture = []
    walls = []

    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue
        label_name = CLASS_LABELS[cls_indices[i]]
        if label_name == "empty":
            continue

        pos = translations[i]
        size = sizes[i]
        ang = get_angle(angles[i])
        aabb = calculate_aabb_2d(pos, size, ang)

        obj_info = {'index': i, 'label': label_name, 'pos': pos, 'size': size, 'aabb': aabb}

        if label_name in FURNITURE_CLASSES:
            furniture.append(obj_info)
        elif label_name == "wall":
            walls.append(obj_info)

    # 1. 家具 vs 墙
    for furn in furniture:
        for wall in walls:
            overlap = aabb_overlap_area(furn['aabb'], wall['aabb'])
            if overlap > 0.001:
                collisions.append({'type': 'furniture_wall', 'furniture': furn, 'wall': wall, 'overlap_area': overlap})

    # 2. 家具 vs 家具
    for i, f1 in enumerate(furniture):
        for f2 in furniture[i+1:]:
            overlap = aabb_overlap_area(f1['aabb'], f2['aabb'])
            if overlap > 0.001:
                collisions.append({'type': 'furniture_furniture', 'furniture1': f1, 'furniture2': f2, 'overlap_area': overlap})

    # 3. 家具 vs 门clearance (新增)
    door_clearances = calculate_door_clearance_boxes(data)
    for furn in furniture:
        for door_clear in door_clearances:
            overlap = aabb_overlap_area(furn['aabb'], door_clear['aabb'])
            if overlap > 0.001:
                collisions.append({
                    'type': 'furniture_clearance',
                    'furniture': furn,
                    'door_clearance': door_clear,
                    'overlap_area': overlap
                })

    # 4. Out of bounds
    min_x, max_x, min_z, max_z = bounds
    for furn in furniture:
        fmin_x, fmax_x, fmin_z, fmax_z = furn['aabb']
        oob = 0.0
        if fmin_x < min_x: oob += min_x - fmin_x
        if fmax_x > max_x: oob += fmax_x - max_x
        if fmin_z < min_z: oob += min_z - fmin_z
        if fmax_z > max_z: oob += fmax_z - max_z

        if oob > 0.001:
            collisions.append({'type': 'out_of_bounds', 'furniture': furn, 'bounds': bounds, 'overlap_area': oob})

    return collisions

def calculate_separation(collision, bounds=None):
    """计算分离向量"""
    if collision['type'] == 'furniture_wall':
        f_aabb = collision['furniture']['aabb']
        w_aabb = collision['wall']['aabb']

        seps = [
            (abs(w_aabb[0] - f_aabb[1]), w_aabb[0] - f_aabb[1], 0),  # 左
            (abs(f_aabb[0] - w_aabb[1]), f_aabb[0] - w_aabb[1], 0),  # 右
            (abs(w_aabb[2] - f_aabb[3]), 0, w_aabb[2] - f_aabb[3]),  # 下
            (abs(f_aabb[2] - w_aabb[3]), 0, f_aabb[2] - w_aabb[3])   # 上
        ]
        seps.sort(key=lambda x: x[0])
        return np.array([seps[0][1], 0, seps[0][2]])

    elif collision['type'] == 'furniture_clearance':
        # 家具 vs 门clearance：推离clearance区域
        f_aabb = collision['furniture']['aabb']
        c_aabb = collision['door_clearance']['aabb']

        seps = [
            (abs(c_aabb[0] - f_aabb[1]), c_aabb[0] - f_aabb[1], 0),  # 左
            (abs(f_aabb[0] - c_aabb[1]), f_aabb[0] - c_aabb[1], 0),  # 右
            (abs(c_aabb[2] - f_aabb[3]), 0, c_aabb[2] - f_aabb[3]),  # 下
            (abs(f_aabb[2] - c_aabb[3]), 0, f_aabb[2] - c_aabb[3])   # 上
        ]
        seps.sort(key=lambda x: x[0])
        return np.array([seps[0][1], 0, seps[0][2]])

    elif collision['type'] == 'furniture_furniture':
        f1_pos = collision['furniture1']['pos']
        f2_pos = collision['furniture2']['pos']
        direction = f1_pos - f2_pos
        direction_2d = np.array([direction[0], direction[2]])

        if np.linalg.norm(direction_2d) < 0.01:
            direction_2d = np.array([1.0, 0.0])
        else:
            direction_2d = direction_2d / np.linalg.norm(direction_2d)

        f1_aabb = collision['furniture1']['aabb']
        f2_aabb = collision['furniture2']['aabb']
        overlap_x = min(f1_aabb[1], f2_aabb[1]) - max(f1_aabb[0], f2_aabb[0])
        overlap_z = min(f1_aabb[3], f2_aabb[3]) - max(f1_aabb[2], f2_aabb[2])

        sep_dist = overlap_x if abs(direction_2d[0]) > abs(direction_2d[1]) else overlap_z
        move_vec = direction_2d * sep_dist
        return np.array([move_vec[0], 0, move_vec[1]])

    elif collision['type'] == 'out_of_bounds':
        furn_aabb = collision['furniture']['aabb']
        min_x, max_x, min_z, max_z = bounds
        dx = dz = 0.0

        if furn_aabb[0] < min_x: dx = min_x - furn_aabb[0]
        elif furn_aabb[1] > max_x: dx = max_x - furn_aabb[1]
        if furn_aabb[2] < min_z: dz = min_z - furn_aabb[2]
        elif furn_aabb[3] > max_z: dz = max_z - furn_aabb[3]

        return np.array([dx, 0, dz])

    return np.array([0, 0, 0])

def do_reduce_collisions(data, max_iterations=5, max_move_per_step=0.2, verbose=True):
    """减少collision - 改进版：记录最佳状态，三级优先级"""
    if verbose:
        print(f"\n>> Starting collision reduction (max_iter={max_iterations}, max_move={max_move_per_step}m)")

    translations = data['translations'].copy()
    bounds = calculate_bounds(data)

    # 记录最佳状态: (num_furn_arch, num_furn_furn, num_furn_clearance, total_area)
    best_translations = translations.copy()
    best_score = (999, 999, 999, 999999.0)

    for iteration in range(max_iterations):
        current_data = data.copy()
        current_data['translations'] = translations
        collisions = detect_collisions(current_data, bounds)

        if not collisions:
            if verbose:
                print(f"   Iteration {iteration+1}: No collisions!")
            best_translations = translations.copy()
            best_score = (0, 0, 0, 0.0)
            break

        # 分类collision并计算score
        furn_arch_colls = [c for c in collisions if c['type'] in ['furniture_wall', 'out_of_bounds']]
        furn_furn_colls = [c for c in collisions if c['type'] == 'furniture_furniture']
        furn_clear_colls = [c for c in collisions if c['type'] == 'furniture_clearance']

        total_area = sum(c['overlap_area'] for c in collisions)
        current_score = (len(furn_arch_colls), len(furn_furn_colls), len(furn_clear_colls), total_area)

        # 更新最佳状态
        if current_score < best_score:
            best_score = current_score
            best_translations = translations.copy()

        # 优先级排序：墙 > 家具 > clearance
        priority_collisions = []

        # 优先级1: 家具vs墙 + out of bounds (最重要)
        for c in furn_arch_colls:
            priority_collisions.append((1, c['overlap_area'], c))

        # 优先级2: 家具vs家具 (次要)
        for c in furn_furn_colls:
            priority_collisions.append((2, c['overlap_area'], c))

        # 优先级3: 家具vs门clearance (最次要，但仍需处理)
        for c in furn_clear_colls:
            priority_collisions.append((3, c['overlap_area'], c))

        # 按优先级和面积排序
        priority_collisions.sort(key=lambda x: (x[0], -x[1]))

        total_moved = 0.0
        moves_count = 0

        for priority, area, coll in priority_collisions:
            sep_vec = calculate_separation(coll, bounds)
            move_mag = np.linalg.norm(sep_vec)

            if move_mag > max_move_per_step:
                sep_vec = sep_vec * (max_move_per_step / move_mag)
                move_mag = max_move_per_step

            if move_mag < 0.001:
                continue

            if coll['type'] in ['furniture_wall', 'out_of_bounds', 'furniture_clearance']:
                # 只移动家具
                furn_idx = coll['furniture']['index']

                # 检查移动后是否会导致更严重的OOB
                old_pos = translations[furn_idx]
                new_pos = old_pos + sep_vec

                # 计算当前和移动后的OOB程度
                furn_size = current_data['sizes'][furn_idx]
                furn_ang = get_angle(current_data['angles'][furn_idx])

                old_aabb = calculate_aabb_2d(old_pos, furn_size, furn_ang)
                new_aabb = calculate_aabb_2d(new_pos, furn_size, furn_ang)

                old_inside_ratio = calculate_inside_ratio(old_aabb, bounds)
                new_inside_ratio = calculate_inside_ratio(new_aabb, bounds)

                # 如果移动会让家具变得更加OOB（inside ratio减少超过5%），跳过这个移动
                if new_inside_ratio < old_inside_ratio - 0.05:
                    # print(f"    Skipping move: would push {coll['furniture']['label']} more OOB ({old_inside_ratio:.2f} -> {new_inside_ratio:.2f})")
                    continue

                translations[furn_idx] += sep_vec
                total_moved += move_mag
                moves_count += 1
            elif coll['type'] == 'furniture_furniture':
                f1 = coll['furniture1']
                f2 = coll['furniture2']
                size1 = np.prod(f1['size'][[0, 2]])
                size2 = np.prod(f2['size'][[0, 2]])

                if size1 < size2 * 0.8:
                    translations[f1['index']] += sep_vec
                    moves_count += 1
                elif size2 < size1 * 0.8:
                    translations[f2['index']] -= sep_vec
                    moves_count += 1
                else:
                    translations[f1['index']] += sep_vec * 0.5
                    translations[f2['index']] -= sep_vec * 0.5
                    moves_count += 2

                total_moved += move_mag

        if verbose:
            print(f"   Iteration {iteration+1}: {len(collisions)} collisions " +
                  f"(FvArch={len(furn_arch_colls)}, FvF={len(furn_furn_colls)}, FvClear={len(furn_clear_colls)}), " +
                  f"moved {total_moved:.3f}m")

        if total_moved < 0.01:
            if verbose:
                print(f"   Converged (movement < 0.01m)")
            break

    # 使用最佳状态
    final_data = data.copy()
    final_data['translations'] = best_translations
    final_collisions = detect_collisions(final_data, bounds)

    furn_arch_final = sum(1 for c in final_collisions if c['type'] in ['furniture_wall', 'out_of_bounds'])
    furn_furn_final = sum(1 for c in final_collisions if c['type'] == 'furniture_furniture')
    furn_clear_final = sum(1 for c in final_collisions if c['type'] == 'furniture_clearance')

    if verbose:
        print(f">> Collision reduction done: {len(final_collisions)} remaining " +
              f"(FvArch={furn_arch_final}, FvF={furn_furn_final}, FvClear={furn_clear_final})")
        print(f"   Best score: FvArch={best_score[0]}, FvF={best_score[1]}, FvClear={best_score[2]}, Area={best_score[3]:.4f}")

    modified = not np.allclose(data['translations'], best_translations, atol=1e-5)
    return final_data, modified

# ================= Squeeze Items Functions =================

def is_furniture_too_small(furniture_label, current_size):
    """检查家具是否已经小于最小尺寸"""
    if furniture_label not in MIN_FURNITURE_SIZES:
        return False

    min_size = MIN_FURNITURE_SIZES[furniture_label]
    # 排序后比较（小，大）
    current_sorted = sorted([current_size[0], current_size[2]])
    min_sorted = sorted(min_size)

    # 如果当前任一维度小于最小值，认为太小
    return current_sorted[0] < min_sorted[0] or current_sorted[1] < min_sorted[1]

def check_wall_attachment(furniture_aabb, wall_aabbs, threshold=0.05):
    """
    检查家具哪些边贴墙
    返回: {'left': bool, 'right': bool, 'top': bool, 'bottom': bool}
    """
    furn_min_x, furn_max_x, furn_min_z, furn_max_z = furniture_aabb

    attachment = {'left': False, 'right': False, 'top': False, 'bottom': False}

    for wall_aabb in wall_aabbs:
        wall_min_x, wall_max_x, wall_min_z, wall_max_z = wall_aabb

        # 检查左边（家具左边缘接近墙右边缘）
        if abs(furn_min_x - wall_max_x) < threshold:
            attachment['left'] = True

        # 检查右边
        if abs(furn_max_x - wall_min_x) < threshold:
            attachment['right'] = True

        # 检查上边（注意Z轴方向）
        if abs(furn_min_z - wall_max_z) < threshold:
            attachment['top'] = True

        # 检查下边
        if abs(furn_max_z - wall_min_z) < threshold:
            attachment['bottom'] = True

    return attachment

def calculate_squeeze_for_collision(collision, furniture_size, attachment):
    """
    计算collision需要的squeeze方向和量
    返回: (axis, amount, collision_side)
        axis: 0=X, 2=Z
        amount: 需要减小的量（正数），已包含额外gap
        collision_side: 'left'/'right'/'top'/'bottom'
    """
    coll_type = collision['type']

    # 获取该collision类型的额外gap
    extra_gap = SQUEEZE_GAPS.get(coll_type, 0.0)

    if coll_type == 'furniture_wall':
        # 家具vs墙：计算intersection方向
        furn_aabb = collision['furniture']['aabb']
        wall_aabb = collision['wall']['aabb']

        furn_min_x, furn_max_x, furn_min_z, furn_max_z = furn_aabb
        wall_min_x, wall_max_x, wall_min_z, wall_max_z = wall_aabb

        # 计算各方向的penetration
        penetrations = []

        # X方向
        pen_left = wall_max_x - furn_min_x  # 家具左边穿入墙
        pen_right = furn_max_x - wall_min_x  # 家具右边穿入墙

        if pen_left > 0 and furn_max_x > wall_min_x:  # 确实有重叠
            penetrations.append((pen_left, 0, 'left'))
        if pen_right > 0 and furn_min_x < wall_max_x:
            penetrations.append((pen_right, 0, 'right'))

        # Z方向
        pen_top = wall_max_z - furn_min_z
        pen_bottom = furn_max_z - wall_min_z

        if pen_top > 0 and furn_max_z > wall_min_z:
            penetrations.append((pen_top, 2, 'top'))
        if pen_bottom > 0 and furn_min_z < wall_max_z:
            penetrations.append((pen_bottom, 2, 'bottom'))

        # 选择penetration最小的方向
        if penetrations:
            penetrations.sort(key=lambda x: x[0])
            pen_amount, axis, side = penetrations[0]

            # 检查这一边是否贴墙
            if attachment[side]:
                # 如果collision的那边贴墙，不能从这边squeeze
                # 尝试从对面squeeze
                opposite = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
                opp_side = opposite[side]
                if not attachment[opp_side]:
                    # 从对面squeeze同样的量
                    return (axis, pen_amount, opp_side)
                else:
                    # 两边都贴墙，无法squeeze
                    return None

            # 添加额外gap
            return (axis, pen_amount + extra_gap, side)

    elif coll_type == 'furniture_clearance':
        # 家具vs门clearance：类似墙处理
        furn_aabb = collision['furniture']['aabb']
        clear_aabb = collision['door_clearance']['aabb']

        furn_min_x, furn_max_x, furn_min_z, furn_max_z = furn_aabb
        clear_min_x, clear_max_x, clear_min_z, clear_max_z = clear_aabb

        penetrations = []

        pen_left = clear_max_x - furn_min_x
        pen_right = furn_max_x - clear_min_x
        if pen_left > 0 and furn_max_x > clear_min_x:
            penetrations.append((pen_left, 0, 'left'))
        if pen_right > 0 and furn_min_x < clear_max_x:
            penetrations.append((pen_right, 0, 'right'))

        pen_top = clear_max_z - furn_min_z
        pen_bottom = furn_max_z - clear_min_z
        if pen_top > 0 and furn_max_z > clear_min_z:
            penetrations.append((pen_top, 2, 'top'))
        if pen_bottom > 0 and furn_min_z < clear_max_z:
            penetrations.append((pen_bottom, 2, 'bottom'))

        if penetrations:
            penetrations.sort(key=lambda x: x[0])
            pen_amount, axis, side = penetrations[0]

            if not attachment[side]:
                # 添加额外gap
                return (axis, pen_amount + extra_gap, side)

    elif coll_type == 'furniture_furniture':
        # 家具vs家具：计算overlap方向
        furn1_aabb = collision['furniture1']['aabb']
        furn2_aabb = collision['furniture2']['aabb']

        furn1_min_x, furn1_max_x, furn1_min_z, furn1_max_z = furn1_aabb
        furn2_min_x, furn2_max_x, furn2_min_z, furn2_max_z = furn2_aabb

        # 计算overlap
        overlap_x = min(furn1_max_x, furn2_max_x) - max(furn1_min_x, furn2_min_x)
        overlap_z = min(furn1_max_z, furn2_max_z) - max(furn1_min_z, furn2_min_z)

        # 选择overlap较小的方向
        if overlap_x < overlap_z:
            # X方向overlap更小，从X方向squeeze
            # 判断应该从哪边squeeze（左还是右）
            furn1_center_x = (furn1_min_x + furn1_max_x) / 2
            furn2_center_x = (furn2_min_x + furn2_max_x) / 2

            if furn1_center_x < furn2_center_x:
                # furn1在左边，从右边squeeze
                side = 'right'
            else:
                # furn1在右边，从左边squeeze
                side = 'left'

            if not attachment[side]:
                return (0, overlap_x, side)
        else:
            # Z方向overlap更小
            furn1_center_z = (furn1_min_z + furn1_max_z) / 2
            furn2_center_z = (furn2_min_z + furn2_max_z) / 2

            if furn1_center_z < furn2_center_z:
                side = 'bottom'
            else:
                side = 'top'

            if not attachment[side]:
                # 添加额外gap
                return (2, overlap_z + extra_gap, side)

    return None

def apply_squeeze(furniture_idx, axis, amount, side, data):
    """
    应用squeeze：减小家具尺寸，调整中心位置
    axis: 0=X, 2=Z
    amount: 减小的量
    side: 'left'/'right'/'top'/'bottom' - 从哪边减小
    """
    sizes = data['sizes']
    translations = data['translations']

    # 减小尺寸
    old_size = sizes[furniture_idx][axis]
    new_size = old_size - amount
    sizes[furniture_idx][axis] = new_size

    # 调整中心位置，保持非squeeze的那边不变
    # 例如：从右边squeeze，保持左边不变
    if side == 'right' or side == 'bottom':
        # 从右/下边减小，中心向左/上移动
        translations[furniture_idx][axis if axis == 0 else 2] -= amount / 2.0
    elif side == 'left' or side == 'top':
        # 从左/上边减小，中心向右/下移动
        translations[furniture_idx][axis if axis == 0 else 2] += amount / 2.0

def do_squeeze_items(data, max_iterations=5, verbose=True):
    """
    通过减小家具尺寸来减少collision
    只处理vanity, shower, tub（不处理toilet）
    """
    if verbose:
        print(f"\n>> Starting furniture squeeze (max_iter={max_iterations})")

    sizes = data['sizes'].copy()
    translations = data['translations'].copy()
    class_labels = data['class_labels']
    angles = data['angles']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    bounds = calculate_bounds(data)

    # 收集所有墙的AABB（用于检查贴墙）
    wall_aabbs = []
    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue
        if CLASS_LABELS[cls_indices[i]] == 'wall':
            pos = translations[i]
            size = sizes[i]
            ang = get_angle(angles[i])
            wall_aabbs.append(calculate_aabb_2d(pos, size, ang))

    # 记录最佳状态
    best_sizes = sizes.copy()
    best_translations = translations.copy()
    best_score = (999, 999, 999, 999999.0)

    for iteration in range(max_iterations):
        # 更新当前data
        current_data = data.copy()
        current_data['sizes'] = sizes
        current_data['translations'] = translations

        # 检测collision
        collisions = detect_collisions(current_data, bounds)

        if not collisions:
            if verbose:
                print(f"   Iteration {iteration+1}: No collisions!")
            best_sizes = sizes.copy()
            best_translations = translations.copy()
            best_score = (0, 0, 0, 0.0)
            break

        # 分类collision
        furn_arch_colls = [c for c in collisions if c['type'] in ['furniture_wall', 'out_of_bounds']]
        furn_furn_colls = [c for c in collisions if c['type'] == 'furniture_furniture']
        furn_clear_colls = [c for c in collisions if c['type'] == 'furniture_clearance']

        total_area = sum(c['overlap_area'] for c in collisions)
        current_score = (len(furn_arch_colls), len(furn_furn_colls), len(furn_clear_colls), total_area)

        # 更新最佳状态
        if current_score < best_score:
            best_score = current_score
            best_sizes = sizes.copy()
            best_translations = translations.copy()

        # 按优先级排序collision
        priority_collisions = []
        for c in furn_arch_colls:
            priority_collisions.append((1, c['overlap_area'], c))
        for c in furn_clear_colls:
            priority_collisions.append((2, c['overlap_area'], c))
        for c in furn_furn_colls:
            priority_collisions.append((3, c['overlap_area'], c))

        priority_collisions.sort(key=lambda x: (x[0], -x[1]))

        squeezed_count = 0
        total_squeezed = 0.0

        for priority, area, coll in priority_collisions:
            furniture = None
            furn_idx = None
            furn_label = None

            # 获取要处理的家具
            if coll['type'] in ['furniture_wall', 'furniture_clearance']:
                furniture = coll['furniture']
                furn_idx = furniture['index']
                furn_label = furniture['label']
            elif coll['type'] == 'furniture_furniture':
                # 对于家具vs家具，选择可squeeze且较大的家具
                f1 = coll['furniture1']
                f2 = coll['furniture2']

                # 检查哪个可以squeeze
                f1_squeezable = f1['label'] in SQUEEZABLE_FURNITURE and not is_furniture_too_small(f1['label'], sizes[f1['index']])
                f2_squeezable = f2['label'] in SQUEEZABLE_FURNITURE and not is_furniture_too_small(f2['label'], sizes[f2['index']])

                if not f1_squeezable and not f2_squeezable:
                    continue  # 两个都不能squeeze
                elif f1_squeezable and not f2_squeezable:
                    furniture = f1
                elif f2_squeezable and not f1_squeezable:
                    furniture = f2
                else:
                    # 两个都能squeeze，选择较大的
                    size1 = np.prod(sizes[f1['index']][[0, 2]])
                    size2 = np.prod(sizes[f2['index']][[0, 2]])
                    furniture = f1 if size1 > size2 else f2

                furn_idx = furniture['index']
                furn_label = furniture['label']
            else:
                continue  # 其他类型暂时跳过

            # 只处理可squeeze的家具
            if furn_label not in SQUEEZABLE_FURNITURE:
                continue

            # 检查是否已经太小
            if is_furniture_too_small(furn_label, sizes[furn_idx]):
                continue

            # 检查贴墙情况
            furn_pos = translations[furn_idx]
            furn_size = sizes[furn_idx]
            furn_ang = get_angle(angles[furn_idx])
            furn_aabb = calculate_aabb_2d(furn_pos, furn_size, furn_ang)
            attachment = check_wall_attachment(furn_aabb, wall_aabbs)

            # 计算squeeze
            squeeze_info = calculate_squeeze_for_collision(coll, furn_size, attachment)
            if squeeze_info is None:
                continue  # 无法squeeze（比如四面贴墙）

            axis, amount, side = squeeze_info

            # 限制不超过最小尺寸
            min_size = MIN_FURNITURE_SIZES[furn_label]
            current_size = furn_size[axis]

            # 找到对应轴的最小尺寸
            # 需要根据家具的形状判断哪个是长哪个是宽
            if axis == 0:  # X轴
                other_size = furn_size[2]
            else:  # Z轴
                other_size = furn_size[0]

            # 排序当前尺寸
            current_sorted = sorted([current_size, other_size])
            min_sorted = sorted(min_size)

            # 判断当前轴是长边还是短边
            if current_size == current_sorted[1]:  # 长边
                min_allowed = min_sorted[1]
            else:  # 短边
                min_allowed = min_sorted[0]

            # 限制squeeze量
            max_squeeze = current_size - min_allowed
            if max_squeeze <= 0:
                continue  # 已经达到最小值

            amount = min(amount, max_squeeze)

            if amount < 0.001:
                continue

            # 应用squeeze
            apply_squeeze(furn_idx, axis, amount, side, current_data)
            sizes = current_data['sizes']
            translations = current_data['translations']

            squeezed_count += 1
            total_squeezed += amount

        if verbose:
            print(f"   Iteration {iteration+1}: {len(collisions)} collisions " +
                  f"(FvArch={len(furn_arch_colls)}, FvF={len(furn_furn_colls)}, FvClear={len(furn_clear_colls)}), " +
                  f"squeezed {squeezed_count} items, total {total_squeezed:.3f}m")

        if squeezed_count == 0:
            if verbose:
                print(f"   No more items can be squeezed")
            break

    # 使用最佳状态
    final_data = data.copy()
    final_data['sizes'] = best_sizes
    final_data['translations'] = best_translations

    final_collisions = detect_collisions(final_data, bounds)
    furn_arch_final = sum(1 for c in final_collisions if c['type'] in ['furniture_wall', 'out_of_bounds'])
    furn_furn_final = sum(1 for c in final_collisions if c['type'] == 'furniture_furniture')
    furn_clear_final = sum(1 for c in final_collisions if c['type'] == 'furniture_clearance')

    if verbose:
        print(f">> Squeeze done: {len(final_collisions)} remaining " +
              f"(FvArch={furn_arch_final}, FvF={furn_furn_final}, FvClear={furn_clear_final})")
        print(f"   Best score: FvArch={best_score[0]}, FvF={best_score[1]}, FvClear={best_score[2]}, Area={best_score[3]:.4f}")

    modified = not (np.allclose(data['sizes'], best_sizes, atol=1e-5) and
                    np.allclose(data['translations'], best_translations, atol=1e-5))

    return final_data, modified

# ================= Remove Out of Bounds Items =================

def calculate_inside_ratio(furniture_aabb, bounds):
    """
    计算家具在bounds内的比例
    返回: 0.0 到 1.0 之间的值
    """
    furn_min_x, furn_max_x, furn_min_z, furn_max_z = furniture_aabb
    bounds_min_x, bounds_max_x, bounds_min_z, bounds_max_z = bounds

    # 计算overlap区域
    overlap_min_x = max(furn_min_x, bounds_min_x)
    overlap_max_x = min(furn_max_x, bounds_max_x)
    overlap_min_z = max(furn_min_z, bounds_min_z)
    overlap_max_z = min(furn_max_z, bounds_max_z)

    # 如果没有overlap
    if overlap_max_x <= overlap_min_x or overlap_max_z <= overlap_min_z:
        return 0.0

    # 计算面积
    overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_z - overlap_min_z)
    furniture_area = (furn_max_x - furn_min_x) * (furn_max_z - furn_min_z)

    if furniture_area < 0.0001:  # 避免除以0
        return 0.0

    return overlap_area / furniture_area

def do_remove_oob_items(data, min_inside_ratio=MIN_INSIDE_RATIO, verbose=True):
    """
    删除过度超出bounds的家具
    如果家具在bounds内的比例小于min_inside_ratio，则删除
    """
    if verbose:
        print(f"\n>> Checking for out-of-bounds items (min_inside_ratio={min_inside_ratio*100:.0f}%)")

    translations = data['translations']
    sizes = data['sizes']
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    # 计算bounds
    bounds = calculate_bounds(data)

    indices_to_keep = []
    indices_to_remove = []
    removed_items = []

    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            indices_to_keep.append(i)
            continue

        label_name = CLASS_LABELS[cls_indices[i]]

        # 只检查家具
        if label_name not in FURNITURE_CLASSES:
            indices_to_keep.append(i)
            continue

        # 计算家具AABB
        pos = translations[i]
        size = sizes[i]
        ang = get_angle(angles[i])
        furniture_aabb = calculate_aabb_2d(pos, size, ang)

        # 计算在bounds内的比例
        inside_ratio = calculate_inside_ratio(furniture_aabb, bounds)

        if inside_ratio < min_inside_ratio:
            # 超出太多，删除
            indices_to_remove.append(i)
            removed_items.append({
                'index': i,
                'label': label_name,
                'inside_ratio': inside_ratio,
                'pos': pos
            })
            if verbose:
                print(f"   Removing OOB: [{i:02d}] {label_name} (only {inside_ratio*100:.1f}% inside bounds)")
        else:
            indices_to_keep.append(i)

    if removed_items:
        if verbose:
            print(f">> Removed {len(removed_items)} out-of-bounds item(s)")

        # 创建新数据，只保留valid indices
        new_data = {}
        for key in data.keys():
            if key in ['translations', 'sizes', 'angles', 'class_labels']:
                # 这些是per-item的数据，需要删除对应rows
                new_data[key] = data[key][indices_to_keep]
            else:
                # 其他数据保持不变
                new_data[key] = data[key]

        return new_data, True
    else:
        if verbose:
            print(f">> No out-of-bounds items found")
        return data, False

# ================= Remove Collision Items =================

def calculate_collision_overlap_ratio(collision):
    """
    计算collision的overlap比例（相对于家具面积）
    """
    overlap_area = collision['overlap_area']

    if collision['type'] in ['furniture_wall', 'furniture_clearance', 'out_of_bounds']:
        furniture_aabb = collision['furniture']['aabb']
        furniture_area = (furniture_aabb[1] - furniture_aabb[0]) * (furniture_aabb[3] - furniture_aabb[2])
        if furniture_area < 0.0001:
            return 0.0
        return overlap_area / furniture_area

    elif collision['type'] == 'furniture_furniture':
        # 对于家具vs家具，计算相对于较小家具的比例
        f1_aabb = collision['furniture1']['aabb']
        f2_aabb = collision['furniture2']['aabb']
        f1_area = (f1_aabb[1] - f1_aabb[0]) * (f1_aabb[3] - f1_aabb[2])
        f2_area = (f2_aabb[1] - f2_aabb[0]) * (f2_aabb[3] - f2_aabb[2])
        smaller_area = min(f1_area, f2_area)
        if smaller_area < 0.0001:
            return 0.0
        return overlap_area / smaller_area

    return 0.0

def filter_severe_collisions(collisions, thresholds=SEVERE_COLLISION_THRESHOLDS):
    """
    筛选出严重的collision
    """
    severe = []
    for coll in collisions:
        coll_type = coll['type']
        if coll_type not in thresholds:
            continue

        overlap_ratio = calculate_collision_overlap_ratio(coll)
        threshold = thresholds[coll_type]

        if overlap_ratio >= threshold:
            severe.append({
                'collision': coll,
                'overlap_ratio': overlap_ratio
            })

    return severe

def count_severe_collisions_per_furniture(severe_collisions, data):
    """
    统计每个家具参与了多少严重collision
    返回: {furniture_index: count}
    """
    class_labels = data['class_labels']
    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    collision_counts = {}

    for severe in severe_collisions:
        coll = severe['collision']

        if coll['type'] in ['furniture_wall', 'furniture_clearance', 'out_of_bounds']:
            furn_idx = coll['furniture']['index']
            collision_counts[furn_idx] = collision_counts.get(furn_idx, 0) + 1

        elif coll['type'] == 'furniture_furniture':
            f1_idx = coll['furniture1']['index']
            f2_idx = coll['furniture2']['index']
            collision_counts[f1_idx] = collision_counts.get(f1_idx, 0) + 1
            collision_counts[f2_idx] = collision_counts.get(f2_idx, 0) + 1

    return collision_counts

def select_furniture_to_remove(collision_counts, data):
    """
    选择要删除的家具（考虑优先级和collision数量）
    优先删除：shower/tub > vanity > toilet
    在同优先级中，删除collision数量最多的
    """
    if not collision_counts:
        return None

    class_labels = data['class_labels']
    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    # 按优先级分组
    by_priority = {1: [], 2: [], 3: []}

    for furn_idx, count in collision_counts.items():
        if count == 0:
            continue

        label_name = CLASS_LABELS[cls_indices[furn_idx]]
        if label_name not in FURNITURE_REMOVAL_PRIORITY:
            continue

        priority = FURNITURE_REMOVAL_PRIORITY[label_name]
        by_priority[priority].append({
            'index': furn_idx,
            'label': label_name,
            'count': count
        })

    # 从优先级1开始找（最先删除shower/tub）
    for priority in [1, 2, 3]:
        if by_priority[priority]:
            # 在这个优先级中，选collision数量最多的
            candidates = by_priority[priority]
            best = max(candidates, key=lambda x: x['count'])
            return best

    return None

def do_remove_collision_items(data, thresholds=SEVERE_COLLISION_THRESHOLDS, verbose=True):
    """
    迭代删除有严重collision的家具
    每次删除collision数量最多的家具（考虑删除优先级）
    """
    if verbose:
        print(f"\n>> Removing furniture with severe collisions")

    translations = data['translations'].copy()
    sizes = data['sizes'].copy()
    angles = data['angles'].copy()
    class_labels = data['class_labels'].copy()

    bounds = calculate_bounds(data)

    # 首先检测并显示所有collision
    initial_data = data.copy()
    all_collisions = detect_collisions(initial_data, bounds)

    if all_collisions:
        if verbose:
            print(f"\n   Found {len(all_collisions)} total collision(s):")
            for i, coll in enumerate(all_collisions):
                overlap_ratio = calculate_collision_overlap_ratio(coll)
                coll_type = coll['type']

                if coll_type == 'furniture_wall':
                    furn = coll['furniture']
                    threshold = thresholds.get(coll_type, 0.0)
                    is_severe = overlap_ratio >= threshold
                    severity_mark = "[SEVERE]" if is_severe else ""
                    print(f"   [{i+1}] {furn['label']} vs wall: {severity_mark}")
                    print(f"       Overlap: {coll['overlap_area']:.4f}m², Ratio: {overlap_ratio*100:.1f}% (threshold: {threshold*100:.0f}%)")

                elif coll_type == 'furniture_clearance':
                    furn = coll['furniture']
                    threshold = thresholds.get(coll_type, 0.0)
                    is_severe = overlap_ratio >= threshold
                    severity_mark = "[SEVERE]" if is_severe else ""
                    print(f"   [{i+1}] {furn['label']} vs clearance: {severity_mark}")
                    print(f"       Overlap: {coll['overlap_area']:.4f}m², Ratio: {overlap_ratio*100:.1f}% (threshold: {threshold*100:.0f}%)")

                elif coll_type == 'furniture_furniture':
                    f1 = coll['furniture1']
                    f2 = coll['furniture2']
                    threshold = thresholds.get(coll_type, 0.0)
                    is_severe = overlap_ratio >= threshold
                    severity_mark = "[SEVERE]" if is_severe else ""
                    print(f"   [{i+1}] {f1['label']} vs {f2['label']}: {severity_mark}")
                    print(f"       Overlap: {coll['overlap_area']:.4f}m², Ratio: {overlap_ratio*100:.1f}% (threshold: {threshold*100:.0f}%)")

                elif coll_type == 'out_of_bounds':
                    furn = coll['furniture']
                    print(f"   [{i+1}] {furn['label']} OUT OF BOUNDS:")
                    print(f"       OOB amount: {coll['overlap_area']:.4f}m")
            print()
    else:
        if verbose:
            print(f"   No collisions found")

    removed_items = []
    iteration = 0

    while True:
        iteration += 1

        # 更新当前data
        current_data = data.copy()
        current_data['translations'] = translations
        current_data['sizes'] = sizes
        current_data['class_labels'] = class_labels
        current_data['angles'] = angles

        # 检测所有collision
        collisions = detect_collisions(current_data, bounds)

        # 筛选严重collision
        severe_collisions = filter_severe_collisions(collisions, thresholds)

        if not severe_collisions:
            if verbose:
                print(f"   No severe collisions remaining after {iteration-1} iterations")
            break

        # 统计每个家具的严重collision数量
        collision_counts = count_severe_collisions_per_furniture(severe_collisions, current_data)

        if not collision_counts:
            break

        # 选择要删除的家具
        furniture_to_remove = select_furniture_to_remove(collision_counts, current_data)

        if furniture_to_remove is None:
            if verbose:
                print(f"   No furniture can be removed")
            break

        furn_idx = furniture_to_remove['index']
        furn_label = furniture_to_remove['label']
        furn_count = furniture_to_remove['count']

        if verbose:
            print(f"   Iteration {iteration}: Removing {furn_label} (index {furn_idx}, had {furn_count} severe collision(s))")

        # 创建保留的indices列表
        indices_to_keep = [i for i in range(len(class_labels)) if i != furn_idx]

        # 删除该家具
        translations = translations[indices_to_keep]
        sizes = sizes[indices_to_keep]
        angles = angles[indices_to_keep]
        class_labels = class_labels[indices_to_keep]

        removed_items.append({
            'original_index': furn_idx,
            'label': furn_label,
            'count': furn_count
        })

    if removed_items:
        if verbose:
            print(f">> Removed {len(removed_items)} furniture item(s) due to severe collisions")

        # 创建新数据
        new_data = {}
        for key in data.keys():
            if key == 'translations':
                new_data[key] = translations
            elif key == 'sizes':
                new_data[key] = sizes
            elif key == 'angles':
                new_data[key] = angles
            elif key == 'class_labels':
                new_data[key] = class_labels
            else:
                new_data[key] = data[key]

        return new_data, True
    else:
        if verbose:
            print(f">> No severe collisions found")
        return data, False

# ================= 2. 几何工具函数 (基于你的向量逻辑) =================

def is_valid_item(label_row):
    """检查 class_labels 是否为有效的 slot"""
    if label_row.ndim == 0: return True
    return np.max(label_row) > 0

def get_angle(angle_data):
    """从不同格式中提取弧度"""
    if isinstance(angle_data, np.ndarray):
        if angle_data.size == 2: # [cos, sin]
            return np.arctan2(angle_data[1], angle_data[0])
        return angle_data.item()
    return angle_data

def debug_print(title, data, verbose=True):
    if not verbose:
        return
    translations = data['translations']
    sizes = data['sizes']
    class_labels = data['class_labels']
    cls_indices = np.argmax(class_labels, axis=-1) if class_labels.ndim > 1 else class_labels.flatten()

    print(f"\n>>>> DEBUG: {title} <<<<")
    count = 0
    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]): continue
        label_name = CLASS_LABELS[cls_indices[i]]
        if label_name == "empty": continue
        pos = translations[i]
        sz = sizes[i]
        print(f"[{count:02d}] {label_name:8s} | Pos: ({pos[0]:6.2f}, {pos[2]:6.2f}) | Size: ({sz[0]:.2f}, {sz[2]:.2f})")
        count += 1
    print("-" * 70)

# ================= 3. 核心逻辑处理 (基于你的 snippet) =================

def do_remove_invalid_items(data, verbose=True):
    """
    移除invalid items：
    1. 家具后面不应该出现architecture items
    2. 特殊情况：如果总数=50，删除最后10个architecture items
    """
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    total_items = len(cls_indices)

    # 找到第一个家具出现的位置
    first_furniture_idx = -1
    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue

        label_name = CLASS_LABELS[cls_indices[i]]
        if label_name in FURNITURE_CLASSES:
            first_furniture_idx = i
            break

    # 收集要删除的indices
    indices_to_keep = []
    indices_to_remove = []

    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            indices_to_keep.append(i)
            continue

        label_name = CLASS_LABELS[cls_indices[i]]
        should_remove = False
        remove_reason = ""

        # 规则1: 家具之后的非家具item（architecture）
        if first_furniture_idx >= 0 and i > first_furniture_idx:
            if label_name not in FURNITURE_CLASSES and label_name != "empty":
                should_remove = True
                remove_reason = "architecture after furniture"

        # 规则2: 总数=50时，删除最后10个architecture items
        if total_items == 50 and i >= 40:  # 最后10个位置 (index 40-49)
            if label_name in ARCHITECTURE_CLASSES:
                should_remove = True
                remove_reason = "architecture in last 10 positions (total=50)"

        if should_remove:
            indices_to_remove.append(i)
            if verbose:
                print(f"   Removing: [{i:02d}] {label_name} ({remove_reason})")
        else:
            indices_to_keep.append(i)

    if indices_to_remove:
        if verbose:
            print(f">> Removed {len(indices_to_remove)} invalid item(s)")

        # 创建新数据，只保留valid indices
        new_data = {}
        for key in data.keys():
            if key in ['translations', 'sizes', 'angles', 'class_labels']:
                # 这些是per-item的数据，需要删除对应rows
                new_data[key] = data[key][indices_to_keep]
            else:
                # 其他数据保持不变
                new_data[key] = data[key]

        return new_data, True
    else:
        if verbose:
            print(f">> No invalid items found")
        return data, False

# ================= 3. 核心逻辑处理 (基于你的 snippet) =================

def do_attach_items(data):
    # 提取数据
    translations = data['translations'].copy()
    sizes = data['sizes'].copy()
    angles_raw = data['angles'].copy()
    class_labels = data['class_labels']

    # 解析索引
    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    modified = False

    # 1. 预计算环境信息
    w_indices = [i for i, idx in enumerate(cls_indices) if is_valid_item(class_labels[i]) and CLASS_LABELS[idx] == "wall"]
    d_indices = [i for i, idx in enumerate(cls_indices) if is_valid_item(class_labels[i]) and CLASS_LABELS[idx] == "door"]
    f_indices = [i for i, idx in enumerate(cls_indices) if is_valid_item(class_labels[i]) and CLASS_LABELS[idx] in FURNITURE_CLASSES]

    if not w_indices: return data, False

    # --- A. 门 Ghost Box (AABB) 预计算 ---
    ghost_boxes = []
    clearance_depth = 0.8
    for di in d_indices:
        d_pos = translations[di]
        d_sz = sizes[di]
        d_ang = get_angle(angles_raw[di])

        # 旋转感知 AABB
        cos_a, sin_a = abs(math.cos(d_ang)), abs(math.sin(d_ang))
        d_width = max(d_sz[0], d_sz[2])
        d_thick = min(d_sz[0], d_sz[2])

        total_g_depth = d_thick + clearance_depth * 2.0
        g_ex = (total_g_depth / 2.0) * cos_a + (d_width / 2.0) * sin_a
        g_ez = (total_g_depth / 2.0) * sin_a + (d_width / 2.0) * cos_a

        ghost_boxes.append({
            'min': np.array([d_pos[0] - g_ex, d_pos[2] - g_ez]),
            'max': np.array([d_pos[0] + g_ex, d_pos[2] + g_ez]),
            'id': di
        })

    # --- B. 墙体法线预计算 ---
    walls_meta = []
    for wi in w_indices:
        w_sz = sizes[wi]
        w_ang = get_angle(angles_raw[wi])

        # 识别厚度轴 (厚度通常是 X 和 Z 里的最小值)
        thick_axis = np.argmin([w_sz[0], w_sz[2]])

        # 计算墙体法线 (World Space)
        if thick_axis == 0: # X 轴是厚度轴
            w_normal = np.array([math.cos(w_ang), 0, math.sin(w_ang)])
        else: # Z 轴是厚度轴
            w_normal = np.array([-math.sin(w_ang), 0, math.cos(w_ang)])

        walls_meta.append({
            'idx': wi,
            'pos': translations[wi],
            'normal': w_normal,
            'half_thick': min(w_sz[0], w_sz[2]) / 2.0
        })

    # --- C. 家具处理 ---
    for fi in range(len(cls_indices)):
        if not is_valid_item(class_labels[fi]): continue
        label_name = CLASS_LABELS[cls_indices[fi]]
        if label_name not in FURNITURE_CLASSES: continue

        f_pos = translations[fi]
        f_sz = sizes[fi]
        f_ang = get_angle(angles_raw[fi])

        # 1. 计算家具 AABB 投影半径
        f_cos, f_sin = abs(math.cos(f_ang)), abs(math.sin(f_ang))
        f_aabb_half_w = (f_cos * f_sz[0] + f_sin * f_sz[2]) / 2.0
        f_aabb_half_d = (f_sin * f_sz[0] + f_cos * f_sz[2]) / 2.0

        # 2. 评估所有墙的 Gap
        wall_cands = []
        for wm in walls_meta:
            rel_p = f_pos - wm['pos']
            dist_to_wall = np.dot(rel_p, wm['normal'])

            # 投影厚度
            f_half_thick = abs(wm['normal'][0]) * f_aabb_half_w + abs(wm['normal'][2]) * f_aabb_half_d
            gap = abs(dist_to_wall) - (f_half_thick + wm['half_thick'])

            # 关键修复1：检查墙和家具在平行于墙的方向上是否有AABB重叠
            # 只处理真正相邻的墙
            w_sz = sizes[wm['idx']]

            if abs(wm['normal'][0]) > 0.5:  # 墙沿Z方向
                # 检查Z方向的AABB重叠
                wall_z_min = wm['pos'][2] - max(w_sz[0], w_sz[2]) / 2.0
                wall_z_max = wm['pos'][2] + max(w_sz[0], w_sz[2]) / 2.0
                furn_z_min = f_pos[2] - max(f_sz[0], f_sz[2]) / 2.0
                furn_z_max = f_pos[2] + max(f_sz[0], f_sz[2]) / 2.0

                if furn_z_max < wall_z_min or furn_z_min > wall_z_max:
                    continue  # 没有重叠，跳过

            else:  # 墙沿X方向
                # 检查X方向的AABB重叠
                wall_x_min = wm['pos'][0] - max(w_sz[0], w_sz[2]) / 2.0
                wall_x_max = wm['pos'][0] + max(w_sz[0], w_sz[2]) / 2.0
                furn_x_min = f_pos[0] - max(f_sz[0], f_sz[2]) / 2.0
                furn_x_max = f_pos[0] + max(f_sz[0], f_sz[2]) / 2.0

                if furn_x_max < wall_x_min or furn_x_min > wall_x_max:
                    continue  # 没有重叠，跳过

            wall_cands.append({
                'meta': wm,
                'dist_to_wall': dist_to_wall,
                'gap': gap,
                'abs_gap': abs(gap)
            })

        # 关键修复2：按abs(gap)排序，选择最接近贴墙的墙
        wall_cands.sort(key=lambda x: abs(x['gap']))

        # 3. 选择最近的墙并尝试吸附
        best_pos = f_pos.copy()
        success = False

        if wall_cands:
            # 只尝试最优先的墙，避免跨墙
            cand = wall_cands[0]
            wm = cand['meta']
            dist_to_wall = cand['dist_to_wall']

            # 计算贴墙位置
            f_half_thick = abs(wm['normal'][0]) * f_aabb_half_w + abs(wm['normal'][2]) * f_aabb_half_d

            if dist_to_wall > 0:
                target_dist = wm['half_thick'] + f_half_thick
            else:
                target_dist = -(wm['half_thick'] + f_half_thick)

            # 计算位移
            displacement = (target_dist - dist_to_wall) * wm['normal']
            temp_pos = f_pos + displacement

            # --- 堵门校验 (Ghost Box AABB) ---
            is_blocking = False
            f_min_2d = temp_pos[[0, 2]] - (f_sz[[0, 2]] / 2.0)
            f_max_2d = temp_pos[[0, 2]] + (f_sz[[0, 2]] / 2.0)

            for gb in ghost_boxes:
                overlap_x = (f_min_2d[0] < gb['max'][0]) and (f_max_2d[0] > gb['min'][0])
                overlap_z = (f_min_2d[1] < gb['max'][1]) and (f_max_2d[1] > gb['min'][1])
                if overlap_x and overlap_z:
                    is_blocking = True
                    break

            if not is_blocking:
                best_pos = temp_pos
                success = True

        if success:
            if not np.allclose(translations[fi], best_pos, atol=1e-3):
                translations[fi] = best_pos
                modified = True

    new_data = data.copy()
    new_data['translations'] = translations
    new_data['angles'] = angles_raw
    return new_data, modified

def post_process_data(input_data, verbose=True):
    debug_print("INPUT DATA", input_data, verbose=verbose)

    # Step 0: 移除invalid items (家具后面的architecture items)
    output_data, modified_remove = do_remove_invalid_items(input_data, verbose=verbose)

    max_trials = 3
    for trial in range(max_trials):
        if verbose:
            print(f">> [{trial+1}/{max_trials}] trial.")

        modified = False
        # Step 1: 执行吸附逻辑
        output_data, modified_attach = do_attach_items(output_data)
        modified = modified or modified_attach

        # Step 2: 执行collision reduction
        output_data, modified_collision = do_reduce_collisions(output_data, max_iterations=5, max_move_per_step=0.3, verbose=verbose)
        modified = modified or modified_collision

        # Step 3: 执行furniture squeeze
        output_data, modified_squeeze = do_squeeze_items(output_data, max_iterations=5, verbose=verbose)
        modified = modified or modified_squeeze

        # Step 4: 移除out of bounds的家具
        output_data, modified_oob = do_remove_oob_items(output_data, min_inside_ratio=MIN_INSIDE_RATIO, verbose=verbose)
        modified = modified or modified_oob

        # Step 5: 移除有严重collision的家具
        output_data, modified_severe = do_remove_collision_items(output_data, thresholds=SEVERE_COLLISION_THRESHOLDS, verbose=verbose)
        modified = modified or modified_severe

        if not modified:
            if verbose:
                print(f">> Stopped earlier at [{trial+1}/{max_trials}].")
            break

    debug_print("OUTPUT DATA", output_data, verbose=verbose)
    return output_data

# ================= 4. 主流程 =================

def process_single_file(input_path, output_path=None, verbose=True):
    """Process a single .npz file"""
    if not os.path.exists(input_path):
        if verbose:
            print(f"Warning: File not found: {input_path}")
        return False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {input_path}")
        print(f"{'='*70}")
    
    with np.load(input_path, allow_pickle=True) as raw:
        input_data = {k: raw[k].copy() for k in raw.files}

    output_data = post_process_data(input_data, verbose=verbose)

    if output_path is None:
        output_path = input_path + ".postprocess.npz"
    
    np.savez(output_path, **output_data)
    if verbose:
        print(f"\nFinal result saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file or directory containing .npz files')
    parser.add_argument('--output', help='Output file (only used when input is a single file)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return

    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file mode (backward compatibility)
        process_single_file(args.input, args.output)
    elif os.path.isdir(args.input):
        # Directory mode: find all .npz files starting with 'boxes'
        print(f"Scanning directory: {args.input}")
        
        # Collect all files to process
        files_to_process = []
        for root, dirs, files in os.walk(args.input):
            npz_files = [f for f in files if f.startswith('boxes') and f.endswith('.npz')]
            for npz_file in npz_files:
                input_path = os.path.join(root, npz_file)
                output_path = input_path + ".postprocess.npz"
                
                # Skip if output already exists
                if not os.path.exists(output_path):
                    files_to_process.append((input_path, output_path))
        
        if not files_to_process:
            print("No files to process (all outputs already exist)")
            return
        
        print(f"Found {len(files_to_process)} file(s) to process")
        processed_count = 0
        
        # Process files with progress bar
        for input_path, output_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            if process_single_file(input_path, output_path, verbose=False):
                processed_count += 1
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Processed: {processed_count} files")
        print(f"  Skipped: {len(files_to_process) - processed_count} files")
        print(f"{'='*70}")
    else:
        print(f"Error: Input path is neither a file nor a directory: {args.input}")

if __name__ == "__main__":
    main()
