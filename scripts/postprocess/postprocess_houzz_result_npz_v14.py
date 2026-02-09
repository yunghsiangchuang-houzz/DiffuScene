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

    #FURNITURE_PADDING = 0.15  # Toilet四周各留15cm空间
    FURNITURE_PADDING = 0.05  # Toilet四周各留15cm空间
    #FURNITURE_PADDING = 0  # Toilet四周各留0cm空间

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

        # ========== 新增：给toilet的AABB加padding ==========
        #if label_name == "toilet":
        if label_name in FURNITURE_CLASSES:
            aabb = (
                aabb[0] - FURNITURE_PADDING,  # min_x
                aabb[1] + FURNITURE_PADDING,  # max_x
                aabb[2] - FURNITURE_PADDING,  # min_z
                aabb[3] + FURNITURE_PADDING,  # max_z
            )
        # ==================================================

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

    # 3. 家具 vs 门clearance
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
            #indices_to_keep.append(i)
            indices_to_remove.append(i)
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

# 改进后的do_attach_items函数
# 新增功能：
# 1. 考虑家具旋转角度的朝向检查
# 2. 检查墙接触比例（至少50%的边靠墙）

MIN_WALL_CONTACT_RATIO = 0.5  # 家具至少50%的边要靠墙

def check_orientation_rule(furniture_type, furniture_size, furniture_angle, wall_meta, angle_threshold=0.9):
    """
    检查家具朝向是否满足规则（考虑旋转角度）
    - toilet: 短边必须平行墙
    - vanity: 长边必须平行墙
    - shower/tub: 不限制
    """
    if furniture_type not in ['toilet', 'vanity']:
        return True

    # 1. 判断家具的长短边（在局部坐标系）
    furn_x_size = furniture_size[0]
    furn_z_size = furniture_size[2]

    if furn_x_size > furn_z_size:
        local_long_edge = np.array([1.0, 0.0, 0.0])
        local_short_edge = np.array([0.0, 0.0, 1.0])
    else:
        local_long_edge = np.array([0.0, 0.0, 1.0])
        local_short_edge = np.array([1.0, 0.0, 0.0])

    # 2. 旋转到世界坐标系
    cos_a = math.cos(furniture_angle)
    sin_a = math.sin(furniture_angle)

    def rotate_vector(v):
        x_world = cos_a * v[0] + sin_a * v[2]
        z_world = -sin_a * v[0] + cos_a * v[2]
        return np.array([x_world, 0.0, z_world])

    world_long_edge = rotate_vector(local_long_edge)
    world_short_edge = rotate_vector(local_short_edge)

    # 归一化
    world_long_edge = world_long_edge / (np.linalg.norm(world_long_edge) + 1e-8)
    world_short_edge = world_short_edge / (np.linalg.norm(world_short_edge) + 1e-8)

    # 3. 获取墙的延伸方向
    wall_normal = wall_meta['normal']
    wall_direction = np.array([-wall_normal[2], 0.0, wall_normal[0]])
    wall_direction_norm = np.linalg.norm(wall_direction)
    if wall_direction_norm > 1e-8:
        wall_direction = wall_direction / wall_direction_norm
    else:
        return True

    # 4. 计算夹角
    if furniture_type == 'toilet':
        dot_product = abs(np.dot(world_short_edge, wall_direction))
        is_parallel = dot_product > angle_threshold
        return is_parallel
    elif furniture_type == 'vanity':
        dot_product = abs(np.dot(world_long_edge, wall_direction))
        is_parallel = dot_product > angle_threshold
        return is_parallel

    return True


def check_wall_contact_ratio(furniture_pos, furniture_size, furniture_angle, wall_meta, min_ratio=0.5):
    """
    检查家具贴墙后的接触比例
    考虑家具旋转后的实际AABB投影

    返回: True=接触面积>=min_ratio, False=接触面积<min_ratio
    """
    # 1. 计算家具旋转后的AABB
    f_cos = abs(math.cos(furniture_angle))
    f_sin = abs(math.sin(furniture_angle))

    f_half_w = (f_cos * furniture_size[0] + f_sin * furniture_size[2]) / 2.0
    f_half_d = (f_sin * furniture_size[0] + f_cos * furniture_size[2]) / 2.0

    furn_x_min = furniture_pos[0] - f_half_w
    furn_x_max = furniture_pos[0] + f_half_w
    furn_z_min = furniture_pos[2] - f_half_d
    furn_z_max = furniture_pos[2] + f_half_d

    # 2. 获取墙的范围
    wall_pos = wall_meta['pos']
    wall_size = wall_meta['size']

    wall_x_min = wall_pos[0] - max(wall_size[0], wall_size[2]) / 2.0
    wall_x_max = wall_pos[0] + max(wall_size[0], wall_size[2]) / 2.0
    wall_z_min = wall_pos[2] - max(wall_size[0], wall_size[2]) / 2.0
    wall_z_max = wall_pos[2] + max(wall_size[0], wall_size[2]) / 2.0

    # 3. 判断墙的延伸方向并计算overlap
    wall_normal = wall_meta['normal']

    if abs(wall_normal[0]) > abs(wall_normal[2]):
        # 墙沿Z方向延伸
        furn_length = furn_z_max - furn_z_min

        # 计算overlap
        overlap_min = max(wall_z_min, furn_z_min)
        overlap_max = min(wall_z_max, furn_z_max)
        overlap_length = max(0, overlap_max - overlap_min)

    else:
        # 墙沿X方向延伸
        furn_length = furn_x_max - furn_x_min

        # 计算overlap
        overlap_min = max(wall_x_min, furn_x_min)
        overlap_max = min(wall_x_max, furn_x_max)
        overlap_length = max(0, overlap_max - overlap_min)

    # 4. 计算比例（相对于家具边长）
    if furn_length < 1e-6:
        return True  # 避免除0

    contact_ratio = overlap_length / furn_length

    # 5. 判断
    return contact_ratio >= min_ratio


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

    if not w_indices:
        return data, False

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
        if thick_axis == 0:  # X 轴是厚度轴
            w_normal = np.array([math.cos(w_ang), 0, math.sin(w_ang)])
        else:  # Z 轴是厚度轴
            w_normal = np.array([-math.sin(w_ang), 0, math.cos(w_ang)])

        walls_meta.append({
            'idx': wi,
            'pos': translations[wi],
            'normal': w_normal,
            'half_thick': min(w_sz[0], w_sz[2]) / 2.0,
            'size': w_sz  # 添加size信息用于contact ratio检查
        })

    # --- C. 家具处理 ---
    for fi in range(len(cls_indices)):
        if not is_valid_item(class_labels[fi]):
            continue
        label_name = CLASS_LABELS[cls_indices[fi]]
        if label_name not in FURNITURE_CLASSES:
            continue

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

            # AABB重叠检查
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

        # 按abs(gap)排序，选择最接近贴墙的墙
        wall_cands.sort(key=lambda x: abs(x['gap']))

        # 3. 尝试attach到墙，如果blocking或朝向不对或接触比例不够则fallback
        best_pos = f_pos.copy()
        success = False

        # 尝试所有候选墙
        for cand in wall_cands:
            wm = cand['meta']

            # ============ 检查1: 朝向规则（考虑旋转角度）============
            if not check_orientation_rule(label_name, f_sz, f_ang, wm, angle_threshold=0.9):
                # 不满足朝向规则，skip这个墙
                continue
            # ======================================================

            # ============ 检查2: 墙接触比例 ============
            if not check_wall_contact_ratio(f_pos, f_sz, f_ang, wm, min_ratio=MIN_WALL_CONTACT_RATIO):
                # 接触比例<50%，skip这个墙
                continue
            # ===========================================

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
                # 找到不blocking、朝向正确、接触比例够的墙
                best_pos = temp_pos
                success = True
                break  # 停止尝试其他墙
            # 如果blocking，继续尝试下一个墙

        if success:
            if not np.allclose(translations[fi], best_pos, atol=1e-3):
                translations[fi] = best_pos
                modified = True

    new_data = data.copy()
    new_data['translations'] = translations
    new_data['angles'] = angles_raw
    return new_data, modified

def do_cut_collision_areas(data, verbose=True):
    """
    切掉家具collision的部分（最后手段，只执行一次）
    按优先级：tub -> shower -> vanity
    不处理toilet
    """
    if verbose:
        print("\n>> Cutting collision areas (last resort)")

    CUTTABLE_FURNITURE = {'tub': 1, 'shower': 2, 'vanity': 3}

    sizes = data['sizes'].copy()
    translations = data['translations'].copy()
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    bounds = calculate_bounds(data)

    # 检测collision
    current_data = data.copy()
    current_data['sizes'] = sizes
    current_data['translations'] = translations

    collisions = detect_collisions(current_data, bounds)

    # 收集需要切的collision（带优先级）
    cut_tasks = []

    for coll in collisions:
        if coll['type'] == 'furniture_wall':
            furn_label = coll['furniture']['label']
            if furn_label in CUTTABLE_FURNITURE:
                priority = CUTTABLE_FURNITURE[furn_label]
                cut_tasks.append((priority, coll, None))

        elif coll['type'] == 'furniture_furniture':
            # 两个家具collision，选择优先级低的（数字大的）来切
            f1_label = coll['furniture1']['label']
            f2_label = coll['furniture2']['label']

            f1_priority = CUTTABLE_FURNITURE.get(f1_label, 999)
            f2_priority = CUTTABLE_FURNITURE.get(f2_label, 999)

            if f1_priority <= f2_priority and f1_label in CUTTABLE_FURNITURE:
                # 切f1（优先级高的先切）
                cut_tasks.append((f1_priority, coll, 'furniture1'))
            elif f2_priority < f1_priority and f2_label in CUTTABLE_FURNITURE:
                # 切f2
                cut_tasks.append((f2_priority, coll, 'furniture2'))

        elif coll['type'] == 'furniture_clearance':
            furn_label = coll['furniture']['label']
            if furn_label in CUTTABLE_FURNITURE:
                priority = CUTTABLE_FURNITURE[furn_label]
                cut_tasks.append((priority, coll, None))

        elif coll['type'] == 'out_of_bounds':
            # OOB也尝试切
            furn_label = coll['furniture']['label']
            if furn_label in CUTTABLE_FURNITURE:
                priority = CUTTABLE_FURNITURE[furn_label]
                cut_tasks.append((priority, coll, None))

    if not cut_tasks:
        if verbose:
            print("   No cuttable collisions found")
        return data, False

    # 按优先级排序（tub=1先切）
    cut_tasks.sort(key=lambda x: x[0])

    cuts_made = 0
    total_cut_volume = 0.0

    for task in cut_tasks:
        priority = task[0]
        coll = task[1]
        which_furniture = task[2]

        # 执行切割
        cut_amount = cut_furniture_for_collision(coll, sizes, translations, which_furniture, verbose=verbose)

        if cut_amount > 0:
            cuts_made += 1
            total_cut_volume += cut_amount

    if verbose:
        print(f"   Made {cuts_made} cuts, total volume reduced: {total_cut_volume:.4f}m³")

    new_data = data.copy()
    new_data['sizes'] = sizes
    new_data['translations'] = translations

    return new_data, cuts_made > 0

def cut_furniture_for_collision(collision, sizes, translations, which_furniture, verbose=True):
    """
    切掉家具collision的部分

    参数:
        collision: collision对象
        sizes: 家具sizes数组（会被修改）
        translations: 家具translations数组（会被修改）
        which_furniture: 'furniture1'或'furniture2'（仅用于furniture_furniture类型）
        verbose: 是否打印详细信息

    返回: 切掉的体积
    """
    # 确定要切的家具和障碍物
    if collision['type'] == 'furniture_wall':
        furn = collision['furniture']
        obstacle_aabb = collision['wall']['aabb']
        obstacle_name = 'wall'

    elif collision['type'] == 'furniture_clearance':
        furn = collision['furniture']
        obstacle_aabb = collision['door_clearance']['aabb']
        obstacle_name = 'door_clearance'

    elif collision['type'] == 'furniture_furniture':
        if which_furniture == 'furniture1':
            furn = collision['furniture1']
            obstacle_aabb = collision['furniture2']['aabb']
            obstacle_name = collision['furniture2']['label']
        else:
            furn = collision['furniture2']
            obstacle_aabb = collision['furniture1']['aabb']
            obstacle_name = collision['furniture1']['label']

    elif collision['type'] == 'out_of_bounds':
        furn = collision['furniture']
        bounds = collision['bounds']
        # 将bounds转换为AABB格式
        obstacle_aabb = bounds
        obstacle_name = 'bounds'
        # 但这里需要反向：切掉OOB的部分 = 保留在bounds内的部分
        return cut_oob_furniture(furn, sizes, translations, bounds, verbose=verbose)

    else:
        return 0.0

    furn_idx = furn['index']
    furn_label = furn['label']
    furn_aabb = furn['aabb']

    # ========== DEBUG: 记录切割前的状态 ==========
    old_size_x = sizes[furn_idx][0]
    old_size_z = sizes[furn_idx][2]
    old_pos_x = translations[furn_idx][0]
    old_pos_z = translations[furn_idx][2]
    # ============================================

    # 计算四个方向的overlap深度
    overlaps = []

    # X方向 - 右边overlap
    if furn_aabb[1] > obstacle_aabb[0] and furn_aabb[0] < obstacle_aabb[0]:
        depth = furn_aabb[1] - obstacle_aabb[0]
        overlaps.append((depth, 'cut_right', 0, +1))

    # X方向 - 左边overlap
    if furn_aabb[0] < obstacle_aabb[1] and furn_aabb[1] > obstacle_aabb[1]:
        depth = obstacle_aabb[1] - furn_aabb[0]
        overlaps.append((depth, 'cut_left', 0, -1))

    # Z方向 - 上边overlap
    if furn_aabb[3] > obstacle_aabb[2] and furn_aabb[2] < obstacle_aabb[2]:
        depth = furn_aabb[3] - obstacle_aabb[2]
        overlaps.append((depth, 'cut_top', 2, +1))

    # Z方向 - 下边overlap
    if furn_aabb[2] < obstacle_aabb[3] and furn_aabb[3] > obstacle_aabb[3]:
        depth = obstacle_aabb[3] - furn_aabb[2]
        overlaps.append((depth, 'cut_bottom', 2, -1))

    if not overlaps:
        return 0.0

    # 选择overlap最小的方向切（切得最少）
    overlaps.sort(key=lambda x: x[0])

    depth, cut_direction, axis, side = overlaps[0]

    # 计算新的size
    current_size = sizes[furn_idx][axis]
    new_size = current_size - depth

    # 检查MIN_SIZE
    min_sizes = MIN_FURNITURE_SIZES.get(furn_label)
    if min_sizes:
        min_size = min(min_sizes)

        if new_size < min_size:
            new_size = min_size
            actual_cut = current_size - min_size
        else:
            actual_cut = depth
    else:
        actual_cut = depth

    if actual_cut <= 0:
        return 0.0

    # 应用切割
    old_size = sizes[furn_idx][axis]
    sizes[furn_idx][axis] = new_size

    # 调整position（因为中心变了）
    shift = -side * actual_cut / 2.0

    if axis == 0:
        translations[furn_idx][0] += shift
    else:
        translations[furn_idx][2] += shift

    # 计算切掉的体积
    if axis == 0:
        other_size = sizes[furn_idx][2]
    else:
        other_size = sizes[furn_idx][0]

    height = sizes[furn_idx][1]
    cut_volume = actual_cut * other_size * height

    # ========== DEBUG: 输出切割详情 ==========
    if verbose:
        axis_name = 'X' if axis == 0 else 'Z'
        new_pos_x = translations[furn_idx][0]
        new_pos_z = translations[furn_idx][2]

        print(f"      [{furn_idx}] {furn_label} vs {obstacle_name}:")
        print(f"          Direction: {cut_direction}, Axis: {axis_name}")
        print(f"          Size: ({old_size_x:.3f}, {old_size_z:.3f}) → ({sizes[furn_idx][0]:.3f}, {sizes[furn_idx][2]:.3f})")
        print(f"          Pos:  ({old_pos_x:.3f}, {old_pos_z:.3f}) → ({new_pos_x:.3f}, {new_pos_z:.3f})")
        print(f"          Cut: {actual_cut:.3f}m, Volume: {cut_volume:.4f}m³")

        if new_size == min_size:
            print(f"          ⚠ Reached MIN_SIZE limit!")
    # =========================================

    return cut_volume


def cut_oob_furniture(furn, sizes, translations, bounds, verbose=True):
    """
    切掉OOB的部分
    """
    furn_idx = furn['index']
    furn_label = furn['label']
    furn_aabb = furn['aabb']

    # ========== DEBUG: 记录切割前状态 ==========
    old_size_x = sizes[furn_idx][0]
    old_size_z = sizes[furn_idx][2]
    old_pos_x = translations[furn_idx][0]
    old_pos_z = translations[furn_idx][2]
    # ==========================================

    min_x, max_x, min_z, max_z = bounds
    fmin_x, fmax_x, fmin_z, fmax_z = furn_aabb

    cuts = []

    # 检查四个方向的OOB
    if fmin_x < min_x:
        depth = min_x - fmin_x
        cuts.append((depth, 0, -1, 'left'))

    if fmax_x > max_x:
        depth = fmax_x - max_x
        cuts.append((depth, 0, +1, 'right'))

    if fmin_z < min_z:
        depth = min_z - fmin_z
        cuts.append((depth, 2, -1, 'bottom'))

    if fmax_z > max_z:
        depth = fmax_z - max_z
        cuts.append((depth, 2, +1, 'top'))

    if cuts and verbose:
        print(f"      [{furn_idx}] {furn_label} vs bounds (OOB):")

    total_volume = 0.0

    for depth, axis, side, direction_name in cuts:
        current_size = sizes[furn_idx][axis]
        new_size = current_size - depth

        # 检查MIN_SIZE
        min_sizes = MIN_FURNITURE_SIZES.get(furn_label)
        if min_sizes:
            min_size = min(min_sizes)
            if new_size < min_size:
                new_size = min_size
                actual_cut = current_size - min_size
            else:
                actual_cut = depth
        else:
            actual_cut = depth

        if actual_cut <= 0:
            continue

        # 应用切割
        sizes[furn_idx][axis] = new_size

        # 调整position
        shift = -side * actual_cut / 2.0
        if axis == 0:
            translations[furn_idx][0] += shift
        else:
            translations[furn_idx][2] += shift

        # 计算体积
        if axis == 0:
            other_size = sizes[furn_idx][2]
        else:
            other_size = sizes[furn_idx][0]

        height = sizes[furn_idx][1]
        cut_vol = actual_cut * other_size * height
        total_volume += cut_vol

        # ========== DEBUG ==========
        if verbose:
            axis_name = 'X' if axis == 0 else 'Z'
            print(f"          Cut {direction_name} ({axis_name}): {actual_cut:.3f}m, Volume: {cut_vol:.4f}m³")
        # ===========================

    if cuts and verbose:
        new_pos_x = translations[furn_idx][0]
        new_pos_z = translations[furn_idx][2]
        print(f"          Size: ({old_size_x:.3f}, {old_size_z:.3f}) → ({sizes[furn_idx][0]:.3f}, {sizes[furn_idx][2]:.3f})")
        print(f"          Pos:  ({old_pos_x:.3f}, {old_pos_z:.3f}) → ({new_pos_x:.3f}, {new_pos_z:.3f})")

    return total_volume

def do_expand_furniture(data, max_iterations=20):
    """
    尝试拉大家具的长边（在切割后优化）
    只处理vanity, shower, tub

    规则：
    - 只拉长边
    - 不超过MAX_SIZE
    - 不超过MAX_AREA
    - 不产生新collision
    - 如果靠墙，保持靠墙边不动；否则居中扩展
    """
    print("\n>> Expanding furniture (optimization after cutting)")

    EXPANDABLE_FURNITURE = {'vanity', 'shower', 'tub'}

    # 最大尺寸 (长, 短)
    MAX_FURNITURE_SIZES = {
        'vanity': (1.5, 0.8),
        'shower': (1.2, 1.2),
        'tub': (2.0, 1.0),
    }

    # 最大面积
    MAX_FURNITURE_AREAS = {
        'vanity': 1.2,   # 1.2m²
        'shower': 1.44,  # 1.44m²
        'tub': 2.0,      # 2.0m²
    }

    EXPAND_STEP = 0.05  # 每次拉大5cm
    WALL_ATTACHMENT_THRESHOLD = 0.1  # 距离墙<10cm算靠墙

    sizes = data['sizes'].copy()
    translations = data['translations'].copy()
    angles = data['angles']
    class_labels = data['class_labels']

    if class_labels.ndim > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()

    bounds = calculate_bounds(data)

    # 收集墙的AABB
    wall_aabbs = []
    for i in range(len(cls_indices)):
        if not is_valid_item(class_labels[i]):
            continue
        if CLASS_LABELS[cls_indices[i]] == 'wall':
            pos = translations[i]
            size = sizes[i]
            ang = get_angle(angles[i])
            wall_aabbs.append(calculate_aabb_2d(pos, size, ang))

    expanded_count = 0
    total_expansion = 0.0
    attempted_count = 0

    # 处理每个家具
    for furn_idx in range(len(cls_indices)):
        if not is_valid_item(class_labels[furn_idx]):
            continue

        furn_label = CLASS_LABELS[cls_indices[furn_idx]]

        if furn_label not in EXPANDABLE_FURNITURE:
            continue

        attempted_count += 1

        furn_pos = translations[furn_idx]
        furn_size = sizes[furn_idx]
        furn_ang = get_angle(angles[furn_idx])

        # 判断长边
        if furn_size[0] > furn_size[2]:
            long_axis = 0  # X是长边
            short_axis = 2
        else:
            long_axis = 2  # Z是长边
            short_axis = 0

        # 获取限制
        max_size_tuple = MAX_FURNITURE_SIZES[furn_label]
        max_long_size = max(max_size_tuple)
        max_area = MAX_FURNITURE_AREAS[furn_label]

        # 检查是否靠墙（长边方向）
        furn_aabb = calculate_aabb_2d(furn_pos, furn_size, furn_ang)
        wall_attachment = check_wall_attachment_direction(
            furn_aabb, wall_aabbs, long_axis, WALL_ATTACHMENT_THRESHOLD
        )

        # ========== DEBUG: 开始扩展前的信息 ==========
        print(f"\n   Attempting to expand [{furn_idx}] {furn_label}:")
        print(f"      Current size: ({furn_size[0]:.3f}, {furn_size[2]:.3f})")
        print(f"      Long axis: {'X' if long_axis == 0 else 'Z'}")
        print(f"      Max allowed size: {max_long_size:.3f}m")
        print(f"      Max allowed area: {max_area:.3f}m²")
        print(f"      Current area: {(furn_size[0] * furn_size[2]):.3f}m²")
        if wall_attachment:
            print(f"      Wall attachment: {wall_attachment['side']} side (shift direction: {wall_attachment['shift_direction']:+.0f})")
        else:
            print(f"      Wall attachment: None (will expand centered)")
        # ==============================================

        # 记录初始尺寸
        initial_size = furn_size[long_axis]
        iterations_done = 0

        # 逐步拉大
        for iteration in range(max_iterations):
            current_size = sizes[furn_idx][long_axis]
            new_size = current_size + EXPAND_STEP

            # 检查1: 不超过max_size
            if new_size > max_long_size:
                print(f"      ✗ Step {iteration+1}: Would exceed max_size ({new_size:.3f} > {max_long_size:.3f})")
                break

            # 检查2: 不超过max_area
            other_size = sizes[furn_idx][short_axis]
            new_area = new_size * other_size
            if new_area > max_area:
                print(f"      ✗ Step {iteration+1}: Would exceed max_area ({new_area:.3f} > {max_area:.3f})")
                break

            # 临时应用扩展（用于collision检测）
            test_sizes = sizes.copy()
            test_translations = translations.copy()

            test_sizes[furn_idx][long_axis] = new_size

            # 调整position（根据是否靠墙）
            if wall_attachment is not None:
                shift = wall_attachment['shift_direction'] * EXPAND_STEP / 2.0
                if long_axis == 0:
                    test_translations[furn_idx][0] += shift
                else:
                    test_translations[furn_idx][2] += shift

            # 检查3: 不产生新collision
            test_data = data.copy()
            test_data['sizes'] = test_sizes
            test_data['translations'] = test_translations

            collisions = detect_collisions(test_data, bounds)

            # 检查是否有涉及这个家具的collision
            has_new_collision = False
            collision_types = []
            for coll in collisions:
                if collision_involves_furniture(coll, furn_idx):
                    has_new_collision = True
                    collision_types.append(coll['type'])

            if has_new_collision:
                # ========== DEBUG: collision详情 ==========
                print(f"      ✗ Step {iteration+1}: Would create collision")
                print(f"          Size would be: {new_size:.3f}m")
                print(f"          Collision types: {', '.join(set(collision_types))}")
                # =========================================
                break

            # ========== DEBUG: 成功扩展 ==========
            if iteration == 0 or (iteration + 1) % 5 == 0:  # 每5步输出一次
                print(f"      ✓ Step {iteration+1}: {current_size:.3f}m → {new_size:.3f}m (area: {new_area:.3f}m²)")
            # ====================================

            # 安全，应用扩展
            sizes[furn_idx][long_axis] = new_size
            if wall_attachment is not None:
                shift = wall_attachment['shift_direction'] * EXPAND_STEP / 2.0
                if long_axis == 0:
                    translations[furn_idx][0] += shift
                else:
                    translations[furn_idx][2] += shift

            iterations_done += 1

        # 统计
        final_size = sizes[furn_idx][long_axis]
        if final_size > initial_size + 0.001:
            expansion = final_size - initial_size
            expanded_count += 1
            total_expansion += expansion

            axis_name = 'X' if long_axis == 0 else 'Z'
            attachment_info = f"(wall-attached)" if wall_attachment else "(centered)"

            # ========== DEBUG: 扩展总结 ==========
            print(f"      ✓ EXPANDED: {axis_name} {initial_size:.3f}m → {final_size:.3f}m (+{expansion:.3f}m, {iterations_done} steps) {attachment_info}")
            final_area = sizes[furn_idx][0] * sizes[furn_idx][2]
            print(f"      Final size: ({sizes[furn_idx][0]:.3f}, {sizes[furn_idx][2]:.3f}), Area: {final_area:.3f}m²")
            # ====================================
        else:
            # ========== DEBUG: 无法扩展 ==========
            print(f"      ✗ NOT EXPANDED: Already at limit or blocked immediately")
            # ====================================

    # ========== 总结 ==========
    print(f"\n>> Expansion summary:")
    print(f"   Total items attempted: {attempted_count}")
    print(f"   Successfully expanded: {expanded_count}")
    print(f"   Total expansion: {total_expansion:.3f}m")
    # =========================

    new_data = data.copy()
    new_data['sizes'] = sizes
    new_data['translations'] = translations

    return new_data, expanded_count > 0

def check_wall_attachment_direction(furn_aabb, wall_aabbs, long_axis, threshold):
    """
    检查家具在长边方向是否靠墙

    返回:
        None: 不靠墙
        dict: {'side': 'min'或'max', 'shift_direction': +1或-1}
    """
    fmin_x, fmax_x, fmin_z, fmax_z = furn_aabb

    if long_axis == 0:
        # 长边在X方向，检查X方向的两端是否靠墙
        for wall_aabb in wall_aabbs:
            wmin_x, wmax_x, wmin_z, wmax_z = wall_aabb

            # 检查Z方向是否有重叠（墙和家具在同一水平位置）
            if fmax_z < wmin_z or fmin_z > wmax_z:
                continue

            # 检查家具左边（min_x）是否靠墙
            if abs(fmin_x - wmax_x) < threshold:
                # 左边靠墙，扩展时向右（+方向）移动
                return {'side': 'min', 'shift_direction': +1.0}

            # 检查家具右边（max_x）是否靠墙
            if abs(fmax_x - wmin_x) < threshold:
                # 右边靠墙，扩展时向左（-方向）移动
                return {'side': 'max', 'shift_direction': -1.0}

    else:  # long_axis == 2
        # 长边在Z方向，检查Z方向的两端是否靠墙
        for wall_aabb in wall_aabbs:
            wmin_x, wmax_x, wmin_z, wmax_z = wall_aabb

            # 检查X方向是否有重叠
            if fmax_x < wmin_x or fmin_x > wmax_x:
                continue

            # 检查家具下边（min_z）是否靠墙
            if abs(fmin_z - wmax_z) < threshold:
                # 下边靠墙，扩展时向上（+方向）移动
                return {'side': 'min', 'shift_direction': +1.0}

            # 检查家具上边（max_z）是否靠墙
            if abs(fmax_z - wmin_z) < threshold:
                # 上边靠墙，扩展时向下（-方向）移动
                return {'side': 'max', 'shift_direction': -1.0}

    return None  # 不靠墙


def collision_involves_furniture(collision, furn_idx):
    """检查collision是否涉及指定的家具"""
    if collision['type'] == 'furniture_wall':
        return collision['furniture']['index'] == furn_idx

    elif collision['type'] == 'furniture_clearance':
        return collision['furniture']['index'] == furn_idx

    elif collision['type'] == 'furniture_furniture':
        return (collision['furniture1']['index'] == furn_idx or
                collision['furniture2']['index'] == furn_idx)

    elif collision['type'] == 'out_of_bounds':
        return collision['furniture']['index'] == furn_idx

    return False

def post_process_data(input_data, verbose=True):
    if verbose:
        debug_print("INPUT DATA", input_data)

    # Step 0: 移除invalid items (家具后面的architecture items)
    output_data, modified_remove = do_remove_invalid_items(input_data, verbose=verbose)

    max_trials = 5
    #max_trials = 1
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
        if trial >= 3:
            output_data, modified_oob = do_remove_oob_items(output_data, min_inside_ratio=MIN_INSIDE_RATIO, verbose=verbose)
            modified = modified or modified_oob

        # Step 5: 移除有严重collision的家具
        output_data, modified_severe = do_remove_collision_items(output_data, thresholds=SEVERE_COLLISION_THRESHOLDS, verbose=verbose)
        modified = modified or modified_severe

        # Step 6: Cut area
        output_data, modified_cut = do_cut_collision_areas(output_data, verbose=verbose)
        modified = modified or modified_cut

        # Step 7: Expand size
        #output_data, modified_expand = do_expand_furniture(output_data)

        if not modified:
            if verbose:
                print(f">> Stopped earlier at [{trial+1}/{max_trials}].")
            break

    if verbose:
        debug_print("OUTPUT DATA", output_data)
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
