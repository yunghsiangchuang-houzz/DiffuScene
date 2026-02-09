"""
Bathroom Scene Scoring System

This module scores generated bathroom scene samples based on:
1. Collision detection (fixture-fixture, fixture-architecture)
2. Boundary compliance (fixtures within room, not overlapping walls/doors/windows)
3. Wall attachment (fixtures should be against walls)
4. Clearance requirements (minimum functional space)
5. Circulation path (accessible path door → vanity → toilet → tub/shower)
6. Wet wall optimization (fewer wet walls = better)
7. Spatial proximity logic (vanity near door, near toilet)

Reference:
- docs/bathroom_guide.txt
- PhyScene CVPR'24 (collision-free scene generation)
- CHOrD (collision-free indoor scenes via 2D layout representations)

Usage:
    scorer = BathroomSceneScorer()
    scores = scorer.score_batch(batch_boxes, arch_boxes)  # (B,) scores
    top_k_indices = torch.topk(scores, k=5).indices
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# Class labels matching the Houzz bathroom preprocessing
CLASS_LABELS = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window", "empty"]
FIXTURE_TYPES = {"vanity", "toilet", "shower", "tub"}
ARCH_TYPES = {"floor", "wall", "door", "window"}

# Indices for quick lookup
IDX_VANITY = CLASS_LABELS.index("vanity")
IDX_TOILET = CLASS_LABELS.index("toilet")
IDX_SHOWER = CLASS_LABELS.index("shower")
IDX_TUB = CLASS_LABELS.index("tub")
IDX_FLOOR = CLASS_LABELS.index("floor")
IDX_WALL = CLASS_LABELS.index("wall")
IDX_DOOR = CLASS_LABELS.index("door")
IDX_WINDOW = CLASS_LABELS.index("window")
IDX_EMPTY = CLASS_LABELS.index("empty")

FIXTURE_INDICES = {IDX_VANITY, IDX_TOILET, IDX_SHOWER, IDX_TUB}


@dataclass
class ScoringWeights:
    """Weights for combining different scoring components."""
    # Critical constraints (high penalty for violation)
    collision_fixture_fixture: float = 10.0    # Fixtures must not overlap each other
    collision_fixture_wall: float = 8.0        # Fixtures should not overlap walls
    collision_fixture_door: float = 8.0        # Fixtures should not block doors
    boundary_violation: float = 10.0           # Fixtures must be inside room
    
    # Functional constraints (bathroom guide requirements)
    clearance_toilet: float = 5.0              # 60cm front clearance for toilet
    clearance_vanity: float = 5.0              # 60cm front clearance for vanity  
    clearance_tub_shower: float = 4.0          # 60cm side access for tub/shower
    gap_between_fixtures: float = 3.0          # 20-30cm minimum gap
    
    # Soft preferences (quality scoring)
    wall_attachment: float = 3.0               # Fixtures should touch walls
    circulation_path: float = 5.0              # Clear path door→vanity→toilet→tub
    wet_wall_count: float = 2.0                # Fewer wet walls = better
    vanity_proximity_door: float = 2.0         # Vanity should be near door
    vanity_proximity_toilet: float = 1.0       # Vanity should be near toilet
    
    # Completeness
    required_fixtures: float = 5.0             # Must have toilet + vanity


@dataclass  
class ClearanceRequirements:
    """Minimum clearance requirements in meters (from bathroom_guide.txt)."""
    toilet_front: float = 0.60                 # 60cm in front of toilet
    vanity_front: float = 0.60                 # 60cm standing space at vanity
    tub_shower_side: float = 0.60              # 60cm path to enter tub/shower
    fixture_gap_min: float = 0.20              # 20cm minimum gap between fixtures
    fixture_gap_preferred: float = 0.30        # 30cm preferred gap
    wall_attachment_threshold: float = 0.15    # Within 15cm = touching wall
    circulation_path_width: float = 0.60       # 60cm wide path


class BathroomSceneScorer:
    """
    Scores bathroom scene samples for quality ranking.
    
    Generates 128 samples per scene → ranks by score → returns top-K.
    """
    
    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        clearance: Optional[ClearanceRequirements] = None,
    ):
        self.weights = weights or ScoringWeights()
        self.clearance = clearance or ClearanceRequirements()
    
    def score_batch(
        self,
        translations: np.ndarray,      # (B, N, 3) - x, y, z centers
        sizes: np.ndarray,             # (B, N, 3) - half-extents
        angles: np.ndarray,            # (B, N, 1 or 2) - rotation
        class_labels: np.ndarray,      # (B, N, C) - one-hot or (B, N) indices
        arch_translations: np.ndarray, # (B, M, 3) - architecture centers (walls, floor, etc)
        arch_sizes: np.ndarray,        # (B, M, 3) - architecture half-extents
        arch_class_labels: np.ndarray, # (B, M, C) or (B, M) - architecture classes
    ) -> np.ndarray:
        """
        Score a batch of scene samples.
        
        Returns:
            scores: (B,) array of scores. Higher = better.
        """
        B = translations.shape[0]
        scores = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            scores[b] = self._score_single(
                translations[b], sizes[b], angles[b], class_labels[b],
                arch_translations[b], arch_sizes[b], arch_class_labels[b]
            )
        
        return scores
    
    def _score_single(
        self,
        translations: np.ndarray,      # (N, 3)
        sizes: np.ndarray,             # (N, 3) 
        angles: np.ndarray,            # (N, 1 or 2)
        class_labels: np.ndarray,      # (N, C) or (N,)
        arch_translations: np.ndarray, # (M, 3)
        arch_sizes: np.ndarray,        # (M, 3)
        arch_class_labels: np.ndarray, # (M, C) or (M,)
    ) -> float:
        """Score a single scene sample."""
        
        # Convert one-hot to indices if needed
        if class_labels.ndim == 2:
            class_indices = np.argmax(class_labels, axis=-1)
        else:
            class_indices = class_labels
            
        if arch_class_labels.ndim == 2:
            arch_indices = np.argmax(arch_class_labels, axis=-1)
        else:
            arch_indices = arch_class_labels
        
        # Extract fixture and architecture data
        fixtures = self._extract_fixtures(translations, sizes, angles, class_indices)
        walls = self._extract_by_type(arch_translations, arch_sizes, arch_indices, IDX_WALL)
        doors = self._extract_by_type(arch_translations, arch_sizes, arch_indices, IDX_DOOR)
        floors = self._extract_by_type(arch_translations, arch_sizes, arch_indices, IDX_FLOOR)
        
        # Start with base score (higher is better, penalties subtract)
        score = 100.0
        
        # === CRITICAL CONSTRAINTS (hard penalties) ===
        
        # 1. Collision: fixture-fixture
        collision_penalty = self._compute_fixture_collision_penalty(fixtures)
        score -= collision_penalty * self.weights.collision_fixture_fixture
        
        # 2. Collision: fixture-wall overlap
        wall_collision_penalty = self._compute_arch_collision_penalty(fixtures, walls)
        score -= wall_collision_penalty * self.weights.collision_fixture_wall
        
        # 3. Collision: fixture-door (blocking)
        door_collision_penalty = self._compute_door_blocking_penalty(fixtures, doors)
        score -= door_collision_penalty * self.weights.collision_fixture_door
        
        # 4. Boundary violation (fixtures outside floor)
        boundary_penalty = self._compute_boundary_violation(fixtures, floors)
        score -= boundary_penalty * self.weights.boundary_violation
        
        # === FUNCTIONAL CONSTRAINTS ===
        
        # 5. Clearance requirements
        clearance_penalty = self._compute_clearance_penalty(fixtures, walls)
        score -= clearance_penalty * self.weights.clearance_toilet
        
        # 6. Gap between fixtures
        gap_penalty = self._compute_gap_penalty(fixtures)
        score -= gap_penalty * self.weights.gap_between_fixtures
        
        # === SOFT PREFERENCES (quality bonuses) ===
        
        # 7. Wall attachment bonus
        wall_bonus = self._compute_wall_attachment_bonus(fixtures, walls)
        score += wall_bonus * self.weights.wall_attachment
        
        # 8. Circulation path bonus
        circulation_bonus = self._compute_circulation_bonus(fixtures, doors)
        score += circulation_bonus * self.weights.circulation_path
        
        # 9. Wet wall optimization (fewer = better)
        wet_wall_bonus = self._compute_wet_wall_bonus(fixtures, walls)
        score += wet_wall_bonus * self.weights.wet_wall_count
        
        # 10. Vanity proximity bonuses
        vanity_door_bonus = self._compute_vanity_door_proximity(fixtures, doors)
        score += vanity_door_bonus * self.weights.vanity_proximity_door
        
        vanity_toilet_bonus = self._compute_vanity_toilet_proximity(fixtures)
        score += vanity_toilet_bonus * self.weights.vanity_proximity_toilet
        
        # 11. Required fixtures check
        completeness_bonus = self._compute_completeness_bonus(fixtures)
        score += completeness_bonus * self.weights.required_fixtures
        
        return max(0.0, score)  # Clamp to non-negative
    
    # ==================== HELPER METHODS ====================
    
    def _extract_fixtures(
        self, 
        trans: np.ndarray, 
        sizes: np.ndarray, 
        angles: np.ndarray,
        class_indices: np.ndarray
    ) -> Dict[str, List[Dict]]:
        """Extract fixture objects grouped by type."""
        fixtures = {name: [] for name in FIXTURE_TYPES}
        
        for i, cls_idx in enumerate(class_indices):
            if cls_idx in FIXTURE_INDICES and cls_idx != IDX_EMPTY:
                label = CLASS_LABELS[cls_idx]
                fixtures[label].append({
                    'center': trans[i],           # (3,)
                    'size': sizes[i],             # (3,) half-extents
                    'angle': angles[i],           # (1,) or (2,)
                    'bbox': self._get_aabb(trans[i], sizes[i]),  # (6,) min/max
                })
        
        return fixtures
    
    def _extract_by_type(
        self,
        trans: np.ndarray,
        sizes: np.ndarray, 
        class_indices: np.ndarray,
        target_idx: int
    ) -> List[Dict]:
        """Extract objects of a specific type."""
        objects = []
        for i, cls_idx in enumerate(class_indices):
            if cls_idx == target_idx:
                objects.append({
                    'center': trans[i],
                    'size': sizes[i],
                    'bbox': self._get_aabb(trans[i], sizes[i]),
                })
        return objects
    
    def _get_aabb(self, center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
        """Get axis-aligned bounding box as [x1, y1, z1, x2, y2, z2]."""
        return np.concatenate([center - half_size, center + half_size])
    
    def _boxes_overlap(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute 3D IoU-style overlap between two AABBs. Returns overlap volume."""
        # bbox format: [x1, y1, z1, x2, y2, z2]
        lt = np.maximum(bbox1[:3], bbox2[:3])
        rb = np.minimum(bbox1[3:], bbox2[3:])
        wh = np.maximum(rb - lt, 0)
        return wh[0] * wh[1] * wh[2]
    
    def _boxes_overlap_2d(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute 2D overlap (floor plan) between two AABBs. Uses X and Z axes."""
        # Using X (index 0) and Z (index 2) for floor plan
        lt = np.maximum(bbox1[[0, 2]], bbox2[[0, 2]])
        rb = np.minimum(bbox1[[3, 5]], bbox2[[3, 5]])
        wh = np.maximum(rb - lt, 0)
        return wh[0] * wh[1]
    
    def _distance_2d(self, center1: np.ndarray, center2: np.ndarray) -> float:
        """2D distance on floor plan (X, Z)."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[2] - center2[2])**2)
    
    def _min_distance_to_box(self, point: np.ndarray, bbox: np.ndarray) -> float:
        """Minimum 2D distance from point to box edge."""
        # Clamp point to box, then compute distance
        clamped = np.clip(point[[0, 2]], bbox[[0, 2]], bbox[[3, 5]])
        return np.sqrt(np.sum((point[[0, 2]] - clamped)**2))
    
    # ==================== SCORING COMPONENTS ====================
    
    def _compute_fixture_collision_penalty(self, fixtures: Dict) -> float:
        """Compute penalty for fixtures overlapping each other."""
        all_fixtures = []
        for fixture_list in fixtures.values():
            all_fixtures.extend(fixture_list)
        
        if len(all_fixtures) < 2:
            return 0.0
        
        total_overlap = 0.0
        for i in range(len(all_fixtures)):
            for j in range(i + 1, len(all_fixtures)):
                overlap = self._boxes_overlap_2d(
                    all_fixtures[i]['bbox'], 
                    all_fixtures[j]['bbox']
                )
                if overlap > 0:
                    # Normalize by fixture area for consistent penalty
                    area_i = (all_fixtures[i]['size'][0] * 2) * (all_fixtures[i]['size'][2] * 2)
                    total_overlap += overlap / max(area_i, 0.01)
        
        return total_overlap
    
    def _compute_arch_collision_penalty(self, fixtures: Dict, walls: List) -> float:
        """Compute penalty for fixtures overlapping walls (excessive penetration)."""
        penalty = 0.0
        threshold = 0.10  # Allow 10cm for wall attachment, penalize more
        
        for fixture_list in fixtures.values():
            for fix in fixture_list:
                for wall in walls:
                    overlap = self._boxes_overlap_2d(fix['bbox'], wall['bbox'])
                    if overlap > 0:
                        # Check if it's just attachment (acceptable) or penetration (bad)
                        # Compute penetration depth
                        fix_center = fix['center'][[0, 2]]
                        wall_center = wall['center'][[0, 2]]
                        
                        # Determine which axis is "thin" (wall direction)
                        wall_sizes_2d = wall['size'][[0, 2]]
                        thin_axis = np.argmin(wall_sizes_2d)
                        
                        # Penetration = how much center is past wall face
                        penetration = abs(fix_center[thin_axis] - wall_center[thin_axis])
                        penetration -= wall_sizes_2d[thin_axis]  # Distance to wall face
                        
                        if penetration > threshold:
                            penalty += penetration - threshold
        
        return penalty
    
    def _compute_door_blocking_penalty(self, fixtures: Dict, doors: List) -> float:
        """Compute penalty for fixtures blocking door swing area."""
        penalty = 0.0
        
        for door in doors:
            # Estimate door swing area (extend door footprint in swing direction)
            door_size = door['size']
            door_width = max(door_size[0], door_size[2]) * 2  # Longer dimension
            
            # Create swing zone: extend door bbox
            swing_bbox = door['bbox'].copy()
            swing_extension = door_width  # Door swings its full width
            
            # Expand bbox (simplified: expand in all horizontal directions)
            swing_bbox[0] -= swing_extension * 0.5
            swing_bbox[2] -= swing_extension * 0.5
            swing_bbox[3] += swing_extension * 0.5
            swing_bbox[5] += swing_extension * 0.5
            
            for fixture_list in fixtures.values():
                for fix in fixture_list:
                    overlap = self._boxes_overlap_2d(fix['bbox'], swing_bbox)
                    if overlap > 0:
                        penalty += overlap / (door_width * door_width + 0.01)
        
        return penalty
    
    def _compute_boundary_violation(self, fixtures: Dict, floors: List) -> float:
        """Compute penalty for fixtures outside floor boundary."""
        if not floors:
            return 0.0  # Can't check if no floor
        
        # Compute floor bounding box (union of all floor segments)
        floor_min = np.array([np.inf, np.inf, np.inf])
        floor_max = np.array([-np.inf, -np.inf, -np.inf])
        for floor in floors:
            floor_min = np.minimum(floor_min, floor['bbox'][:3])
            floor_max = np.maximum(floor_max, floor['bbox'][3:])
        
        penalty = 0.0
        for fixture_list in fixtures.values():
            for fix in fixture_list:
                # Check if fixture bbox is outside floor bbox
                fix_bbox = fix['bbox']
                
                # Compute how much fixture extends outside floor (in 2D)
                outside_min = np.maximum(floor_min[[0, 2]] - fix_bbox[[0, 2]], 0)
                outside_max = np.maximum(fix_bbox[[3, 5]] - floor_max[[0, 2]], 0)
                
                total_outside = np.sum(outside_min) + np.sum(outside_max)
                penalty += total_outside
        
        return penalty
    
    def _compute_clearance_penalty(self, fixtures: Dict, walls: List) -> float:
        """Compute penalty for insufficient clearance in front of fixtures."""
        penalty = 0.0
        
        # Check toilet clearance (60cm front)
        for toilet in fixtures.get('toilet', []):
            front_clear = self._check_front_clearance(
                toilet, fixtures, walls, self.clearance.toilet_front
            )
            if not front_clear:
                penalty += 1.0
        
        # Check vanity clearance (60cm front)
        for vanity in fixtures.get('vanity', []):
            front_clear = self._check_front_clearance(
                vanity, fixtures, walls, self.clearance.vanity_front
            )
            if not front_clear:
                penalty += 1.0
        
        # Check tub/shower side access (60cm)
        for tub in fixtures.get('tub', []):
            side_clear = self._check_side_clearance(
                tub, fixtures, walls, self.clearance.tub_shower_side
            )
            if not side_clear:
                penalty += 0.5
                
        for shower in fixtures.get('shower', []):
            side_clear = self._check_side_clearance(
                shower, fixtures, walls, self.clearance.tub_shower_side
            )
            if not side_clear:
                penalty += 0.5
        
        return penalty
    
    def _check_front_clearance(
        self, 
        fixture: Dict, 
        all_fixtures: Dict,
        walls: List,
        required: float
    ) -> bool:
        """Check if there's sufficient clearance in front of a fixture."""
        # Simplified: create clearance box in front and check for obstacles
        center = fixture['center']
        size = fixture['size']
        
        # Front is typically -Z direction (simplified, should use angle)
        clearance_box = np.array([
            center[0] - size[0],      # x1
            center[1] - size[1],      # y1  
            center[2] - size[2] - required,  # z1 (front)
            center[0] + size[0],      # x2
            center[1] + size[1],      # y2
            center[2] - size[2],      # z2
        ])
        
        # Check collision with other fixtures
        for fixture_type, fixture_list in all_fixtures.items():
            for other in fixture_list:
                if np.allclose(other['center'], fixture['center']):
                    continue  # Skip self
                if self._boxes_overlap_2d(clearance_box, other['bbox']) > 0.01:
                    return False
        
        # Check collision with walls
        for wall in walls:
            if self._boxes_overlap_2d(clearance_box, wall['bbox']) > 0.01:
                return False
        
        return True
    
    def _check_side_clearance(
        self,
        fixture: Dict,
        all_fixtures: Dict, 
        walls: List,
        required: float
    ) -> bool:
        """Check if there's clearance on at least one side for access."""
        center = fixture['center']
        size = fixture['size']
        
        # Check both sides (simplified: +X and -X)
        sides = [
            np.array([center[0] + size[0], center[1], center[2], 
                     center[0] + size[0] + required, center[1] + size[1], center[2] + size[2]]),
            np.array([center[0] - size[0] - required, center[1], center[2],
                     center[0] - size[0], center[1] + size[1], center[2] + size[2]]),
        ]
        
        for side_box in sides:
            clear = True
            for fixture_type, fixture_list in all_fixtures.items():
                for other in fixture_list:
                    if np.allclose(other['center'], fixture['center']):
                        continue
                    if self._boxes_overlap_2d(side_box, other['bbox']) > 0.01:
                        clear = False
                        break
                if not clear:
                    break
            
            if clear:
                for wall in walls:
                    if self._boxes_overlap_2d(side_box, wall['bbox']) > 0.01:
                        clear = False
                        break
            
            if clear:
                return True  # At least one side is clear
        
        return False
    
    def _compute_gap_penalty(self, fixtures: Dict) -> float:
        """Compute penalty for fixtures too close together (< 20cm gap)."""
        all_fixtures = []
        for fixture_list in fixtures.values():
            all_fixtures.extend(fixture_list)
        
        if len(all_fixtures) < 2:
            return 0.0
        
        penalty = 0.0
        min_gap = self.clearance.fixture_gap_min
        
        for i in range(len(all_fixtures)):
            for j in range(i + 1, len(all_fixtures)):
                # Compute gap (distance between edges)
                bbox_i = all_fixtures[i]['bbox']
                bbox_j = all_fixtures[j]['bbox']
                
                # Gap in each axis
                gap_x = max(0, max(bbox_i[0], bbox_j[0]) - min(bbox_i[3], bbox_j[3]))
                gap_z = max(0, max(bbox_i[2], bbox_j[2]) - min(bbox_i[5], bbox_j[5]))
                
                # If overlapping, gap = 0 (handled by collision penalty)
                if self._boxes_overlap_2d(bbox_i, bbox_j) > 0:
                    continue
                
                # Minimum of the two gaps
                gap = min(gap_x, gap_z) if gap_x > 0 and gap_z > 0 else max(gap_x, gap_z)
                
                if gap < min_gap:
                    penalty += (min_gap - gap) / min_gap
        
        return penalty
    
    def _compute_wall_attachment_bonus(self, fixtures: Dict, walls: List) -> float:
        """Compute bonus for fixtures properly attached to walls."""
        if not walls:
            return 0.0
        
        bonus = 0.0
        threshold = self.clearance.wall_attachment_threshold
        
        total_fixtures = sum(len(fl) for fl in fixtures.values())
        if total_fixtures == 0:
            return 0.0
        
        attached_count = 0
        for fixture_list in fixtures.values():
            for fix in fixture_list:
                # Check if fixture is close to any wall
                for wall in walls:
                    dist = self._compute_fixture_wall_distance(fix, wall)
                    if dist < threshold:
                        attached_count += 1
                        break
        
        # Bonus proportional to attachment ratio
        bonus = attached_count / total_fixtures
        return bonus
    
    def _compute_fixture_wall_distance(self, fixture: Dict, wall: Dict) -> float:
        """Compute minimum distance from fixture edge to wall face."""
        fix_bbox = fixture['bbox']
        wall_bbox = wall['bbox']
        
        # Find the thin axis of wall
        wall_size = wall['size']
        thin_axis = 0 if wall_size[0] < wall_size[2] else 2
        
        # Distance from fixture to wall face
        if thin_axis == 0:
            # Wall is thin in X, fixture approaches from X direction
            dist = min(
                abs(fix_bbox[3] - wall_bbox[0]),  # fixture right to wall left
                abs(fix_bbox[0] - wall_bbox[3])   # fixture left to wall right
            )
        else:
            # Wall is thin in Z
            dist = min(
                abs(fix_bbox[5] - wall_bbox[2]),
                abs(fix_bbox[2] - wall_bbox[5])
            )
        
        return max(0, dist)
    
    def _compute_circulation_bonus(self, fixtures: Dict, doors: List) -> float:
        """Compute bonus for clear circulation path: door → vanity → toilet → tub/shower."""
        if not doors:
            return 0.0
        
        # Find door center
        door_center = doors[0]['center'][[0, 2]]
        
        # Build path points
        path_points = [door_center]
        
        if fixtures['vanity']:
            vanity_center = fixtures['vanity'][0]['center'][[0, 2]]
            path_points.append(vanity_center)
        
        if fixtures['toilet']:
            toilet_center = fixtures['toilet'][0]['center'][[0, 2]]
            path_points.append(toilet_center)
        
        if fixtures['tub']:
            tub_center = fixtures['tub'][0]['center'][[0, 2]]
            path_points.append(tub_center)
        elif fixtures['shower']:
            shower_center = fixtures['shower'][0]['center'][[0, 2]]
            path_points.append(shower_center)
        
        if len(path_points) < 3:
            return 0.5  # Partial path
        
        # Check path connectivity (simplified: check if path segments are clear)
        # For now, give bonus based on path length and fixture count
        bonus = len(path_points) / 4.0  # Max 4 points in path
        
        return bonus
    
    def _compute_wet_wall_bonus(self, fixtures: Dict, walls: List) -> float:
        """Compute bonus for minimizing wet walls (plumbing efficiency)."""
        if not walls:
            return 0.0
        
        # Count how many unique walls have fixtures attached
        wet_wall_count = 0
        threshold = 0.20  # 20cm to be considered "on" a wall
        
        for wall in walls:
            has_fixture = False
            for fixture_list in fixtures.values():
                for fix in fixture_list:
                    dist = self._compute_fixture_wall_distance(fix, wall)
                    if dist < threshold:
                        has_fixture = True
                        break
                if has_fixture:
                    break
            if has_fixture:
                wet_wall_count += 1
        
        # Bonus: fewer wet walls = higher bonus
        # 1 wall = 1.0, 2 walls = 0.5, 3+ walls = 0.0
        if wet_wall_count <= 1:
            return 1.0
        elif wet_wall_count == 2:
            return 0.5
        else:
            return 0.0
    
    def _compute_vanity_door_proximity(self, fixtures: Dict, doors: List) -> float:
        """Compute bonus for vanity being close to door (commonly used fixture)."""
        if not doors or not fixtures['vanity']:
            return 0.0
        
        door_center = doors[0]['center']
        vanity_center = fixtures['vanity'][0]['center']
        
        dist = self._distance_2d(door_center, vanity_center)
        
        # Closer is better, normalize by typical room size (~3m)
        max_dist = 3.0
        proximity = max(0, 1.0 - dist / max_dist)
        
        return proximity
    
    def _compute_vanity_toilet_proximity(self, fixtures: Dict) -> float:
        """Compute bonus for vanity being reasonably close to toilet."""
        if not fixtures['vanity'] or not fixtures['toilet']:
            return 0.0
        
        vanity_center = fixtures['vanity'][0]['center']
        toilet_center = fixtures['toilet'][0]['center']
        
        dist = self._distance_2d(vanity_center, toilet_center)
        
        # Optimal distance: 0.5m - 1.5m (not too close, not too far)
        if 0.5 <= dist <= 1.5:
            return 1.0
        elif dist < 0.5:
            return 0.5  # Too close
        else:
            return max(0, 1.0 - (dist - 1.5) / 1.5)
    
    def _compute_completeness_bonus(self, fixtures: Dict) -> float:
        """Compute bonus for having required fixtures (toilet + vanity mandatory)."""
        bonus = 0.0
        
        # Mandatory fixtures
        if fixtures['toilet']:
            bonus += 0.5
        if fixtures['vanity']:
            bonus += 0.5
        
        # Penalty for missing mandatory
        if not fixtures['toilet'] or not fixtures['vanity']:
            bonus -= 1.0
        
        return max(-1.0, bonus)


def select_top_k(
    scores: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Select top-K samples by score."""
    return np.argsort(scores)[-k:][::-1]  # Highest scores first


def score_and_select(
    translations: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    class_labels: np.ndarray,
    arch_translations: np.ndarray,
    arch_sizes: np.ndarray,
    arch_class_labels: np.ndarray,
    k: int = 5,
    weights: Optional[ScoringWeights] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score batch and select top-K samples.
    
    Args:
        translations: (B, N, 3) generated fixture positions
        sizes: (B, N, 3) generated fixture sizes
        angles: (B, N, 1) generated fixture rotations
        class_labels: (B, N, C) generated fixture classes (one-hot)
        arch_translations: (B, M, 3) architecture positions (walls, doors, floor)
        arch_sizes: (B, M, 3) architecture sizes
        arch_class_labels: (B, M, C) architecture classes
        k: number of top samples to select
        weights: optional custom scoring weights
    
    Returns:
        top_indices: (K,) indices of top-K samples
        top_scores: (K,) scores of top-K samples
    """
    scorer = BathroomSceneScorer(weights=weights)
    scores = scorer.score_batch(
        translations, sizes, angles, class_labels,
        arch_translations, arch_sizes, arch_class_labels
    )
    
    top_indices = select_top_k(scores, k)
    top_scores = scores[top_indices]
    
    return top_indices, top_scores


# ==================== INTEGRATION EXAMPLE ====================

def integrate_with_generation(
    network,
    room_mask: torch.Tensor,
    partial_boxes: torch.Tensor,
    config: dict,
    device: str,
    n_samples: int = 128,
    top_k: int = 5,
) -> Tuple[Dict, np.ndarray]:
    """
    Example integration with generation pipeline.
    
    Generates n_samples, scores them, returns top_k.
    """
    # Generate batch
    with torch.no_grad():
        room_mask_batch = room_mask.expand(n_samples, -1, -1, -1)
        partial_boxes_batch = partial_boxes.expand(n_samples, -1, -1)
        
        bbox_params = network.complete_scene(
            room_mask=room_mask_batch,
            num_points=config["network"]["sample_num_points"],
            point_dim=config["network"]["point_dim"],
            partial_boxes=partial_boxes_batch,
            batch_size=n_samples,
            device=device,
            keep_empty=True
        )
    
    # Extract architecture from partial boxes
    partial_num = config["network"].get("partial_num_points", 40)
    arch_trans = partial_boxes[0, :partial_num, :3].cpu().numpy()
    arch_sizes = partial_boxes[0, :partial_num, 3:6].cpu().numpy()
    arch_classes = partial_boxes[0, :partial_num, 8:].cpu().numpy()  # After angles
    
    # Repeat for batch
    B = n_samples
    arch_trans = np.tile(arch_trans[None], (B, 1, 1))
    arch_sizes = np.tile(arch_sizes[None], (B, 1, 1))
    arch_classes = np.tile(arch_classes[None], (B, 1, 1))
    
    # Extract generated fixtures
    translations = bbox_params[:, partial_num:, :3].cpu().numpy()
    sizes = bbox_params[:, partial_num:, 3:6].cpu().numpy()
    angles = bbox_params[:, partial_num:, 6:8].cpu().numpy()
    class_labels = bbox_params[:, partial_num:, 8:].cpu().numpy()
    
    # Score and select
    top_indices, top_scores = score_and_select(
        translations, sizes, angles, class_labels,
        arch_trans, arch_sizes, arch_classes,
        k=top_k
    )
    
    # Return top-k boxes
    top_boxes = {
        'translations': translations[top_indices],
        'sizes': sizes[top_indices],
        'angles': angles[top_indices],
        'class_labels': class_labels[top_indices],
        'scores': top_scores,
    }
    
    return top_boxes, top_indices


# ==================== VISUALIZATION ====================

import math
import re
import json
from pathlib import Path

# Color palette matching visualize_houzz_results.py
COLOR_PALETTE = {
    "void": [255, 255, 255],
    "floor": [211, 211, 211],
    "wall": [0, 0, 153],
    "door": [153, 0, 0],
    "window": [255, 153, 153],
    "toilet": [152, 223, 138],
    "vanity": [105, 183, 100],
    "tub": [44, 160, 44],
    "shower": [197, 176, 213],
    "clearance": [255, 240, 200],
    "empty": [255, 255, 255]
}


def rotate_point(x, z, yaw_rad):
    """Rotate a point around origin by yaw angle (in radians)"""
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x_rot = x * cos_yaw - z * sin_yaw
    z_rot = x * sin_yaw + z * cos_yaw
    return x_rot, z_rot


def get_bounding_box_corners(center, size, angle):
    """Get 2D corners of rotated bounding box."""
    cx, _, cz = center
    sx, _, sz = size
    half_sx, half_sz = sx / 2, sz / 2
    
    local_corners = [
        (-half_sx, -half_sz), (half_sx, -half_sz),
        (half_sx, half_sz), (-half_sx, half_sz)
    ]
    
    corners = []
    for lx, lz in local_corners:
        rx, rz = rotate_point(lx, lz, -angle)
        corners.append((cx + rx, cz + rz))
    return corners


def world_to_pixel(world_coords, bounds, img_size, unit_length=10):
    """Convert world coordinates to pixel coordinates."""
    min_x, max_x, min_z, max_z = bounds
    width, height = img_size
    scale = unit_length
    offset_x = (width - (max_x - min_x) * scale) / 2
    offset_y = (height - (max_z - min_z) * scale) / 2
    
    pixel_coords = []
    for x, z in world_coords:
        px = (x - min_x) * scale + offset_x
        py = (z - min_z) * scale + offset_y
        pixel_coords.append((px, py))
    return pixel_coords, scale


def compute_scene_bounds(translations, sizes, angles, class_labels):
    """Compute bounds of the scene."""
    if class_labels.ndim == 2:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels
    
    if angles.ndim == 2:
        if angles.shape[1] == 1:
            angles = angles.flatten()
        elif angles.shape[1] == 2:
            angles = np.arctan2(angles[:, 1], angles[:, 0])
    
    empty_idx = CLASS_LABELS.index('empty')
    valid_mask = cls_indices != empty_idx
    
    all_points = []
    for i in range(len(translations)):
        if not valid_mask[i]:
            continue
        corners = get_bounding_box_corners(translations[i], sizes[i], angles[i])
        all_points.extend(corners)
    
    if not all_points:
        return (-1, 1, -1, 1)
    
    xs = [p[0] for p in all_points]
    zs = [p[1] for p in all_points]
    return (min(xs), max(xs), min(zs), max(zs))


def visualize_scene_to_svg(translations, sizes, angles, class_labels, fixed_bounds=None):
    """Create SVG visualization from scene parameters."""
    if class_labels.ndim == 2:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()
    
    if angles.ndim == 2:
        if angles.shape[1] == 1:
            angles = angles.flatten()
        elif angles.shape[1] == 2:
            angles = np.arctan2(angles[:, 1], angles[:, 0])
    
    n_objs = len(cls_indices)
    empty_idx = CLASS_LABELS.index('empty')
    valid_mask = cls_indices != empty_idx
    
    if fixed_bounds is not None:
        bounds = fixed_bounds
    else:
        bounds = compute_scene_bounds(translations, sizes, angles, class_labels)
    
    W, H = 120, 120
    svg_lines = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{H}" fill="white"/>'
    ]
    
    indices = [i for i in range(n_objs) if valid_mask[i]]
    
    def get_priority(cls_idx):
        name = CLASS_LABELS[cls_idx]
        if name == 'floor': return 0
        if name in ['wall', 'window', 'door']: return 1
        return 2
    
    indices.sort(key=lambda i: get_priority(cls_indices[i]))
    
    for i in indices:
        cls_idx = cls_indices[i]
        label = CLASS_LABELS[cls_idx]
        color = COLOR_PALETTE.get(label, [0, 0, 0])
        rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
        
        t, s, a = translations[i], sizes[i], angles[i]
        if isinstance(a, np.ndarray) and a.size > 0:
            a = a.item()
        
        corners_world = get_bounding_box_corners(t, s, a)
        corners_px, _ = world_to_pixel(corners_world, bounds, (W, H), unit_length=10)
        pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in corners_px])
        
        svg_lines.append(
            f'<polygon points="{pts_str}" fill="{rgb_str}" stroke="{rgb_str}" data-type="{label}" />'
        )
    
    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


def create_comparison_grid_svg(grid_items: List[Tuple[str, str]], output_path: str):
    """
    Create a grid SVG comparing multiple scenes.
    
    Args:
        grid_items: List of (label, svg_inner_content) tuples
        output_path: Path to save the SVG
    """
    cols = 4
    cell_w, cell_h = 120, 120
    margin = 20
    label_h = 20
    
    num_items = len(grid_items)
    num_rows = math.ceil(num_items / cols)
    legend_h = 60
    
    total_w = cols * (cell_w + margin) + margin
    total_h = num_rows * (cell_h + margin + label_h) + margin + legend_h
    
    grid_svg = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>'
    ]
    
    for i, (label, content) in enumerate(grid_items):
        r = i // cols
        c = i % cols
        x = c * (cell_w + margin) + margin
        y = r * (cell_h + margin + label_h) + margin + label_h
        
        # Handle multi-line labels
        label_parts = label.split('\n')
        text_svg = ""
        # If multiple lines, stack them upwards from y=-5
        for line_idx, part in enumerate(reversed(label_parts)):
            y_offset = -5 - (line_idx * 12)
            text_svg += f'<text x="{cell_w/2}" y="{y_offset}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">{part}</text>\n'
        
        grid_svg.append(f'''<g transform="translate({x}, {y})">
            {text_svg}
            {content}
        </g>''')
    
    # Add Legend
    legend_y = num_rows * (cell_h + margin + label_h) + margin
    svg_legend = [f'<g transform="translate({margin}, {legend_y})">']
    svg_legend.append('<text x="0" y="0" font-family="Arial" font-size="10" font-weight="bold">Legend:</text>')
    
    item_per_row = 4
    rect_w, rect_h = 15, 10
    spacing_x, spacing_y = 120, 15
    
    legend_data = [(k, v) for k, v in COLOR_PALETTE.items() if k not in ["void", "empty"]]
    for i, (name, color) in enumerate(legend_data):
        row_idx = i // item_per_row
        col_idx = i % item_per_row
        lx = col_idx * spacing_x
        ly = 10 + row_idx * spacing_y
        rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
        svg_legend.append(f'''
            <rect x="{lx}" y="{ly}" width="{rect_w}" height="{rect_h}" fill="{rgb_str}" stroke="black" stroke-width="0.5"/>
            <text x="{lx + rect_w + 5}" y="{ly + 9}" font-family="Arial" font-size="9">{name}</text>
        ''')
    svg_legend.append('</g>')
    grid_svg.extend(svg_legend)
    
    grid_svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(grid_svg))


def extract_svg_inner(svg_content: str) -> str:
    """Extract inner content from SVG (without outer svg tags)."""
    match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
    return match.group(1) if match else svg_content


# ==================== MAIN CLI ====================

def load_samples_from_folder(folder_path: str) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """
    Load all boxes_*.npz samples from a folder.
    
    Returns:
        samples: List of dicts with translations, sizes, angles, class_labels
        bounds: Scene bounds computed from first sample (for consistent visualization)
    """
    folder = Path(folder_path)
    npz_files = sorted(folder.glob("boxes_*.npz"), key=lambda p: int(p.stem.split('_')[1]))
    
    if not npz_files:
        raise ValueError(f"No boxes_*.npz files found in {folder_path}")
    
    samples = []
    bounds = None
    
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        sample = {
            'translations': data['translations'],
            'sizes': data['sizes'],
            'angles': data['angles'],
            'class_labels': data['class_labels'],
            'path': str(npz_path),
        }
        samples.append(sample)
        
        # Compute bounds from first sample for consistent visualization
        if bounds is None:
            bounds = compute_scene_bounds(
                sample['translations'], sample['sizes'],
                sample['angles'], sample['class_labels']
            )
    
    return samples, bounds


def get_bounds_from_json(json_path: str) -> Optional[Tuple[float, float, float, float]]:
    """Get scene bounds from original JSON file."""
    try:
        with open(json_path, 'r') as f:
            scene_data = json.load(f)
        
        json_items = scene_data.get('items', scene_data)
        all_points = []
        
        for item in json_items:
            cx = item.get('center_x', 0)
            cz = item.get('center_z', 0)
            sx = item.get('size_x', 0)
            sz = item.get('size_z', 0)
            y_rad = math.radians(item.get('yaw', 0))
            
            h_sx, h_sz = sx / 2, sz / 2
            for lx, lz in [(-h_sx, -h_sz), (h_sx, -h_sz), (h_sx, h_sz), (-h_sx, h_sz)]:
                rx = lx * math.cos(y_rad) - lz * math.sin(y_rad)
                rz = lx * math.sin(y_rad) + lz * math.cos(y_rad)
                all_points.append((cx + rx, cz + rz))
        
        if all_points:
            xs = [p[0] for p in all_points]
            zs = [p[1] for p in all_points]
            return (min(xs), max(xs), min(zs), max(zs))
    except Exception:
        pass
    
    return None


def score_folder(
    folder_path: str,
    top_k: int = 5,
    partial_num_points: int = 40,
    weights: Optional[ScoringWeights] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Score all samples in a folder and return top-K indices.
    
    Args:
        folder_path: Path to folder containing boxes_*.npz files
        top_k: Number of top samples to select
        partial_num_points: Number of architecture points (first N points)
        weights: Optional scoring weights
    
    Returns:
        top_indices: Indices of top-K samples (sorted by score, highest first)
        all_scores: Scores for all samples
        samples: List of all loaded sample dicts
    """
    samples, bounds = load_samples_from_folder(folder_path)
    print(f"Loaded {len(samples)} samples from {folder_path}")
    
    scorer = BathroomSceneScorer(weights=weights)
    all_scores = np.zeros(len(samples), dtype=np.float32)
    
    for i, sample in enumerate(samples):
        trans = sample['translations']
        sizes = sample['sizes']
        angles = sample['angles']
        class_labels = sample['class_labels']
        
        # Split into architecture (first partial_num_points) and fixtures (rest)
        arch_trans = trans[:partial_num_points]
        arch_sizes = sizes[:partial_num_points]
        arch_classes = class_labels[:partial_num_points]
        
        fix_trans = trans[partial_num_points:]
        fix_sizes = sizes[partial_num_points:]
        fix_angles = angles[partial_num_points:]
        fix_classes = class_labels[partial_num_points:]
        
        # Score this sample
        score = scorer._score_single(
            fix_trans, fix_sizes, fix_angles, fix_classes,
            arch_trans, arch_sizes, arch_classes
        )
        all_scores[i] = score
    
    # Get top-K indices
    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    
    return top_indices, all_scores, samples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Score bathroom scene samples and select top-K candidates"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing boxes_*.npz files (e.g., 214503916_Bath_US_simple_design_filtered)"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=5,
        help="Number of top samples to select (default: 5)"
    )
    parser.add_argument(
        "--partial_num_points",
        type=int,
        default=40,
        help="Number of architecture points (first N points in each sample)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Default: best_<scene_id>.svg in the input folder"
    )
    
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder {folder} does not exist")
        return
    
    # Extract scene ID from folder name
    scene_id = folder.name
    if "_simple_design_filtered" in scene_id:
        scene_id_short = scene_id.replace("_simple_design_filtered", "").split("_")[0]
    else:
        scene_id_short = scene_id.split("_")[0]
    
    # Score samples
    print(f"\n{'='*60}")
    print(f"Scoring samples in: {folder}")
    print(f"{'='*60}")
    
    top_indices, all_scores, samples = score_folder(
        str(folder),
        top_k=args.top_k,
        partial_num_points=args.partial_num_points
    )
    
    print(f"\nScore statistics:")
    print(f"  Min: {all_scores.min():.2f}, Max: {all_scores.max():.2f}")
    print(f"  Mean: {all_scores.mean():.2f}, Std: {all_scores.std():.2f}")
    
    print(f"\nTop {args.top_k} samples:")
    for rank, idx in enumerate(top_indices):
        print(f"  #{rank+1}: Sample {idx} (score: {all_scores[idx]:.2f})")
    
    # Get scene bounds for consistent visualization
    json_path = folder / "simple_design_filtered.json"
    bounds = get_bounds_from_json(str(json_path))
    if bounds is None:
        # Use bounds from first sample
        s = samples[0]
        bounds = compute_scene_bounds(s['translations'], s['sizes'], s['angles'], s['class_labels'])
    
    # Create grid items
    grid_items = []
    
    # 1. Add original label (ground truth) first
    label_svg_path = folder / "simple_design_label.svg"
    if label_svg_path.exists():
        with open(label_svg_path, 'r') as f:
            label_svg = f.read()
        grid_items.append(("GT Label", extract_svg_inner(label_svg)))
    
    # 2. Add top-K samples
    for rank, idx in enumerate(top_indices):
        sample = samples[idx]
        svg_content = visualize_scene_to_svg(
            sample['translations'],
            sample['sizes'],
            sample['angles'],
            sample['class_labels'],
            fixed_bounds=bounds
        )
        score = all_scores[idx]
        grid_items.append((f"#{rank+1} (S{idx}, {score:.1f})", extract_svg_inner(svg_content)))
    
    # 3. Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = folder / f"best_{scene_id_short}.svg"
    
    # 4. Create and save grid SVG
    create_comparison_grid_svg(grid_items, str(output_path))
    print(f"\nSaved visualization to: {output_path}")
    
    # 5. Also save a JSON with scores for later analysis
    scores_json_path = folder / f"scores_{scene_id_short}.json"
    scores_data = {
        'scene_id': scene_id,
        'num_samples': len(samples),
        'top_k': args.top_k,
        'top_indices': top_indices.tolist(),
        'top_scores': all_scores[top_indices].tolist(),
        'all_scores': all_scores.tolist(),
        'score_stats': {
            'min': float(all_scores.min()),
            'max': float(all_scores.max()),
            'mean': float(all_scores.mean()),
            'std': float(all_scores.std()),
        }
    }
    with open(scores_json_path, 'w') as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved scores to: {scores_json_path}")


if __name__ == "__main__":
    main()

