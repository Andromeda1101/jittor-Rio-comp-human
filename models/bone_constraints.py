import jittor as jt
import jittor.nn as nn
import numpy as np
from dataset.format import parents

class BoneConstraints:
    def __init__(self):
        # 预定义的骨骼长度范围 (min_length, max_length)
        self.bone_length_ranges = {
            (0, 1): (0.15, 0.25),  # hips -> spine
            (1, 2): (0.10, 0.20),  # spine -> chest
            (2, 3): (0.10, 0.20),  # chest -> upper_chest
            (3, 4): (0.05, 0.15),  # upper_chest -> neck
            (4, 5): (0.08, 0.15),  # neck -> head
            # 左臂
            (3, 6): (0.10, 0.20),  # upper_chest -> l_shoulder
            (6, 7): (0.20, 0.35),  # l_shoulder -> l_upper_arm
            (7, 8): (0.20, 0.35),  # l_upper_arm -> l_lower_arm
            (8, 9): (0.10, 0.20),  # l_lower_arm -> l_hand
            # 右臂
            (3, 10): (0.10, 0.20), # upper_chest -> r_shoulder
            (10, 11): (0.20, 0.35), # r_shoulder -> r_upper_arm
            (11, 12): (0.20, 0.35), # r_upper_arm -> r_lower_arm
            (12, 13): (0.10, 0.20), # r_lower_arm -> r_hand
            # 左腿
            (0, 14): (0.10, 0.20), # hips -> l_upper_leg
            (14, 15): (0.35, 0.50), # l_upper_leg -> l_lower_leg
            (15, 16): (0.35, 0.50), # l_lower_leg -> l_foot
            (16, 17): (0.05, 0.15), # l_foot -> l_toe_base
            # 右腿
            (0, 18): (0.10, 0.20), # hips -> r_upper_leg
            (18, 19): (0.35, 0.50), # r_upper_leg -> r_lower_leg
            (19, 20): (0.35, 0.50), # r_lower_leg -> r_foot
            (20, 21): (0.05, 0.15), # r_foot -> r_toe_base
        }

        # 预定义的关节角度范围 (min_angle, max_angle) 弧度制
        self.joint_angle_ranges = {
            1: (-0.5, 0.5),    # spine 
            2: (-0.5, 0.5),    # chest
            3: (-0.3, 0.3),    # upper_chest
            4: (-0.7, 0.7),    # neck
            5: (-0.7, 0.7),    # head
            # 左臂
            6: (-np.pi, np.pi), # l_shoulder
            7: (-np.pi, np.pi), # l_upper_arm
            8: (0, np.pi),      # l_lower_arm
            9: (-0.7, 0.7),     # l_hand
            # 右臂
            10: (-np.pi, np.pi), # r_shoulder
            11: (-np.pi, np.pi), # r_upper_arm
            12: (0, np.pi),      # r_lower_arm
            13: (-0.7, 0.7),     # r_hand
            # 左腿
            14: (-np.pi/2, np.pi/2), # l_upper_leg
            15: (0, np.pi),          # l_lower_leg
            16: (-0.7, 0.7),         # l_foot
            17: (-0.4, 0.4),         # l_toe
            # 右腿
            18: (-np.pi/2, np.pi/2), # r_upper_leg
            19: (0, np.pi),          # r_lower_leg
            20: (-0.7, 0.7),         # r_foot
            21: (-0.4, 0.4),         # r_toe
        }

    def compute_bone_lengths(self, joints):
        """计算所有骨骼长度"""
        B = joints.shape[0]
        bone_lengths = {}
        for child, parent in enumerate(parents):
            if parent is not None:
                bone = (parent, child)
                bone_vec = joints[:, child] - joints[:, parent]
                bone_lengths[bone] = jt.norm(bone_vec, dim=-1)  # [B]
        return bone_lengths

    def compute_joint_angles(self, joints):
        """计算所有关节角度"""
        B = joints.shape[0]
        joint_angles = {}
        
        for joint_id, parent_id in enumerate(parents):
            if parent_id is not None:
                # 获取当前骨骼的方向向量
                current_vec = joints[:, joint_id] - joints[:, parent_id]
                current_vec = current_vec / (jt.norm(current_vec, dim=-1, keepdim=True) + 1e-6)
                
                # 获取父节点的父节点
                grand_parent_id = parents[parent_id]
                if grand_parent_id is not None:
                    # 计算父骨骼的方向向量
                    parent_vec = joints[:, parent_id] - joints[:, grand_parent_id]
                    parent_vec = parent_vec / (jt.norm(parent_vec, dim=-1, keepdim=True) + 1e-6)
                    
                    # 计算角度
                    cos_angle = jt.sum(current_vec * parent_vec, dim=-1)
                    cos_angle = jt.clamp(cos_angle, -0.999999, 0.999999)
                    angle = jt.arccos(cos_angle)
                    joint_angles[joint_id] = angle  # [B]
                
        return joint_angles

    def compute_constraint_loss(self, joints):
        """计算约束损失"""
        bone_lengths = self.compute_bone_lengths(joints)
        joint_angles = self.compute_joint_angles(joints)
        
        length_loss = jt.array(0.0)
        angle_loss = jt.array(0.0)
        
        # 骨骼长度约束
        for bone, lengths in bone_lengths.items():
            min_len, max_len = self.bone_length_ranges[bone]
            length_loss = length_loss + jt.mean(nn.relu(min_len - lengths) + nn.relu(lengths - max_len))
        
        # 关节角度约束
        for joint_id, angles in joint_angles.items():
            if joint_id in self.joint_angle_ranges:
                min_angle, max_angle = self.joint_angle_ranges[joint_id]
                angle_loss = angle_loss + jt.mean(nn.relu(min_angle - angles) + nn.relu(angles - max_angle))
        
        return length_loss + 0.5 * angle_loss
