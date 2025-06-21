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
        
        # 修改：重新定义骨骼链权重，使用完整的链名称作为键
        self.chain_weights = {
            'spine': 2.0,    # 脊柱链
            'l_arm': 1.5,    # 左臂
            'r_arm': 1.5,    # 右臂
            'l_leg': 1.5,    # 左腿
            'r_leg': 1.5     # 右腿
        }
        
        # 骨骼链定义保持不变
        self.bone_chains = {
            'spine': [(0,1), (1,2), (2,3), (3,4), (4,5)],
            'l_arm': [(3,6), (6,7), (7,8), (8,9)],
            'r_arm': [(3,10), (10,11), (11,12), (12,13)],
            'l_leg': [(0,14), (14,15), (15,16), (16,17)],
            'r_leg': [(0,18), (18,19), (19,20), (20,21)]
        }

        # 验证骨骼链配置
        for chain_name in self.bone_chains.keys():
            if chain_name not in self.chain_weights:
                raise KeyError(f"Weight for chain '{chain_name}' not defined in chain_weights")

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

    def compute_chain_consistency(self, joints):
        """计算骨骼链的一致性约束"""
        chain_loss = jt.array(0.0)
        
        for chain_name, bones in self.bone_chains.items():
            chain_dir = None
            prev_dir = None
            
            for parent, child in bones:
                curr_dir = joints[:, child] - joints[:, parent]
                curr_dir = curr_dir / (jt.norm(curr_dir, dim=-1, keepdim=True) + 1e-6)
                
                if prev_dir is not None:
                    # 计算相邻骨骼方向的一致性
                    cos_sim = jt.sum(prev_dir * curr_dir, dim=-1)
                    # 确定目标余弦值
                    target = 0.7 if 'arm' in chain_name or 'leg' in chain_name else 0.9
                    # 使用完整的chain_name作为键
                    chain_loss = chain_loss + self.chain_weights[chain_name] * jt.mean((cos_sim - target) ** 2)
                
                prev_dir = curr_dir
        
        return chain_loss

    def compute_constraint_loss(self, joints):
        """计算约束损失"""
        bone_lengths = self.compute_bone_lengths(joints)
        joint_angles = self.compute_joint_angles(joints)
        chain_loss = self.compute_chain_consistency(joints)
        
        length_loss = jt.array(0.0)
        angle_loss = jt.array(0.0)
        
        # 骨骼长度约束（使用smooth L1 loss）
        for bone, lengths in bone_lengths.items():
            min_len, max_len = self.bone_length_ranges[bone]
            center_len = (min_len + max_len) / 2
            length_diff = jt.abs(lengths - center_len)
            length_loss = length_loss + jt.mean(
                jt.where(length_diff < 1.0,
                        0.5 * length_diff ** 2,
                        length_diff - 0.5)
            )
        
        # 关节角度约束（使用带margin的hinge loss）
        for joint_id, angles in joint_angles.items():
            if joint_id in self.joint_angle_ranges:
                min_angle, max_angle = self.joint_angle_ranges[joint_id]
                margin = 0.1  # 添加余量
                angle_loss = angle_loss + jt.mean(
                    nn.relu(angles - (max_angle - margin)) +
                    nn.relu((min_angle + margin) - angles)
                )
        
        # 返回加权组合的损失
        return length_loss + 2.0 * angle_loss + 1.5 * chain_loss
