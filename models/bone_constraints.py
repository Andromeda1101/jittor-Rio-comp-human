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

        # 更新关节角度范围，移除负值范围
        self.joint_angle_ranges = {
            # 脊柱链保持近乎直立
            1: (0.0, 0.15),    # spine
            2: (0.0, 0.15),    # chest 
            3: (0.0, 0.15),     # upper_chest
            4: (0.0, 0.2),     # neck
            5: (0.0, 0.2),     # head
            
            # 左右肩膀 - 限制在合适的角度范围
            6: (0.3, 1.0),     # l_shoulder (向外上方的角度)
            10: (0.3, 1.0),    # r_shoulder (对称角度)
            
            # 手臂 - 与脊柱保持合适角度
            7: (1.0, 1.4),     # l_upper_arm 
            8: (0.0, 0.2),     # l_lower_arm (保持近直线)
            11: (1.0, 1.4),    # r_upper_arm 
            12: (0.0, 0.2),    # r_lower_arm (保持近直线)
            
            # 手腕 - 限制保持近直线
            9: (0.0, 0.15),    # l_hand
            13: (0.0, 0.15),   # r_hand
            
            # 髋部 - 限制在合适的角度范围
            14: (1.0, 1.4),    # l_upper_leg
            18: (1.0, 1.4),    # r_upper_leg
            
            # 腿部 - 保持近似平行
            15: (0.0, 0.2),    # l_lower_leg
            19: (0.0, 0.2),    # r_lower_leg
            
            # 脚部 - 允许适度弯曲
            16: (0.0, 0.3),    # l_foot
            20: (0.0, 0.3),    # r_foot
            17: (0.5, 1.2),   # l_toe
            21: (0.5, 1.2),   # r_toe
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

        # 添加对称关节对
        self.symmetric_pairs = [
            (6, 10),   # 左右肩膀
            (7, 11),   # 左右上臂
            (8, 12),   # 左右前臂
            (9, 13),   # 左右手
            (14, 18),  # 左右大腿
            (15, 19),  # 左右小腿
            (16, 20),  # 左右脚
            (17, 21),  # 左右脚趾
        ]

        # 添加共线约束的关节链
        self.collinear_chains = [
            [0, 1, 2, 3, 4, 5],  # 脊柱链
            [3, 6, 7, 8, 9],     # 左臂
            [3, 10, 11, 12, 13], # 右臂
            [0, 14, 15, 16],     # 左腿
            [0, 18, 19, 20],     # 右腿
        ]

        # 添加需要进行平面性检查的关节组（排除脚趾）
        self.planarity_joints = [
            0, 1, 2, 3, 4, 5,     # 脊柱链
            6, 7, 8, 9,           # 左臂
            10, 11, 12, 13,       # 右臂
            14, 15, 16,           # 左腿（不含脚趾）
            18, 19, 20,           # 右腿（不含脚趾）
        ]
        
        # 定义平面性和对称性权重
        self.planarity_weight = 0.8
        self.symmetry_tolerance = 0.1  # 对称性容差

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

    def compute_symmetry_loss(self, joints):
        """计算改进的对称性损失，允许一定的不对称"""
        symmetry_loss = 0.0
        
        # 计算躯干中线（从臀部到头部的线）作为对称轴
        spine_vector = joints[:, 5] - joints[:, 0]  # 头部到臀部
        spine_direction = spine_vector / (jt.norm(spine_vector, dim=-1, keepdim=True) + 1e-6)
        
        for left_id, right_id in self.symmetric_pairs:
            left_vec = joints[:, left_id]
            right_vec = joints[:, right_id]
            
            # 计算到脊柱中线的距离
            left_to_spine = self._point_to_line_distance(left_vec, joints[:, 0], spine_direction)
            right_to_spine = self._point_to_line_distance(right_vec, joints[:, 0], spine_direction)
            
            # 计算距离差异，但允许一定的容差
            distance_diff = jt.abs(left_to_spine - right_to_spine)
            soft_diff = jt.where(
                distance_diff > self.symmetry_tolerance,
                (distance_diff - self.symmetry_tolerance) ** 2,
                jt.zeros_like(distance_diff)
            )
            
            symmetry_loss = symmetry_loss + jt.mean(soft_diff)
            
            # 检查高度（Y轴）对称性
            height_diff = jt.abs(left_vec[:, 1] - right_vec[:, 1])
            soft_height_diff = jt.where(
                height_diff > self.symmetry_tolerance,
                (height_diff - self.symmetry_tolerance) ** 2,
                jt.zeros_like(height_diff)
            )
            
            symmetry_loss = symmetry_loss + 0.5 * jt.mean(soft_height_diff)
            
        return symmetry_loss

    def _point_to_line_distance(self, point, line_point, line_direction):
        """计算点到直线的距离"""
        # 向量从线上的点指向目标点
        point_vector = point - line_point
        
        # 计算点到直线的垂直分量
        projection = point_vector - jt.sum(point_vector * line_direction, dim=-1, keepdim=True) * line_direction
        
        return jt.norm(projection, dim=-1)

    def compute_collinear_loss(self, joints):
        """计算共线性损失"""
        collinear_loss = 0.0
        for chain in self.collinear_chains:
            for i in range(len(chain)-2):
                # 获取连续的三个关节点
                p1 = joints[:, chain[i]]
                p2 = joints[:, chain[i+1]]
                p3 = joints[:, chain[i+2]]
                
                # 计算两个向量
                v1 = p2 - p1
                v2 = p3 - p2
                
                # 标准化向量
                v1 = v1 / (jt.norm(v1, dim=-1, keepdim=True) + 1e-6)
                v2 = v2 / (jt.norm(v2, dim=-1, keepdim=True) + 1e-6)
                
                # 计算向量夹角的余弦值，应该接近1
                cos_angle = jt.sum(v1 * v2, dim=-1)
                collinear_loss = collinear_loss + jt.mean((1 - cos_angle) ** 2)
        
        return collinear_loss

    def compute_perpendicular_loss(self, joints):
        """计算垂直性损失 - 确保手臂与脊柱近似垂直"""
        loss = 0.0
        
        # 计算脊柱方向 (使用upper_chest到spine的向量)
        spine_dir = joints[:, 3] - joints[:, 1]  # upper_chest to spine
        spine_dir = spine_dir / (jt.norm(spine_dir, dim=-1, keepdim=True) + 1e-6)
        
        # 计算左右上臂方向
        l_arm_dir = joints[:, 7] - joints[:, 6]  # l_upper_arm方向
        r_arm_dir = joints[:, 11] - joints[:, 10]  # r_upper_arm方向
        
        l_arm_dir = l_arm_dir / (jt.norm(l_arm_dir, dim=-1, keepdim=True) + 1e-6)
        r_arm_dir = r_arm_dir / (jt.norm(r_arm_dir, dim=-1, keepdim=True) + 1e-6)
        
        # 计算点积的绝对值，应接近0（垂直）
        l_perp = jt.abs(jt.sum(spine_dir * l_arm_dir, dim=-1))
        r_perp = jt.abs(jt.sum(spine_dir * r_arm_dir, dim=-1))
        
        loss = jt.mean(l_perp ** 2 + r_perp ** 2)
        
        return loss

    def compute_arm_alignment_loss(self, joints):
        """计算手臂各关节的对齐损失 - 确保上臂、前臂和手保持直线"""
        loss = 0.0
        
        # 左臂对齐
        l_upper = joints[:, 7] - joints[:, 6]    # 左上臂
        l_lower = joints[:, 8] - joints[:, 7]    # 左前臂
        l_hand = joints[:, 9] - joints[:, 8]     # 左手
        
        # 右臂对齐
        r_upper = joints[:, 11] - joints[:, 10]  # 右上臂
        r_lower = joints[:, 12] - joints[:, 11]  # 右前臂
        r_hand = joints[:, 13] - joints[:, 12]   # 右手
        
        # 标准化向量
        vectors = [l_upper, l_lower, l_hand, r_upper, r_lower, r_hand]
        vectors = [v / (jt.norm(v, dim=-1, keepdim=True) + 1e-6) for v in vectors]
        
        # 计算相邻向量的对齐损失
        for i in range(0, len(vectors), 3):
            v1, v2, v3 = vectors[i:i+3]
            # 使用点积计算对齐程度，应接近1（同向）
            align1 = jt.sum(v1 * v2, dim=-1)
            align2 = jt.sum(v2 * v3, dim=-1)
            loss = loss + jt.mean((1 - align1) ** 2 + (1 - align2) ** 2)
        
        return loss

    def compute_planarity_loss(self, joints):
        """计算骨架的平面性损失（除脚趾外）"""
        # 选择参与平面性计算的关节
        planar_joints = joints[:, self.planarity_joints]
        B = joints.shape[0]
        
        # 对每个批次单独计算
        total_loss = 0
        for i in range(B):
            # 计算最佳拟合平面的法向量
            centered = planar_joints[i] - jt.mean(planar_joints[i], dim=0, keepdim=True)
            _, _, vh = jt.svd(centered)
            normal = vh[-1]  # 最小特征值对应的向量为法向量
            
            # 计算每个点到平面的距离
            distances = jt.abs(jt.matmul(centered, normal))
            
            # 使用软约束：允许少量偏离平面
            soft_distances = jt.where(
                distances > 0.05,  # 允许5cm的偏离
                (distances - 0.05) ** 2,
                jt.zeros_like(distances)
            )
            
            total_loss += jt.mean(soft_distances)
            
        return total_loss / B

    def compute_constraint_loss(self, joints):
        """更新后的总约束损失计算"""
        # 原有的约束损失计算
        bone_lengths = self.compute_bone_lengths(joints)
        joint_angles = self.compute_joint_angles(joints)
        chain_loss = self.compute_chain_consistency(joints)
        
        length_loss = jt.array(0.0)
        angle_loss = jt.array(0.0)
        
        # 骨骼长度约束
        for bone, lengths in bone_lengths.items():
            min_len, max_len = self.bone_length_ranges[bone]
            length_diff = jt.clamp(lengths - max_len, min_v=0) + jt.clamp(min_len - lengths, min_v=0)
            length_loss = length_loss + length_diff.mean()
        
        # 关节角度约束
        for joint_id, angles in joint_angles.items():
            if joint_id in self.joint_angle_ranges:
                min_angle, max_angle = self.joint_angle_ranges[joint_id]
                angle_diff = jt.clamp(angles - max_angle, min_v=0) + jt.clamp(min_angle - angles, min_v=0)
                angle_loss = angle_loss + angle_diff.mean()
        
        # 计算对称性和共线性损失
        symmetry_loss = self.compute_symmetry_loss(joints)
        collinear_loss = self.compute_collinear_loss(joints)
        
        # 添加新的约束损失
        perpendicular_loss = self.compute_perpendicular_loss(joints)
        arm_alignment_loss = self.compute_arm_alignment_loss(joints)
        planarity_loss = self.compute_planarity_loss(joints)
        
        # 更新总损失计算，增加新约束的权重
        total_loss = (length_loss + 
                     2.5 * angle_loss +          # 增加角度约束权重
                     1.5 * chain_loss + 
                     2.0 * symmetry_loss + 
                     1.0 * collinear_loss +
                     2.0 * perpendicular_loss +   # 手臂垂直约束
                     1.5 * arm_alignment_loss +   # 手臂对齐约束
                     self.planarity_weight * planarity_loss)  # 添加平面性约束
        
        return total_loss
