import jittor as jt
from jittor import nn

class SkinConstraints:
    def __init__(self):
        # 定义蒙皮权重的有效半径
        self.radius_scales = {
            'spine': 0.15,    # 脊柱链的影响半径较大
            'limbs': 0.12,    # 四肢的影响半径适中
            'extremities': 0.08  # 末端(手、脚)影响半径较小
        }
        
        # 将关节分组
        self.joint_groups = {
            'spine': [0,1,2,3,4,5],  # 脊柱链
            'limbs': [6,7,8,10,11,12,14,15,18,19],  # 四肢主要部分
            'extremities': [9,13,16,17,20,21]  # 末端
        }
    
    def _get_joint_radius(self, joint_idx):
        """根据关节类型返回影响半径"""
        for group, radius in self.radius_scales.items():
            if joint_idx in self.joint_groups[group]:
                return radius
        return self.radius_scales['limbs']  # 默认返回四肢半径

    def compute_distance_weights(self, vertices, joints):
        """计算基于距离的权重矩阵"""
        B, N, _ = vertices.shape
        _, J, _ = joints.shape
        
        # 计算每个顶点到每个关节的距离
        dist = jt.norm(vertices.unsqueeze(2) - joints.unsqueeze(1), dim=-1)  # [B, N, J]
        
        # 为每个关节计算自适应的权重
        weights = []
        for j in range(J):
            radius = self._get_joint_radius(j)
            # 使用高斯核计算权重
            w = jt.exp(-dist[..., j] ** 2 / (2 * radius ** 2))
            weights.append(w)
        
        weights = jt.stack(weights, dim=-1)  # [B, N, J]
        return weights

    def compute_locality_regularization(self, skin_weights):
        """计算局部性正则化损失"""
        # 确保每个顶点主要受到少数关节的影响
        topk_values, _ = jt.topk(skin_weights, k=4, dim=-1)
        locality_loss = jt.mean(skin_weights) - jt.mean(topk_values)
        return locality_loss

    def compute_smoothness_loss(self, skin_weights, vertices):
        """计算空间平滑度损失"""
        B, N, J = skin_weights.shape
        
        # 将vertices调整为正确的形状 (B, N, 3)
        if vertices.shape[1] == 3:
            vertices = vertices.permute(0, 2, 1)
        
        # 计算顶点之间的距离矩阵
        vertex_dist = jt.norm(vertices.unsqueeze(2) - vertices.unsqueeze(1), dim=-1)  # [B, N, N]
        
        # 找到每个顶点的最近邻
        _, nn_idx = jt.topk(-vertex_dist, k=6, dim=2)  # [B, N, 6]
        
        # 为每个顶点收集其邻居的权重
        batch_idx = jt.arange(B).reshape(B, 1, 1).repeat(1, N, 6)  # [B, N, 6]
        vertex_idx = nn_idx  # [B, N, 6]
        
        # 重构索引张量
        gather_idx = jt.stack([
            batch_idx.reshape(-1),  # 批次索引
            vertex_idx.reshape(-1)  # 顶点索引
        ], dim=0)  # [2, B*N*6]
        
        # 收集邻居权重
        skin_weights_flat = skin_weights.reshape(B*N, J)  # [B*N, J]
        nn_weights = jt.gather(skin_weights_flat, dim=0, index=gather_idx[1])  # [B*N*6, J]
        nn_weights = nn_weights.reshape(B, N, 6, J)  # [B, N, 6, J]
        
        # 计算中心顶点与邻居之间的权重差异
        weight_diff = (skin_weights.unsqueeze(2) - nn_weights).abs()  # [B, N, 6, J]
        
        # 基于距离的权重
        dist_weight = jt.exp(-jt.gather(vertex_dist, dim=2, index=nn_idx))  # [B, N, 6]
        dist_weight = dist_weight.unsqueeze(-1)  # [B, N, 6, 1]
        
        # 计算加权平均损失
        smoothness_loss = (weight_diff * dist_weight).mean()
        
        return smoothness_loss

    def compute_constraint_loss(self, skin_weights, vertices, joints):
        """计算总的约束损失"""
        # 计算基于距离的目标权重
        target_weights = self.compute_distance_weights(vertices, joints)
        
        # 计算与目标权重的差异
        weight_diff_loss = nn.mse_loss(skin_weights, target_weights)
        
        # 局部性约束
        locality_loss = self.compute_locality_regularization(skin_weights)
        
        # 平滑度约束
        smoothness_loss = self.compute_smoothness_loss(skin_weights, vertices)
        
        # 非负性和归一化约束
        non_negative_loss = nn.relu(-skin_weights).mean()
        sum_to_one_loss = (jt.sum(skin_weights, dim=-1) - 1.0).abs().mean()
        
        total_loss = (weight_diff_loss + 
                     0.1 * locality_loss + 
                     0.1 * smoothness_loss + 
                     10.0 * non_negative_loss + 
                     1.0 * sum_to_one_loss)
        
        return total_loss
