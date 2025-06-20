import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
from PCT.networks.cls.unifiedpct import UnifiedPointTransformer

class UnifiedModel(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 共享的Transformer骨干网络
        self.transformer = UnifiedPointTransformer(output_channels=feat_dim)
        
        # 骨骼预测分支
        self.joint_predictor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_joints * 3)
        )
        
        # 蒙皮特征提取
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        
        # 骨骼约束模块
        self.joint_constraint = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim)
        )
        
        # 交叉注意力机制
        self.cross_att = MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        
        # 蒙皮预测分支
        self.skin_predictor = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_joints)
        )
    
    def execute(self, vertices):
        # 共享特征提取 (B, 3, N) -> (B, feat_dim)
        shape_latent = self.transformer(vertices)
        
        # 骨骼预测
        joint_pred = self.joint_predictor(shape_latent).view(-1, self.num_joints, 3)
        
        # 蒙皮特征准备
        B, _, N = vertices.shape
        vertices_flat = vertices.permute(0, 2, 1).reshape(B * N, 3)
        shape_latent_exp = shape_latent.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
        
        # 顶点特征增强
        vertex_feat = self.vertex_encoder(concat([vertices_flat, shape_latent_exp], dim=1))
        vertex_feat = vertex_feat.view(B, N, -1)
        
        # 骨骼约束特征
        joint_feat = self.joint_constraint(joint_pred.view(B * self.num_joints, 3))
        joint_feat = joint_feat.view(B, self.num_joints, -1)
        
        # 交叉注意力
        att_output, _ = self.cross_att(
            vertex_feat, 
            joint_feat,
            joint_feat
        )
        
        # 骨骼约束融合
        constrained_feat = concat([vertex_feat, att_output], dim=-1)
        
        # 蒙皮权重预测
        skin_logits = self.skin_predictor(constrained_feat)
        
        # 骨骼距离约束
        vertices_3d = vertices.permute(0, 2, 1)  # (B, N, 3)
        dist = jt.norm(
            vertices_3d.unsqueeze(2) - joint_pred.unsqueeze(1),
            dim=-1
        )  # (B, N, J)
        
        # 应用距离约束
        skin_weights = nn.softmax(skin_logits + 1.0/(dist + 1e-6), dim=-1)
        
        return joint_pred, skin_weights