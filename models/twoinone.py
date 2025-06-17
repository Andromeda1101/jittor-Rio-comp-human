import jittor as jt
from jittor import nn
from jittor.contrib import concat
from math import sqrt
from PCT.networks.cls.enhancedpct import EnhancedPointTransformer

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def execute(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out += identity
        return out

class JointSkinModel(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22, attn_heads=8, attn_layers=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints

        # 主干特征提取
        self.transformer = EnhancedPointTransformer(output_channels=feat_dim)

        # joints分支
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.joint_mlp = nn.Sequential(
            ResidualMLP(feat_dim, 512, 512),
            nn.GELU(),
            nn.Linear(512, num_joints * 3)
        )

        # skin分支
        self.vertex_encoder = ResidualMLP(3 + feat_dim, 512, feat_dim)
        self.joint_encoder = ResidualMLP(3 + feat_dim, 512, feat_dim)

        # 多层交互注意力
        self.cross_att_layers = nn.ModuleList([
            nn.MultiheadAttention(feat_dim, attn_heads, batch_first=True)
            for _ in range(attn_layers)
        ])

        # 空间关系编码
        self.dist_encoder = nn.Sequential(
            nn.Linear(4, 64),  # (距离, 方向向量3维)
            nn.GELU(),
            nn.Linear(64, feat_dim)
        )

        # 输出层
        self.output = nn.Sequential(
            ResidualMLP(feat_dim, 256, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, num_joints)
        )

    def execute(self, vertices):
        # vertices: [B, N, 3]
        B, N, _ = vertices.shape

        # 1. 主干特征
        x = self.transformer(vertices.permute(0, 2, 1))  # [B, feat_dim, N]

        # 2. joints预测
        x_global = self.global_pool(x)  # [B, feat_dim, 1]
        x_global = x_global.squeeze(-1) # [B, feat_dim]
        joints_pred = self.joint_mlp(x_global).view(B, self.num_joints, 3)  # [B, J, 3]

        # 3. skin权重预测
        # 点特征
        x_points = x.transpose(0, 2, 1)  # [B, N, feat_dim]
        vertex_feat = concat([vertices, x_points], dim=-1)  # [B, N, 3+feat_dim]
        vertex_feat = self.vertex_encoder(vertex_feat)       # [B, N, feat_dim]

        # joints特征
        joint_feat = concat([joints_pred, x_global.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1)  # [B, J, 3+feat_dim]
        joint_feat = self.joint_encoder(joint_feat)  # [B, J, feat_dim]

        # 空间关系编码
        # 计算每个点到每个joint的距离和方向
        vertices_exp = vertices.unsqueeze(2).repeat(1, 1, self.num_joints, 1)  # [B, N, J, 3]
        joints_exp = joints_pred.unsqueeze(1).repeat(1, N, 1, 1)              # [B, N, J, 3]
        rel_vec = vertices_exp - joints_exp                                    # [B, N, J, 3]
        rel_dist = jt.norm(rel_vec, dim=-1, keepdim=True)                      # [B, N, J, 1]
        rel_feat = concat([rel_dist, rel_vec], dim=-1)                         # [B, N, J, 4]
        rel_feat = self.dist_encoder(rel_feat)                                 # [B, N, J, feat_dim]

        # 点特征和joint特征融合
        vertex_feat_exp = vertex_feat.unsqueeze(2).repeat(1, 1, self.num_joints, 1)  # [B, N, J, feat_dim]
        joint_feat_exp = joint_feat.unsqueeze(1).repeat(1, N, 1, 1)                  # [B, N, J, feat_dim]
        fusion = vertex_feat_exp + joint_feat_exp + rel_feat                         # [B, N, J, feat_dim]
        fusion = fusion.view(B, N * self.num_joints, self.feat_dim)                  # [B, N*J, feat_dim]

        # 多层交互注意力
        for attn in self.cross_att_layers:
            fusion, _ = attn(fusion, fusion, fusion)
        fusion = fusion.view(B, N, self.num_joints, self.feat_dim)

        # 输出
        skin_weights = self.output(fusion)  # [B, N, J, num_joints]
        skin_weights = skin_weights.mean(-1)  # [B, N, J]
        skin_weights = nn.softmax(skin_weights, dim=-1)

        return joints_pred, skin_weights