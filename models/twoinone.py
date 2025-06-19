import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
from math import sqrt
from PCT.networks.cls.enhancedpct import EnhancedPointTransformer

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.act2 = nn.Sigmoid()
    def execute(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        w = self.fc1(x_flat)
        w = self.act1(w)
        w = self.fc2(w)
        w = self.act2(w)
        w = w.reshape(*orig_shape[:-1], orig_shape[-1])
        return x * w

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

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.mlp = ResidualMLP(dim, dim*2, dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def execute(self, x):
        # x: [B, N, C]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class CrossTransformerLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = ResidualMLP(dim, dim*2, dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
    def execute(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        x = self.norm1(q + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class JointSkinModel(nn.Module):
    def __init__(self, feat_dim=384, num_joints=22, attn_heads=8, attn_layers=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        # 多尺度主干特征
        self.point_transformer = EnhancedPointTransformer(output_channels=feat_dim, return_point_features=True)
        self.local_transformer = EnhancedPointTransformer(output_channels=feat_dim//2, return_point_features=True)
        self.fuse_proj = nn.Linear(feat_dim + feat_dim//2, feat_dim)
        # 深层Transformer编码
        self.encoder = nn.Sequential(*[TransformerLayer(feat_dim, attn_heads) for _ in range(attn_layers)])
        # joints分支
        self.joint_proj = nn.Linear(feat_dim, feat_dim)
        self.joint_decoder = nn.Sequential(
            ResidualMLP(feat_dim, 512, 512),
            nn.GELU(),
            nn.Linear(512, num_joints * 3)
        )
        # skin分支
        self.vertex_encoder = nn.Sequential(
            ResidualMLP(3 + feat_dim, 512, feat_dim),
            SEBlock(feat_dim)
        )
        self.joint_encoder = nn.Sequential(
            ResidualMLP(3 + feat_dim, 512, feat_dim),
            SEBlock(feat_dim)
        )
        # 深层交叉Transformer
        self.cross_layers = nn.ModuleList([
            CrossTransformerLayer(feat_dim, attn_heads) for _ in range(attn_layers)
        ])
        # 空间关系编码
        self.dist_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, feat_dim)
        )
        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.Sigmoid()
        )
        # 输出
        self.output = nn.Sequential(
            ResidualMLP(feat_dim, 256, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, num_joints)
        )

    def execute(self, vertices):
        # vertices: [B, N, 3]
        B, N, _ = vertices.shape
        # 主干特征
        x_global = self.point_transformer(vertices.permute(0, 2, 1))  # [B, feat_dim, N]
        x_local = self.local_transformer(vertices.permute(0, 2, 1))
        x = concat([x_global, x_local], dim=1)  # [B, feat_dim+feat_dim//2, N]
        x = x.permute(0, 2, 1)  # [B, N, C]
        x = self.fuse_proj(x)
        # 深层编码
        x = self.encoder(x)
        # joints分支
        x_pool = jt.max(x, 1)  # [B, feat_dim]
        x_joint = self.joint_proj(x_pool)
        joints_pred = self.joint_decoder(x_joint).view(B, self.num_joints, 3)  # [B, J, 3]
        # skin分支
        vertex_feat = concat([vertices, x], dim=-1)
        vertex_feat = self.vertex_encoder(vertex_feat)  # [B, N, feat_dim]
        joint_feat = concat([joints_pred, x_joint.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1)
        joint_feat = self.joint_encoder(joint_feat)  # [B, J, feat_dim]
        # 空间关系
        vertices_exp = vertices.unsqueeze(2).repeat(1, 1, self.num_joints, 1)     # [B, N, J, 3]
        joints_exp = joints_pred.unsqueeze(1).repeat(1, N, 1, 1)                  # [B, N, J, 3]
        rel_vec = vertices_exp - joints_exp                                       # [B, N, J, 3]
        rel_dist = (rel_vec ** 2).sum(-1, keepdims=True).sqrt()                   # [B, N, J, 1]
        rel_feat = concat([rel_dist, rel_vec], dim=-1)                            # [B, N, J, 4]
        rel_feat = self.dist_encoder(rel_feat)                                    # [B, N, J, feat_dim]
        # 融合
        vertex_feat_exp = vertex_feat.unsqueeze(2).repeat(1, 1, self.num_joints, 1)  # [B, N, J, feat_dim]
        joint_feat_exp = joint_feat.unsqueeze(1).repeat(1, N, 1, 1)                  # [B, N, J, feat_dim]
        fusion_cat = concat([vertex_feat_exp, joint_feat_exp, rel_feat], dim=-1)     # [B, N, J, feat_dim*3]
        gate = self.fusion_gate(fusion_cat)
        fusion = gate * vertex_feat_exp + (1 - gate) * (joint_feat_exp + rel_feat)   # [B, N, J, feat_dim]
        fusion = fusion.view(B, N * self.num_joints, self.feat_dim)
        # 深层交叉融合
        for cross in self.cross_layers:
            fusion = cross(fusion, fusion, fusion)
        fusion = fusion.view(B, N, self.num_joints, self.feat_dim)
        # 输出
        skin_weights = self.output(fusion)                     # [B, N, J, num_joints]
        skin_weights = skin_weights.mean(-1)                   # [B, N, J]
        skin_weights = nn.softmax(skin_weights, dim=-1)
        return joints_pred, skin_weights