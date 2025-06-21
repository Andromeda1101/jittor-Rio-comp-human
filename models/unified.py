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
        
        # 增强的顶点特征提取
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, feat_dim)
        )
        
        # 多尺度骨骼特征提取
        self.joint_features = nn.ModuleList([
            self._make_res_block(3, 64, feat_dim//4),
            self._make_res_block(3, 128, feat_dim//4),
            self._make_res_block(3, 256, feat_dim//4),
            self._make_res_block(3, 512, feat_dim//4)
        ])
        
        # 层次化交叉注意力
        self.cross_attentions = nn.ModuleList([
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True),
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True),
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        ])
        
        # 改进的蒙皮预测分支 - 调整维度处理
        self.skin_predictor = nn.Sequential(
            nn.Conv1d(feat_dim * 4, feat_dim * 2, 1),  # 输入通道增加到4倍
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(),
            ResidualConvBlock(feat_dim * 2, feat_dim * 2),
            ResidualConvBlock(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Conv1d(feat_dim, num_joints, 1),
        )
    
    def _make_res_block(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
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
        
        # 多尺度骨骼特征
        joint_feats = []
        for feat_extractor in self.joint_features:
            feat = feat_extractor(joint_pred.reshape(B * self.num_joints, 3))
            joint_feats.append(feat.view(B, self.num_joints, -1))
        joint_feat = concat(joint_feats, dim=-1)  # B, J, feat_dim
        
        # 层次化注意力交互
        att_output1, _ = self.cross_attentions[0](vertex_feat, joint_feat, joint_feat)
        att_output2, _ = self.cross_attentions[1](att_output1, joint_feat, joint_feat)
        att_output3, _ = self.cross_attentions[2](att_output2, joint_feat, joint_feat)

        # 特征融合并调整维度
        fusion_feat = concat([
            vertex_feat, 
            att_output1, 
            att_output2,
            att_output3
        ], dim=-1)  # (B, N, feat_dim*3)
        
        # 调整维度以适应Conv1d
        fusion_feat = fusion_feat.permute(0, 2, 1)  # (B, feat_dim*3, N)
        
        # 改进的距离约束计算
        vertices_3d = vertices.permute(0, 2, 1)  # (B, N, 3)
        dist = jt.norm(vertices_3d.unsqueeze(2) - joint_pred.unsqueeze(1), dim=-1)  # (B, N, J)
        dist_weight = jt.exp(-dist * 0.1)  # 软化的距离权重
        
        # 蒙皮权重预测
        skin_logits = self.skin_predictor(fusion_feat)  # (B, num_joints, N)
        skin_weights = nn.softmax(skin_logits * dist_weight.permute(0, 2, 1), dim=1)  # (B, num_joints, N)
        skin_weights = skin_weights.permute(0, 2, 1)  # (B, N, num_joints)
        
        return joint_pred, skin_weights
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def execute(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

# 新增卷积残差块
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def execute(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)