import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
from dataset.format import parents
from PCT.networks.cls.unifiedpct import UnifiedPointTransformer
from PCT.networks.cls.pct import Point_Transformer2
from .bone_constraints import BoneConstraints

class UnifiedModel(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22, transformer_name='unified'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        if transformer_name == 'unified':
            # 共享的Transformer骨干网络
            self.transformer = UnifiedPointTransformer(output_channels=feat_dim)
        elif transformer_name == 'pct2':
            self.transformer = Point_Transformer2(output_channels=feat_dim)
        
        # 增强的骨骼预测分支 - 多尺度特征金字塔
        self.joint_pyramid = nn.ModuleList([
            nn.Linear(feat_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)  # 新增更细粒度特征
        ])
        
        # 添加骨骼约束
        self.bone_constraints = BoneConstraints()
        
        # 修改关节预测器，使用分层预测
        self.joint_predictor = HierarchicalJointPredictor(
            in_channels=512 + 256 + 128 + 64,
            num_joints=num_joints
        )
        
        # 修改顶点特征提取以适应新的SE模块
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            LightweightSE(512),  # 调整后的SE模块
            ResidualBlock(512, 512),
            LightweightSE(512),  # 调整后的SE模块
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
        
        # 简化的蒙皮预测分支
        self.skin_predictor = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_joints),
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
        # 保留其他必要组件
        self.fusion_weights = nn.Parameter(jt.ones(4))
        self.constraint_net = ConstraintNetwork(feat_dim, num_joints)
        self.constraint_weights = {
            'symmetry': 1.0,
            'planarity': 0.8
        }
    
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
        
        # 骨骼预测 - 使用特征金字塔
        pyramid_features = []
        x = shape_latent
        for layer in self.joint_pyramid:
            x = layer(x)
            pyramid_features.append(x)
        
        # 连接多尺度特征
        multi_scale_features = concat(pyramid_features, dim=1)
        joint_pred = self.joint_predictor(multi_scale_features)
        joint_pred = joint_pred.view(-1, self.num_joints, 3)
        
        # 约束网络处理 - 移除距离相关约束
        B, _, N = vertices.shape
        constraint_input = jt.concat([
            shape_latent,
            joint_pred.reshape(B, -1)
        ], dim=1)
        
        constraint_feat = self.constraint_net.constraint_encoder(constraint_input)
        
        # 只保留基本的形状约束
        symmetry_adjust = self.constraint_net.symmetry_predictor(constraint_feat)
        planarity_adjust = self.constraint_net.planarity_predictor(constraint_feat)
        
        # 将调整reshape为关节形状
        symmetry_adjust = symmetry_adjust.view(B, self.num_joints, 3)
        planarity_adjust = planarity_adjust.view(B, self.num_joints, 3)
        
        # 简化的约束调整
        joint_adjust = (
            self.constraint_weights['symmetry'] * symmetry_adjust +
            self.constraint_weights['planarity'] * planarity_adjust
        )
        
        # 软约束：通过残差连接应用调整
        joint_pred = joint_pred + 0.1 * joint_adjust
        
        # 简化的蒙皮预测
        vertices_flat = vertices.permute(0, 2, 1).reshape(B * N, 3)
        shape_latent_exp = shape_latent.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
        
        # 简单连接特征
        fusion_feat = concat([vertices_flat, shape_latent_exp], dim=1)
        
        # 直接预测蒙皮权重
        skin_weights = self.skin_predictor(fusion_feat)
        skin_weights = skin_weights.reshape(B, N, -1)  # [B, N, J]
        
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

# 修改 LightweightSE 模块以修复维度问题
class LightweightSE(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def execute(self, x):
        # 处理输入维度 (B*N, C) 或 (B, C)
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # 如果是3D输入，将其压缩为2D
            B, N, C = orig_shape
            x = x.reshape(-1, C)
        
        # 计算通道注意力
        y = jt.mean(x, dim=0)  # (C,)
        y = self.fc(y)  # (C,)
        
        # 广播乘法
        out = x * y
        
        # 恢复原始维度
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape)
        return out

# 修改 LightweightSEConv 模块以修复维度问题
class LightweightSEConv(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        
    def execute(self, x):
        # x shape: (B, C, N)
        y = jt.mean(x, dim=2, keepdims=True)  # (B, C, 1)
        y = self.fc(y)  # (B, C, 1)
        return x * y  # 广播乘法 (B, C, N)

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

class HierarchicalJointPredictor(nn.Module):
    def __init__(self, in_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        
        # 共享特征提取
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分层预测各个关节
        self.joint_predictors = nn.ModuleList()
        for i in range(num_joints):
            # 每个关节预测器获取父节点信息
            input_dim = 512 + (3 if parents[i] is not None else 0)
            self.joint_predictors.append(
                nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3)
                )
            )
    
    def execute(self, x):
        batch_size = x.shape[0]
        shared_features = self.shared_mlp(x)
        
        # 按照骨骼层次结构预测关节
        joint_positions = []
        for i in range(self.num_joints):
            if parents[i] is None:
                # 根节点直接预测
                joint_pos = self.joint_predictors[i](shared_features)
            else:
                # 非根节点基于父节点位置预测
                parent_pos = joint_positions[parents[i]]
                features = jt.concat([shared_features, parent_pos], dim=1)
                joint_pos = self.joint_predictors[i](features)
                # 相对位置预测
                joint_pos = parent_pos + joint_pos
                
            joint_positions.append(joint_pos)
            
        return jt.stack(joint_positions, dim=1)  # [B, num_joints, 3]

# 新增约束网络
class ConstraintNetwork(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22):
        super().__init__()
        self.num_joints = num_joints
        
        # 骨架约束感知网络
        self.constraint_encoder = nn.Sequential(
            nn.Linear(feat_dim + num_joints * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            nn.Linear(512, 256)
        )
        
        # 各部分约束预测器
        self.symmetry_predictor = nn.Linear(256, num_joints * 3)
        self.planarity_predictor = nn.Linear(256, num_joints * 3)