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
        
        # 修改多尺度骨骼特征提取
        self.joint_features = nn.ModuleList([
            self._make_res_block(3, 64),
            self._make_res_block(64, 128),
            self._make_res_block(128, 256),
            self._make_res_block(256, feat_dim//4)
        ])
        
        # 层次化交叉注意力
        self.cross_attentions = nn.ModuleList([
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True),
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True),
            MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        ])
        
        # 添加维度转换辅助层
        self.vertex_feature_transform = nn.Sequential(
            nn.Linear(3 + feat_dim, feat_dim * 4),
            nn.BatchNorm1d(feat_dim * 4),
            nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim * 4)  # 添加额外线性层
        )
        
        # 修改蒙皮预测分支的输入维度处理
        self.skin_predictor = nn.Sequential(
            nn.Conv1d(feat_dim * 4, feat_dim * 2, 1),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(),
            ResidualConvBlock(feat_dim * 2, feat_dim * 2),
            ResidualConvBlock(feat_dim * 2, feat_dim),
            nn.Conv1d(feat_dim, num_joints, 1)
        )
        
        # 修改fusion_weights初始化方式
        self.fusion_weights = jt.array([1.0, 1.0, 1.0, 1.0]).float()
        self.fusion_weights.requires_grad = True
        
        self.constraint_net = ConstraintNetwork(feat_dim, num_joints)
        self.constraint_weights = {
            'symmetry': 1.0,
            'planarity': 0.8
        }
        
        # 修改参数初始化方式
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight = jt.init.xavier_gauss_(m.weight)
                if m.bias is not None:
                    m.bias.zero_()
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True
        
        # 应用初始化
        self.apply(init_weights)
        
        # 确保所有参数可训练
        for p in self.parameters():
            p.requires_grad = True

    def _make_res_block(self, in_dim, out_dim):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # 添加LayerNorm
            nn.Linear(out_dim, out_dim)
        )

    def execute(self, vertices):
        # 确保输入张量格式正确且有梯度
        vertices = vertices.float()
        vertices.requires_grad = True
        B, _, N = vertices.shape
        
        # 使用梯度检查点包装关键操作
        def checkpoint(function, *args):
            args = [arg.stop_grad() if isinstance(arg, jt.Var) else arg for arg in args]
            return function(*args)
        
        # 1. 修改提取顶点特征的部分(添加残差连接)
        vertex_features = []
        x = vertices.permute(0, 2, 1)  # [B,N,3]
        residual = x
        
        for i, layer in enumerate(self.joint_features):
            if i > 0:
                x = x + residual  # 添加残差连接
            x = checkpoint(layer, x)  # 使用检查点
            x.requires_grad = True
            vertex_features.append(x)
            residual = x
        
        # 2. 提取全局特征(确保梯度)
        shape_latent = checkpoint(self.transformer, vertices)
        shape_latent.requires_grad = True
        
        # 3. 骨骼预测分支(添加梯度监控)
        pyramid_features = []
        x = shape_latent
        for layer in self.joint_pyramid:
            def hook_fn(grad):
                if grad is None:
                    print(f"Gradient is None for layer {layer}")
                return grad
            
            x = layer(x)
            x.register_hook(hook_fn)
            x.requires_grad = True
            pyramid_features.append(x)
        
        # 4. 特征融合
        multi_scale_features = concat(pyramid_features, dim=1)
        multi_scale_features.requires_grad = True
        
        # 5. 预测关节
        joint_pred = self.joint_predictor(multi_scale_features)
        joint_pred = joint_pred.view(-1, self.num_joints, 3)
        joint_pred.requires_grad = True
        
        # 6. 约束处理
        constraint_input = jt.concat([
            shape_latent,
            joint_pred.reshape(B, -1)
        ], dim=1)
        constraint_input.requires_grad = True
        
        # 7. 特征变换与蒙皮预测
        vertices_feat = vertices.permute(0, 2, 1).reshape(B * N, 3)  # [B*N, 3]
        shape_feat = shape_latent.unsqueeze(1).expand(-1, N, -1)  # [B, N, feat_dim]
        shape_feat = shape_feat.reshape(B * N, -1)  # [B*N, feat_dim]
        
        # 确保特征有梯度
        vertices_feat.requires_grad = True
        shape_feat.requires_grad = True
        
        # 特征融合
        fusion_feat = jt.concat([vertices_feat, shape_feat], dim=1)
        fusion_feat.requires_grad = True
        
        # 特征变换
        transformed_feat = self.vertex_feature_transform(fusion_feat)
        transformed_feat = transformed_feat.reshape(B, N, -1).permute(0, 2, 1)
        transformed_feat.requires_grad = True
        
        # 预测蒙皮权重
        skin_weights = self.skin_predictor(transformed_feat)
        skin_weights = skin_weights.permute(0, 2, 1)  # [B, N, num_joints]
        
        # 应用softmax
        skin_weights = jt.nn.softmax(skin_weights, dim=-1)
        skin_weights.requires_grad = True
        
        # 计算约束
        constraint_feat = self.constraint_net.constraint_encoder(constraint_input)
        symmetry_adjust = self.constraint_net.symmetry_predictor(constraint_feat)
        planarity_adjust = self.constraint_net.planarity_predictor(constraint_feat)
        
        # 应用约束
        symmetry_adjust = symmetry_adjust.view(B, self.num_joints, 3)
        planarity_adjust = planarity_adjust.view(B, self.num_joints, 3)
        
        joint_adjust = (
            self.constraint_weights['symmetry'] * symmetry_adjust +
            self.constraint_weights['planarity'] * planarity_adjust
        )
        
        # 最终调整
        joint_pred = joint_pred + 0.1 * joint_adjust
        
        return joint_pred, skin_weights
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)  # 使用LayerNorm
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)  # 使用LayerNorm
        
        self.shortcut = (nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        ) if in_dim != out_dim else nn.Identity())
        
    def execute(self, x):
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.ln2(out)
        
        out = out + identity
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
        
        # 修改shortcut确保梯度传递
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def execute(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
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