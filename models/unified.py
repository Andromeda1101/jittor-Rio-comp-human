import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
from dataset.format import parents
from PCT.networks.cls.unifiedpct import UnifiedPointTransformer
from PCT.networks.cls.pct import Point_Transformer2
from .bone_constraints import BoneConstraints

class UnifiedModel(nn.Module):
    def __init__(self, feat_dim=512, num_joints=22, transformer_name='unified'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 1. 增强骨干网络
        if transformer_name == 'unified':
            self.transformer = UnifiedPointTransformer(output_channels=feat_dim)
        elif transformer_name == 'pct2':
            self.transformer = Point_Transformer2(output_channels=feat_dim)
            
        # 2. 多尺度特征提取
        self.joint_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),  # 改为输出feat_dim//2
                nn.LayerNorm(feat_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(feat_dim // 2, feat_dim // 4),  # 改为输出feat_dim//4
                nn.LayerNorm(feat_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(feat_dim // 4, feat_dim // 8),  # 改为输出feat_dim//8
                nn.LayerNorm(feat_dim // 8),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])

        self.pyramid_proj = nn.Linear(
            (feat_dim // 2) + (feat_dim // 4) + (feat_dim // 8),
            feat_dim
        )

        # 3. 点云特征提取增强 - 修改输入维度处理
        self.point_feature_net = nn.Sequential(
            nn.Linear(3 + feat_dim, feat_dim * 2),  # [B, N, 3+feat_dim*2] -> [B, N, feat_dim*2]
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

        # 4. 层次化关节预测器
        self.joint_predictor = HierarchicalJointPredictor(
            in_channels=feat_dim * 4,
            num_joints=num_joints
        )

        # 5. 跨模态注意力融合
        self.cross_attention = nn.ModuleList([
            MultiheadAttention(feat_dim, 8, batch_first=True),  # 第一个注意力层
            MultiheadAttention(feat_dim, 8, batch_first=True)   # 第二个注意力层
        ])

        # 6. 条件特征生成器 - 调整维度
        self.feature_adapter = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )

        self.condition_generator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Tanh()
        )

        # 7. 骨架约束感知模块
        self.bone_constraints = BoneConstraints()
        self.constraint_encoder = nn.Sequential(
            nn.Linear(num_joints * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # 8. 蒙皮预测
        self.skin_predictor = nn.Sequential(
            nn.Conv1d(feat_dim * 2, feat_dim * 2, 1),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(),
            ResidualConvBlock(feat_dim * 2, feat_dim * 2),
            ResidualConvBlock(feat_dim * 2, feat_dim),
            nn.Conv1d(feat_dim, num_joints, 1)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = jt.init.xavier_gauss_(m.weight)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                if m.bias is not None:
                    m.bias.zero_()
                if m.weight is not None:
                    m.weight.fill_(1.0)

        # 确保所有参数可训练
        for p in self.parameters():
            p.requires_grad = True

    def execute(self, vertices):
        vertices = vertices.float()
        B, _, N = vertices.shape
        
        # 1. 点云特征提取
        shape_latent = self.transformer(vertices)  # [B, feat_dim]
        vertices_local = vertices.permute(0, 2, 1)  # [B, N, 3]
        
        # 2. 扩展全局特征并连接局部特征
        shape_feat_expanded = shape_latent.unsqueeze(1).expand(-1, N, -1)  # [B, N, feat_dim*2]
        vertex_input = jt.concat([vertices_local, shape_feat_expanded], dim=-1)  # [B, N, 3+feat_dim*2]
        
        # 3. 点云局部特征提取
        point_features = self.point_feature_net(vertex_input)  # [B, N, feat_dim]
        
        # 4. 跨模态特征融合
        attn_output1, _ = self.cross_attention[0](
            point_features,  # query: [B, N, feat_dim]
            point_features,  # key: [B, N, feat_dim]
            point_features  # value: [B, N, feat_dim]
        )
        
        # 5. 全局特征聚合
        global_point_feat = jt.mean(attn_output1, dim=1)  # [B, feat_dim]
        
        # 6. 多尺度特征金字塔
        pyramid_features = []
        x = shape_latent
        for layer in self.joint_pyramid:
            x = layer(x)
            pyramid_features.append(x)

        # 投影到统一维度
        multi_scale_features = jt.concat(pyramid_features, dim=1)
        multi_scale_features = self.pyramid_proj(multi_scale_features)  # [B, feat_dim]
        
        # 7. 第二次注意力融合
        attn_output2, _ = self.cross_attention[1](
            multi_scale_features.unsqueeze(1),  # [B, 1, feat_dim*7/4]
            global_point_feat.unsqueeze(1),     # [B, 1, feat_dim]
            global_point_feat.unsqueeze(1)      # [B, 1, feat_dim]
        )
        
        # 8. 生成条件特征
        condition_features = self.condition_generator(shape_latent)  # [B, feat_dim]
        
        # 9. 特征整合
        fused_features = jt.concat([
            multi_scale_features,
            attn_output2.squeeze(1),
            global_point_feat,
            condition_features
        ], dim=1)
        
        # 10. 预测关节位置和蒙皮权重
        joint_pred = self.joint_predictor(fused_features)  # [B, num_joints, 3]
        
        # 11. 优化约束
        constraint_feat = self.constraint_encoder(joint_pred.reshape(B, -1))  # [B, feat_dim]
        constraint_feat = constraint_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, feat_dim]
        
        # 12. 蒙皮权重预测
        skin_input = jt.concat([point_features, constraint_feat], dim=-1)  # [B, N, feat_dim*2]
        skin_features = skin_input.permute(0, 2, 1)  # [B, feat_dim*2, N]
        skin_weights = self.skin_predictor(skin_features)  # [B, num_joints, N]
        skin_weights = skin_weights.permute(0, 2, 1)  # [B, N, num_joints]
        skin_weights = nn.softmax(skin_weights, dim=-1)  # 确保权重和为1
        
        # 同步返回值
        joint_pred.sync()
        skin_weights.sync()
        
        return joint_pred, skin_weights
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        
        # 确保shortcut维度正确
        self.shortcut = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        ) if in_dim != out_dim else nn.Identity()
        
    def execute(self, x):
        # 保存原始维度
        orig_shape = x.shape
        if len(orig_shape) == 3:
            B, N, C = orig_shape
            x = x.reshape(B*N, C)
        
        # 主路径
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.ln2(out)
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        # 恢复原始维度
        if len(orig_shape) == 3:
            out = out.reshape(B, N, -1)
            
        return out

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
        # 输入维度处理
        orig_shape = x.shape
        if len(orig_shape) == 3:
            B, N, C = orig_shape
            x = x.reshape(B*N, C)  # 展平为 [B*N, C]
        
        # 计算通道注意力 - 修正为按特征维度平均
        y = jt.mean(x, dim=0, keepdims=True)  # [1, C] 保持维度
        y = self.fc(y)  # [1, C]
        
        # 应用注意力权重
        out = x * y  # 广播乘法 [B*N, C] * [1, C]
        
        # 恢复原始维度
        if len(orig_shape) == 3:
            out = out.reshape(B, N, C)
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
        
        # 确保shortcut维度匹配
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def execute(self, x):
        # 保存输入维度
        identity = self.shortcut(x)
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        # 确保梯度传播
        out.sync()
        return out

class HierarchicalJointPredictor(nn.Module):
    def __init__(self, in_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        
        # 1. 增强共享特征提取和骨骼链编码
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 2. 骨骼链特征编码器 - 使用ModuleList替代ModuleDict
        self.chain_names = ['spine', 'l_arm', 'r_arm', 'l_leg', 'r_leg']
        self.chain_encoders = nn.ModuleList([
            self._make_chain_encoder(512, 256) for _ in range(len(self.chain_names))
        ])
        # 创建名称到索引的映射
        self.chain_name_to_idx = {name: idx for idx, name in enumerate(self.chain_names)}
        
        # 3. 全局骨架特征提取器
        self.global_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 4. 父节点注意力模块
        self.parent_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128)
            ) for _ in range(num_joints)
        ])
        
        # 5. 改进的关节预测器
        self.joint_predictors = nn.ModuleList()
        for i in range(num_joints):
            input_dim = 512 + 128 + 256  # 局部特征 + 全局特征 + 链特征
            if parents[i] is not None:
                input_dim += 128  # 父节点注意力特征
            
            self.joint_predictors.append(nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 3)
            ))
        
        # 6. 位置细化网络
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 + 3 + 256, 128),  # 全局特征 + 相对位置 + 链特征
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 3)
            ) for _ in range(num_joints)
        ])

    def _make_chain_encoder(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def _get_chain_for_joint(self, joint_idx):
        # 根据关节索引返回所属的骨骼链
        spine_joints = [0, 1, 2, 3, 4, 5]
        l_arm_joints = [6, 7, 8, 9]
        r_arm_joints = [10, 11, 12, 13]
        l_leg_joints = [14, 15, 16, 17]
        r_leg_joints = [18, 19, 20, 21]
        
        if joint_idx in spine_joints:
            return 'spine'
        elif joint_idx in l_arm_joints:
            return 'l_arm'
        elif joint_idx in r_arm_joints:
            return 'r_arm'
        elif joint_idx in l_leg_joints:
            return 'l_leg'
        elif joint_idx in r_leg_joints:
            return 'r_leg'
        return None

    def execute(self, x):
        batch_size = x.shape[0]
        
        # 1. 提取共享特征
        shared_features = self.shared_mlp(x)  # [B, 512]
        
        # 2. 提取全局特征
        global_feat = self.global_features(shared_features)  # [B, 128]
        
        # 3. 生成骨骼链特征 - 使用ModuleList
        chain_features = {}
        for name in self.chain_names:
            idx = self.chain_name_to_idx[name]
            chain_features[name] = self.chain_encoders[idx](shared_features)
        
        # 4. 层次化预测
        joint_positions = []
        for i in range(self.num_joints):
            # 获取当前关节所属的骨骼链特征
            chain_name = self._get_chain_for_joint(i)
            chain_feat = chain_features[chain_name]
            
            if parents[i] is None:
                # 根节点预测
                features = jt.concat([
                    shared_features,
                    global_feat,
                    chain_feat
                ], dim=1)
                joint_pos = self.joint_predictors[i](features)
            else:
                # 非根节点预测
                parent_pos = joint_positions[parents[i]]
                
                # 计算父节点注意力特征
                parent_attn = self.parent_attention[i](parent_pos)
                
                # 特征融合
                features = jt.concat([
                    shared_features,
                    global_feat,
                    chain_feat,
                    parent_attn
                ], dim=1)
                
                # 预测相对位置
                relative_pos = self.joint_predictors[i](features)
                
                # 应用细化网络
                refine_input = jt.concat([
                    global_feat,
                    relative_pos,
                    chain_feat
                ], dim=1)
                refined_offset = self.refinement[i](refine_input)
                
                # 最终位置 = 父节点位置 + 细化后的偏移
                joint_pos = parent_pos + refined_offset
            
            joint_positions.append(joint_pos)
        
        joints = jt.stack(joint_positions, dim=1)  # [B, num_joints, 3]
        
        # 确保梯度传播
        joints.sync()
        return joints

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