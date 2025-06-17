import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

class EnhancedPointTransformer(nn.Module):
    def __init__(self, output_channels, layers, feat_dim=512):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 多尺度特征提取
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        
        # 分层局部特征聚合
        self.local_sa_layers = nn.ModuleList()
        for i in range(layers):
            self.local_sa_layers.append(
                LocalFeatureAggregation(256, k=16, groups=8)
            )
        
        # 自适应层间特征融合
        self.adaptive_fusion = AdaptiveLayerFusion(256, layers)
        
        # 层注意力
        self.layer_attention = LayerAttention(256, layers)
        
        # 特征融合后增加自注意力
        self.attn_fusion = MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        
        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256 * layers, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(scale=0.2)
        )
        
        # 分类头
        self.linear1 = nn.Linear(feat_dim, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        self.relu = nn.ReLU()
        
        # 残差连接权重
        self.res_weights = nn.Parameter(jt.zeros(layers))

    def execute(self, x):
        batch_size, C, N = x.size()
        x_input = x
        
        # 特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 分层局部特征聚合
        local_features = []
        curr_feat = x
        for i, sa_layer in enumerate(self.local_sa_layers):
            curr_feat = sa_layer(curr_feat, x_input)
            # 带权重的残差连接
            identity = x if i == 0 else local_features[i-1]
            curr_feat = identity + self.res_weights[i] * curr_feat
            local_features.append(curr_feat)
            
        # 自适应层间特征融合
        fused_feat = self.adaptive_fusion(local_features)
        
        # 将特征连接
        x = concat(local_features, dim=1)  # [B, C*L, N]
        
        # 层注意力和后续处理
        x = self.layer_attention(x)
        x = self.conv_fuse(x)
        
        # 自注意力融合
        x = x.permute(0, 2, 1)  # [B, N, C]
        x, _ = self.attn_fusion(x, x, x)
        x = x.permute(0, 2, 1)  # [B, C, N]
        
        # 全局特征
        global_feat = jt.max(x, 2)
        
        # 分类头
        x = self.relu(self.bn6(self.linear1(global_feat)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class LocalFeatureAggregation(nn.Module):
    def __init__(self, channels, k, groups):
        super().__init__()
        self.k = k
        self.groups = groups
        self.channels_per_group = channels // groups
        
        self.conv_q = nn.Conv1d(channels, channels, 1, groups=groups)
        self.conv_k = nn.Conv1d(channels, channels, 1, groups=groups) 
        self.conv_v = nn.Conv1d(channels, channels, 1, groups=groups)
        
        self.conv_pos = nn.Conv2d(3, channels, 1)
        self.conv_final = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.ln = nn.LayerNorm(channels)  # 增加层归一化
        self.relu = nn.ReLU()
        
    def execute(self, x, xyz):
        identity = x  # 保存输入用于残差连接
        
        batch_size, channels, num_points = x.size()
        
        # KNN查询
        idx = knn_point(self.k, xyz.permute(0,2,1), xyz.permute(0,2,1))
        neighbors = index_points(x.permute(0,2,1), idx)  # [B, N, k, C]
        pos_grad = index_points(xyz.permute(0,2,1), idx) - xyz.permute(0,2,1).unsqueeze(2)  # [B, N, k, 3]
        
        # 位置编码
        pos_emb = self.conv_pos(pos_grad.permute(0,3,1,2))  # [B, C, N, k]
        
        # 多头注意力
        q = self.conv_q(x)  # [B, C, N]
        k = self.conv_k(x)  # [B, C, N]
        v = self.conv_v(x)  # [B, C, N]
        
        # 重新分组
        q = q.view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        k = k.view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        v = v.view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        
        # 为每个点找到k个最近邻
        k_grouped = []
        v_grouped = []
        for g in range(self.groups):
            k_g = k[:, g, :, :]
            v_g = v[:, g, :, :]
            
            k_idx = index_points(k_g.permute(0, 2, 1), idx)  # [B, N, k, C/G]
            k_idx = k_idx.permute(0, 3, 1, 2)  # [B, C/G, N, k]
            
            v_idx = index_points(v_g.permute(0, 2, 1), idx)  # [B, N, k, C/G]
            v_idx = v_idx.permute(0, 3, 1, 2)  # [B, C/G, N, k]
            
            k_grouped.append(k_idx)
            v_grouped.append(v_idx)
        
        k = jt.stack(k_grouped, dim=1)
        v = jt.stack(v_grouped, dim=1)
        
        # 处理位置编码
        pos_emb = pos_emb.view(
            batch_size, 
            self.groups, 
            self.channels_per_group, 
            num_points, 
            self.k
        )
        
        # 位置编码加到值特征上
        v = v + pos_emb
        
        # 准备查询向量 [B, G, C/G, N, 1]
        q = q.unsqueeze(-1)
        
        # 计算注意力分数
        energy = (q * k).sum(2) / jt.sqrt(float(self.channels_per_group))  # [B, G, N, k]
        attn = nn.softmax(energy, -1)  # [B, G, N, k]
        
        # 应用注意力到值特征
        x_transformed = (v * attn.unsqueeze(2)).sum(-1)  # [B, G, C/G, N]
        
        # 合并组特征 [B, C, N]
        x_transformed = x_transformed.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, num_points)
        
        # 残差连接 + 层归一化
        x_transformed = self.conv_final(x_transformed)
        x = identity + self.relu(self.bn(x_transformed))
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)  # 层归一化
        return x


class AdaptiveLayerFusion(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels
        
        # 增加融合权重网络复杂度
        self.conv_weights = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(scale=0.2),
            nn.Conv1d(channels * 2, num_layers, 1),
            # 移除 Softmax，稍后手动应用
        )
        
    def execute(self, features):
        # 计算全局特征
        global_features = [jt.mean(feat, dim=2) for feat in features]
        global_features = jt.stack(global_features, dim=1)  # [B, L, C]
        
        # 转置为 [B, C, L]
        global_features = global_features.permute(0, 2, 1)
        
        # 计算注意力权重 [B, num_layers, L]
        weights = self.conv_weights(global_features)
        
        # 应用 Softmax 并移除不必要的维度
        weights = nn.softmax(weights, dim=1)  # [B, num_layers, L]
        
        # 加权融合
        out = jt.zeros_like(features[0])
        for i in range(self.num_layers):
            # 选择当前层的权重 [B, 1, L] -> [B, 1]
            layer_weight = weights[:, i:i+1, :].mean(dim=2)
            
            # 应用权重 [B, C, N]
            out += features[i] * layer_weight.unsqueeze(-1).unsqueeze(-1)
        
        return out

class LayerAttention(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        
        # 增强注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(channels * num_layers, channels * 4, 1),
            nn.BatchNorm1d(channels * 4),
            nn.LeakyReLU(scale=0.2),
            nn.Conv1d(channels * 4, channels * 2, 1),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(scale=0.2),
            nn.Conv1d(channels * 2, num_layers, 1),
            nn.Sigmoid()
        )
        
    def execute(self, x):
        batch_size = x.size(0)
        weights = self.attention(x)  # [B, L, N]
        
        # 将输入特征调整为 [B, L, C, N]
        x = x.view(batch_size, self.num_layers, self.channels, -1)
        
        # 调整权重维度为 [B, L, 1, N]
        weights = weights.view(batch_size, self.num_layers, 1, -1)
        
        # 应用注意力权重
        weighted = x * weights
        return weighted.view(batch_size, -1, weighted.size(-1))