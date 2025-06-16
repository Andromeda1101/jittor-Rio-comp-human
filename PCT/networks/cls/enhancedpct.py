import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

class EnhancedPointTransformer(nn.Module):
    def __init__(self, output_channels, layers):
        super().__init__()

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
        
        # 自适应层间特征聚合
        self.adaptive_fusion = AdaptiveLayerFusion(256, layers)
        
        # 层注意力
        self.layer_attention = LayerAttention(256, layers)
        
        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256 * layers, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )
        
        # 分类头
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        batch_size, C, N = x.size()
        x_input = x
        
        # 特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 分层局部特征聚合
        local_features = []
        curr_feat = x
        for sa_layer in self.local_sa_layers:
            curr_feat = sa_layer(curr_feat, x_input)
            local_features.append(curr_feat)
            
        # 自适应层间特征聚合
        x = self.adaptive_fusion(local_features)
        
        # 将特征连接
        x = concat(local_features, dim=1)  # [B, C*L, N]
        
        # 层注意力和后续处理
        x = self.layer_attention(x)
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        
        # 分类头
        x = self.relu(self.bn6(self.linear1(x)))
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
        self.relu = nn.ReLU()
        
    def execute(self, x, xyz):
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
        # 使用分组索引
        k_grouped = []
        v_grouped = []
        for g in range(self.groups):
            # 提取当前组的特征 [B, C/G, N]
            k_g = k[:, g, :, :]
            v_g = v[:, g, :, :]
            
            # 索引当前组的最近邻 [B, C/G, N, k]
            k_idx = index_points(k_g.permute(0, 2, 1), idx)  # [B, N, k, C/G]
            k_idx = k_idx.permute(0, 3, 1, 2)  # [B, C/G, N, k]
            
            v_idx = index_points(v_g.permute(0, 2, 1), idx)  # [B, N, k, C/G]
            v_idx = v_idx.permute(0, 3, 1, 2)  # [B, C/G, N, k]
            
            k_grouped.append(k_idx)
            v_grouped.append(v_idx)
        
        # 合并组 [B, G, C/G, N, k]
        k = jt.stack(k_grouped, dim=1)
        v = jt.stack(v_grouped, dim=1)
        
        # 处理位置编码
        # 将位置编码分组以匹配值特征的维度
        pos_emb = pos_emb.view(
            batch_size, 
            self.groups, 
            self.channels_per_group, 
            num_points, 
            self.k
        )
        
        # 位置编码加到值特征上
        v = v + pos_emb  # 添加位置信息
        
        # 准备查询向量 [B, G, C/G, N, 1]
        q = q.unsqueeze(-1)
        
        # 计算注意力分数
        # k的维度是 [B, G, C/G, N, k]
        energy = (q * k).sum(2) / jt.sqrt(float(self.channels_per_group))  # [B, G, N, k]
        attn = nn.softmax(energy, -1)  # [B, G, N, k]
        
        # 应用注意力到值特征
        # v的维度是 [B, G, C/G, N, k]
        # attn的维度是 [B, G, N, k] -> 扩展为 [B, G, 1, N, k]
        x_transformed = (v * attn.unsqueeze(2)).sum(-1)  # [B, G, C/G, N]
        
        # 合并组特征 [B, C, N]
        x_transformed = x_transformed.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, num_points)
        
        # 残差连接
        x = x + self.relu(self.bn(self.conv_final(x_transformed)))
        return x


class AdaptiveLayerFusion(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels
        # 修改卷积层结构，确保维度匹配
        self.conv_weights = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
            nn.Conv1d(channels // 2, num_layers, 1),
            nn.Softmax(dim=1)
        )
        
    def execute(self, features):
        batch_size = features[0].shape[0]
        # 将所有特征堆叠 [B, L, C, N]
        x = jt.stack(features, dim=1)
        
        # 计算注意力权重 [B, L, N]
        # 使用最后一层特征来计算权重
        weights = self.conv_weights(features[-1])  # [B, L, N]
        weights = weights.unsqueeze(2)  # [B, L, 1, N]
        
        # 应用权重
        out = (x * weights).sum(1)  # [B, C, N]
        return out

class LayerAttention(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.attention = nn.Sequential(
            nn.Conv1d(channels * num_layers, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, num_layers, 1),
            nn.Sigmoid()
        )
        
    def execute(self, x):
        batch_size = x.size(0)
        # x 是融合后的特征图 [B, C*L, N]
        
        # 计算注意力权重 [B, L, N]
        weights = self.attention(x)
        
        # 将输入特征调整为 [B, L, C, N]
        x = x.view(batch_size, self.num_layers, self.channels, -1)
        
        # 调整权重维度为 [B, L, 1, N]
        weights = weights.unsqueeze(2)
        
        # 应用注意力权重
        weighted = x * weights  # [B, L, C, N]
        
        # 重新整形为 [B, L*C, N]
        return weighted.view(batch_size, -1, weighted.size(-1))

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 

    network = EnhancedPointTransformer()
    out_logits = network(input_points)
    print (out_logits.shape)
