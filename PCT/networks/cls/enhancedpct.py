import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

class EnhancedPointTransformer(nn.Module):
    def __init__(self, output_channels=40, layers=8):
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
        for sa_layer in self.local_sa_layers:
            x = sa_layer(x, x_input)
            local_features.append(x)
            
        # 自适应层间特征聚合
        x = self.adaptive_fusion(local_features)
        
        # 层注意力
        x = self.layer_attention(x)
        x = self.conv_fuse(x)
        x = jt.max(x, 2).view(batch_size, -1)
        
        # 分类头
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class LocalFeatureAggregation(nn.Module):
    def __init__(self, channels, k=16, groups=8):
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
        q = self.conv_q(x).view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        k = self.conv_k(x).view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        v = self.conv_v(x).view(batch_size, self.groups, self.channels_per_group, num_points)  # [B, G, C/G, N]
        
        # 为每个点找到k个最近邻
        k = index_points(k.permute(0,2,3,1), idx).permute(0,3,1,2)  # [B, G, N, k]
        v = index_points(v.permute(0,2,3,1), idx).permute(0,3,1,2)  # [B, G, N, k]
        
        # Reshape for attention computation
        q = q.unsqueeze(-1)  # [B, G, C/G, N, 1]
        k = k.unsqueeze(2)   # [B, G, 1, N, k]
        v = v.unsqueeze(2)   # [B, G, 1, N, k]
        
        # Compute attention scores
        energy = (q * k).sum(2) / jt.sqrt(float(self.channels_per_group))  # [B, G, N, k]
        attn = nn.softmax(energy, -1)  # [B, G, N, k]
        
        # Apply attention to values
        x_transformed = (v * attn.unsqueeze(2)).sum(-1)  # [B, G, C/G, N]
        x_transformed = x_transformed.view(batch_size, -1, num_points)
        
        # 残差连接
        x = x + self.relu(self.bn(self.conv_final(x_transformed)))
        return x

class AdaptiveLayerFusion(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.conv_weights = nn.Conv1d(channels, num_layers, 1)
        
    def execute(self, features):
        # 计算自适应权重
        x = concat(features, dim=1)
        weights = nn.softmax(self.conv_weights(features[-1]), dim=1)
        
        # 加权融合
        out = 0
        for i, feat in enumerate(features):
            out += feat * weights[:, i:i+1]
        return out

class LayerAttention(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * num_layers, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels * num_layers, 1),
            nn.Sigmoid()
        )
        
    def execute(self, x):
        weights = self.attention(x)
        return x * weights

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = EnhancedPointTransformer()
    out_logits = network(input_points)
    print (out_logits.shape)

