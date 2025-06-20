import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

class UnifiedPointTransformer(nn.Module):
    def __init__(self, output_channels=256, layers=6, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        
        # 增强的SA层
        self.sa_layers = nn.ModuleList()
        for i in range(layers):
            self.sa_layers.append(SA_Layer(256, reduction=reduction))
        
        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256 * layers, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )
        
        # 全局特征提取
        self.global_pool = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(512, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        
        self.relu = nn.ReLU()
    
    def execute(self, x):
        batch_size, C, N = x.size()
        
        # 特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 多级SA处理
        sa_outputs = []
        for sa_layer in self.sa_layers:
            x = sa_layer(x, x)
            sa_outputs.append(x)
        
        # 多尺度特征融合
        x = concat(sa_outputs, dim=1)
        x = self.conv_fuse(x)
        
        x = self.global_pool(x)  # (B, 512, N)
        global_feat = x.mean(dim=2)  # 使用平均池化
        output = self.output(global_feat)
        
        return output

class SA_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # 残差连接
        self.residual = nn.Conv1d(channels, channels, 1)

    def execute(self, x, xyz):
        # 残差连接
        residual = self.residual(x)
        
        # 位置编码
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B, N, C//4)
        x_k = self.k_conv(x)  # (B, C//4, N)
        x_v = self.v_conv(x)  # (B, C, N)
        
        # 自注意力
        energy = nn.bmm(x_q, x_k)  # (B, N, N)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        
        # 通道注意力
        channel_att = self.channel_att(x_v.permute(0, 2, 1).mean(1))  # (B, C)
        x_v = x_v * channel_att.unsqueeze(2)  # (B, C, N)
        
        # 特征融合
        x_r = nn.bmm(x_v, attention)  # (B, C, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        
        # 残差连接
        return residual + x_r