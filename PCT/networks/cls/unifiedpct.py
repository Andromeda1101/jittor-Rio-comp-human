import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler, knn_point, index_points

def sample_and_group(npoint, nsample, xyz, points):
    """
    Input:
        npoint: number of points to sample
        nsample: number of points in each local region
        xyz: input points coordinates [B, N, C]
        points: input points features [B, N, D]
    Return:
        new_xyz: sampled points coordinates [B, npoint, C]
        new_points: sampled points features [B, npoint, nsample, D+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz)
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class UnifiedPointTransformer(nn.Module):
    def __init__(self, output_channels=256, layers=4):
        super().__init__()
        # 初始特征提取
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        
        # 层次化处理
        self.sa_layers = nn.ModuleList()
        self.npoints = [512, 256, 128, 64]  # 逐层降采样
        self.nsamples = [32, 32, 32, 32]    # 局部区域点数
        
        in_channel = 128
        for i in range(layers):
            self.sa_layers.append(
                SA_Layer(in_channel * 2)  # *2是因为sample_and_group会拼接特征
            )
            in_channel *= 2
        
        # 全局特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(in_channel, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_channels)
        )
        
        self.relu = nn.ReLU()
    
    def execute(self, x):
        # 输入预处理
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        batch_size = x.size(0)
        
        # 初始特征提取
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 128, N]
        points = x.permute(0, 2, 1)  # [B, N, 128]
        
        # 多层特征提取
        for i in range(len(self.sa_layers)):
            # 采样和分组
            new_xyz, new_points = sample_and_group(
                self.npoints[i], 
                self.nsamples[i],
                xyz, 
                points
            )
            
            # 特征转换 - 明确维度处理
            B, N, S, D = new_points.shape
            new_points = new_points.reshape(B * N, S, D)  # [B*N, S, D]
            new_xyz_expand = new_xyz.reshape(B * N, 1, 3)  # [B*N, 1, 3]
            
            # SA层处理
            new_points = self.sa_layers[i](new_points, new_xyz_expand)  # [B*N, S, C]
            new_points = new_points.reshape(B, N, S, -1)  # [B, N, S, C]
            new_points = new_points.mean(dim=2)  # [B, N, C] - 池化操作
            
            # 更新点和特征
            xyz = new_xyz
            points = new_points
        
        # 调整维度用于特征提取
        x = points.permute(0, 2, 1)  # [B, C, N]
        
        # 全局特征提取
        x = self.conv_fuse(x)
        x = jt.max(x, dim=2)
        x = x.view(batch_size, -1)
        
        # 分类
        x = self.classifier(x)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        # 简化位置编码实现
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU()
        )
    
    def execute(self, x, xyz):
        # 输入x: [B*N, S, C], xyz: [B*N, 1, 3]
        B, S, C = x.shape
        
        # 1. 计算位置编码
        xyz = xyz.repeat(1, S, 1)  # [B*N, S, 3]
        pos_enc = self.pos_mlp(xyz)  # [B*N, S, C]
        
        # 2. 添加位置信息到输入特征
        x = x + pos_enc  # 直接相加而不是concat
        
        # 3. 调整维度进行注意力计算
        x = x.permute(0, 2, 1)  # [B*N, C, S]
        
        # 4. Self-attention 计算
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B*N, S, C/4]
        x_k = self.k_conv(x)  # [B*N, C/4, S]
        x_v = self.v_conv(x)  # [B*N, C, S]
        
        # 5. 计算注意力得分
        energy = nn.bmm(x_q, x_k)  # [B*N, S, S]
        attention = self.softmax(energy / jt.sqrt(C/4))
        
        # 6. 应用注意力
        x_r = nn.bmm(x_v, attention.permute(0, 2, 1))  # [B*N, C, S]
        x_r = self.trans_conv(x - x_r)
        x_r = self.after_norm(x_r)
        x_r = self.act(x_r)
        x = x + x_r
        
        return x.permute(0, 2, 1)  # [B*N, S, C]