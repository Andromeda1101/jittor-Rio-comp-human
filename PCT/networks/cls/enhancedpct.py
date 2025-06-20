import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

class EnhancedPointTransformer(nn.Module):
    def __init__(self, output_channels=40, layers=8, num_joints=22):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        
        # 增加SA层数量并添加残差
        self.sa_layers = nn.ModuleList()
        for i in range(layers):
            self.sa_layers.append(EnhancedSA_Layer(256))
        
        # 骨骼结构感知模块
        self.joint_aware_att = nn.ModuleList([
            JointAwareAttention(256, num_joints),
            JointAwareAttention(256, num_joints)
        ])
        
        # 多尺度特征融合
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256 * (layers + 2), 1024, kernel_size=1, bias=False),
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
        
        # 多级SA处理
        sa_outputs = []
        for sa_layer in self.sa_layers:
            x = sa_layer(x, x_input)
            sa_outputs.append(x)
        
        # 骨骼结构感知
        joint_aware_feat = x
        for att_layer in self.joint_aware_att:
            joint_aware_feat = att_layer(joint_aware_feat)
        
        # 多尺度特征融合
        x = concat(sa_outputs + [joint_aware_feat], dim=1)
        x = self.conv_fuse(x)
        x = jt.max(x, 2).view(batch_size, -1)
        
        # 分类头
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class EnhancedSA_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)
        
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def execute(self, x, xyz):
        xyz_feat = self.xyz_proj(xyz)
        x = x + xyz_feat
        
        # 残差连接
        residual = x
        
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        
        energy = nn.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        
        # 通道注意力
        channel_att = self.channel_att(x_v.permute(0, 2, 1).mean(1))
        x_v = x_v * channel_att.unsqueeze(2)
        
        x_r = nn.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(residual - x_r)))
        
        # 残差连接
        return residual + x_r

class JointAwareAttention(nn.Module):
    def __init__(self, channels, num_joints):
        super().__init__()
        # 关节嵌入参数
        self.joint_embeddings = nn.Parameter(init.gauss((num_joints, channels), 'float32'))
        self.attn = jt.attention.MultiheadAttention(channels, num_heads=4, batch_first=True)
    
    def execute(self, x):
        """
        x: [B, C, N] 点云特征
        """
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # [B, N, C]
        
        # 关节嵌入 [1, J, C] -> [B, J, C]
        joint_emb = self.joint_embeddings.unsqueeze(0).repeat(B, 1, 1)
        
        # 点与关节的交叉注意力
        attn_output, _ = self.attn(x, joint_emb, joint_emb)
        
        # 残差连接
        return (x + attn_output).permute(0, 2, 1)  # 恢复为 [B, C, N]

if __name__ == '__main__':
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')
    network = EnhancedPointTransformer()
    out_logits = network(input_points)
    print(out_logits.shape)