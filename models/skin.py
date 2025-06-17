import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
from jittor.attention import MultiheadAttention
import numpy as np
import sys
from math import sqrt

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class SimpleSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res

class SimpleSkinModel2(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = Point_Transformer2(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res

from PCT.networks.cls.enhancedpct import EnhancedPointTransformer
class EnhancedSkinModel(nn.Module):
    def __init__(self, feat_dim=256, num_joints=22):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        
        # 使用增强版Transformer
        self.pct = EnhancedPointTransformer(output_channels=feat_dim, layers=6)
        
        # 顶点特征提取
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        
        # 关节特征提取
        self.joint_encoder = nn.Sequential(
            nn.Linear(3 + feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        
        # 交叉注意力机制
        self.cross_att = MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, num_joints)
        )

    def execute(self, vertices, joints):
        # 形状特征提取
        shape_latent = self.pct(vertices.permute(0, 2, 1))
        
        # 顶点特征增强
        vertex_feat = concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1)
        vertex_feat = self.vertex_encoder(vertex_feat.view(-1, 3 + self.feat_dim))
        vertex_feat = vertex_feat.view(vertices.shape[0], vertices.shape[1], self.feat_dim)
        
        # 关节特征增强
        joint_feat = concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1)
        joint_feat = self.joint_encoder(joint_feat.view(-1, 3 + self.feat_dim))
        joint_feat = joint_feat.view(joints.shape[0], self.num_joints, self.feat_dim)
        
        # 交叉注意力
        att_output, _ = self.cross_att(
            vertex_feat, 
            joint_feat, 
            joint_feat
        )
        
        # 门控融合
        gate_value = self.gate(concat([vertex_feat, att_output], dim=-1))
        fused_feat = gate_value * vertex_feat + (1 - gate_value) * att_output
        
        # 输出蒙皮权重
        skin_weights = self.output(fused_feat)
        return nn.softmax(skin_weights, dim=-1)

import jsparse.nn as spnn
from PCT.networks.jts.pct import PointTransformer3

class JSMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            spnn.Linear(input_dim, 512),
            spnn.BatchNorm(512),
            spnn.ReLU(),
            spnn.Linear(512, output_dim),
            spnn.BatchNorm(output_dim),
            spnn.ReLU(),
        )
    
    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)

class JSSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        self.pct = PointTransformer3(output_channels=feat_dim)
        self.joint_mlp = JSMLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = JSMLP(3 + feat_dim, feat_dim)
        self.relu = spnn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # (B, latents)
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # (B, N, latents)
        vertices_latent = (
            self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        )

        # (B, num_joints, latents)
        joints_latent = (
            self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        )

        # (B, N, num_joints)
        res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()

        return res

# Factory function to create models
def create_model(model_name='pct', feat_dim=256, **kwargs):
    if model_name == "pct":
        return SimpleSkinModel(feat_dim=feat_dim, num_joints=22)
    elif model_name == "jspct":
        return JSSkinModel(feat_dim=feat_dim, num_joints=22)
    elif model_name == "pct2":
        return SimpleSkinModel2(feat_dim=feat_dim, num_joints=22)
    elif model_name == "enhanced":
        return EnhancedSkinModel(feat_dim=feat_dim, num_joints=22)
    raise NotImplementedError()