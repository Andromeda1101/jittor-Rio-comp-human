import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

class SimpleSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

class SimpleSkeletonModel2(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer2(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

from PCT.networks.cls.enhancedpct import EnhancedPointTransformer

class EnhancedSkeletonModel(nn.Module):
    def __init__(self, feat_dim=512, output_channels=66):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        
        # 使用增强点云Transformer
        self.transformer = EnhancedPointTransformer(output_channels=feat_dim, layers=6)
        
        # 增强的MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            
            nn.Linear(512, output_channels)
        )
        
        # 辅助输出层（深度监督）
        self.aux_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_channels)
            )
            for _ in range(3)
        ])

    def execute(self, vertices: jt.Var):
        # 提取特征
        x = self.transformer(vertices)
        
        # 主输出
        main_output = self.mlp(x)
        
        # 辅助输出（深度监督）
        aux_outputs = []
        for aux_layer in self.aux_outputs:
            aux_outputs.append(aux_layer(x))
        
        return main_output, aux_outputs
    

    
import jsparse.nn as spnn
from PCT.networks.jts.pct import PointTransformer3

class JSSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = PointTransformer3(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            spnn.Linear(feat_dim, 512),
            spnn.BatchNorm(512),
            spnn.ReLU(),
            spnn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    elif model_name == "jspct":
        return JSSkeletonModel(feat_dim=256, output_channels=output_channels)
    elif model_name == "pct2":
        return SimpleSkeletonModel2(feat_dim=256, output_channels=output_channels)
    elif model_name == "enhanced":
        return EnhancedSkeletonModel(feat_dim=256, output_channels=output_channels)
    raise NotImplementedError()
