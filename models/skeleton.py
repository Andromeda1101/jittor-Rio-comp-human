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
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        
        self.transformer = EnhancedPointTransformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

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
    elif model_name == "unified":
        from models.unified import UnifiedModel
        return UnifiedModel(feat_dim=256, num_joints=output_channels // 3)
    raise NotImplementedError()
