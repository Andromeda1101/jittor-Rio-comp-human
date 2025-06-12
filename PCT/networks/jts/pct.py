import jittor as jt
from jittor import nn
import numpy as np
from jittor.contrib import concat
from jsparse import SparseTensor
import jsparse.nn as spnn
from jsparse.utils import sparse_quantize
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.bn1 = spnn.BatchNorm(out_channels)
        self.bn2 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU()

    def execute(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SparseTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, npoint=512, nsample=32):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        
        # Local feature extraction
        self.local_op = Local_op(in_channels, out_channels)
        
        # Position encoding
        self.pos_enc = spnn.Conv3d(3, out_channels, kernel_size=1)
        
        # Feature transformation
        self.conv1 = spnn.Conv3d(out_channels, out_channels, kernel_size=3)
        self.bn1 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU()

    def execute(self, x, coord):
        # Local feature learning
        local_feat = self.local_op(x)
        
        # Position encoding
        pos_feat = self.pos_enc(coord)
        
        # Combine features
        x = local_feat + pos_feat
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Sample and group
        new_coord, grouped_feat = sample_and_group(
            self.npoint, self.nsample,
            coord, x.features
        )
        
        return SparseTensor(
            values=grouped_feat,
            indices=new_coord.int(),
            size=x.size
        ), new_coord

class PointTransformer3(nn.Module):
    def __init__(self, output_channels=40):
        super().__init__()
        
        # Initial sparse convolution layers
        self.stem = nn.Sequential(
            spnn.Conv3d(3, 64, kernel_size=3),
            spnn.BatchNorm(64),
            spnn.ReLU()
        )
        
        # Enhanced transformer blocks
        self.transformer1 = SparseTransformerBlock(64, 128, npoint=512, nsample=32)
        self.transformer2 = SparseTransformerBlock(128, 256, npoint=256, nsample=32)
        self.transformer3 = SparseTransformerBlock(256, 512, npoint=128, nsample=32)
        
        # Add feature fusion
        self.conv_fuse = nn.Sequential(
            spnn.Conv3d(896, 1024, kernel_size=1),
            spnn.BatchNorm(1024),
            spnn.ReLU()
        )
        
        # Global pooling and classification
        self.classifier = nn.Sequential(
            spnn.Linear(1024, 256),
            spnn.BatchNorm(256),
            spnn.ReLU(),
            spnn.Dropout(p=0.5),
            spnn.Linear(256, 128),
            spnn.BatchNorm(128),
            spnn.ReLU(),
            spnn.Dropout(p=0.5),
            spnn.Linear(128, output_channels)
        )

    def execute(self, x):
        # x is [B, C, N] from permute(0, 2, 1)
        batch_size = x.shape[0]
        num_points = x.shape[2]
        
        # Convert point cloud to SparseTensor
        coords = x.permute(0, 2, 1)  # [B, N, 3]
        
        # Create single sparse tensor for whole batch
        batch_ids = jt.arange(batch_size).reshape(-1, 1).repeat(1, num_points).reshape(-1, 1)
        points = coords.reshape(-1, 3)
        indices = jt.concat([batch_ids, points], dim=1)
        
        # Create sparse tensor
        x = SparseTensor(
            values=points,
            indices=indices.int(),
            size=(batch_size, *[100]*3)
        )

        # Network forward
        x = self.stem(x)
        x, coord = self.transformer1(x, coords)
        x, coord = self.transformer2(x, coord)
        x, coord = self.transformer3(x, coord)
        
        # Feature fusion
        features = x.features
        batch_indices = x.indices[:, 0]
        pooled_features = []
        
        for i in range(batch_size):
            batch_mask = (batch_indices == i)
            if batch_mask.sum() > 0:
                batch_features = features[batch_mask]
                pooled_features.append(jt.max(batch_features, dim=0))
        
        x = jt.stack(pooled_features)
        x = self.conv_fuse(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
