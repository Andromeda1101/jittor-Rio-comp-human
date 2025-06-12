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
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class SparseTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, npoint=512, nsample=32):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        
        # Original sparse convolution layer
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn1 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU()
        
        # Self-attention mechanism
        self.q_conv = spnn.Conv3d(out_channels, out_channels//4, kernel_size=1)
        self.k_conv = spnn.Conv3d(out_channels, out_channels//4, kernel_size=1)
        self.v_conv = spnn.Conv3d(out_channels, out_channels, kernel_size=1)
        
        self.after_norm = spnn.BatchNorm(out_channels)
        self.act = spnn.ReLU()
        
        # Add point cloud sampling and local feature extraction
        self.local_sa = nn.Sequential(
            spnn.Conv3d(out_channels*2, out_channels, 1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU()
        )

    def execute(self, x, coord):
        # Sparse convolution processing
        x_sparse = self.relu(self.bn1(self.conv1(x)))
        
        # Point cloud local feature extraction
        new_coord, grouped_feat = sample_and_group(
            self.npoint, self.nsample, 
            coord, x_sparse.features
        )
        
        # Merge features
        local_feat = self.local_sa(grouped_feat)
        
        # Convert back to sparse tensor
        x = SparseTensor(
            features=local_feat,
            indices=new_coord,
            size=x.size
        )
        
        # Self-attention
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        
        # Compute attention scores
        attention = spnn.sparse_matmul(q, k.transpose())
        attention = jt.nn.softmax(attention, dim=-1)
        
        # Apply attention
        out = spnn.sparse_matmul(attention, v)
        out = self.act(self.after_norm(out))
        
        return x + out, new_coord

class PointTransformer3(nn.Module):
    def __init__(self, output_channels=40):
        super().__init__()
        
        # Initial sparse convolution layers
        self.stem = nn.Sequential(
            spnn.Conv3d(3, 64, kernel_size=3),
            spnn.BatchNorm(64),
            spnn.ReLU()
        )
        
        # Modify transformer blocks to support point cloud sampling
        self.transformer1 = SparseTransformerBlock(64, 128, npoint=512, nsample=32)
        self.transformer2 = SparseTransformerBlock(128, 256, npoint=256, nsample=32)
        self.transformer3 = SparseTransformerBlock(256, 512, npoint=128, nsample=32)
        
        # Global pooling and classification
        self.global_pool = spnn.GlobalPool(op="max")
        
        self.classifier = nn.Sequential(
            spnn.Linear(512, 256),
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
        # x is [B, C, N] from permute(0, 2, 1), need to match Simple model
        batch_size = x.shape[0]
        num_points = x.shape[2]
        
        # Convert point cloud to SparseTensor
        coords = x.permute(0, 2, 1)  # [B, N, 3]
        
        # Process each batch item
        sparse_tensors = []
        for i in range(batch_size):
            points = coords[i]  # [N, 3]
            
            # Normalize coordinates to positive values
            points = points - points.min(dim=0)[0]
            
            # Create sparse tensor
            coords_i = jt.concat([jt.ones(num_points, 1) * i, points], dim=1)
            sparse_tensor = SparseTensor(
                values=points,
                indices=coords_i.int(),
                size=(batch_size, *[100]*3)  # Use appropriate spatial size
            )
            sparse_tensors.append(sparse_tensor)
        
        # Combine batch
        x = SparseTensor.cat(sparse_tensors)

        # Network forward
        x = self.stem(x)
        x, coord = self.transformer1(x, coords)
        x, coord = self.transformer2(x, coord)
        x, coord = self.transformer3(x, coord)
        
        # Global pooling to get [B, C] format
        x = self.global_pool(x)
        
        # Classification to match output format
        x = self.classifier(x)  # Output shape: [B, output_channels]
        
        return x  # Return [B, output_channels] to match Simple model
