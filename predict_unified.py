import jittor as jt
import numpy as np
import os
import argparse

from dataset.asset import Asset
from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.unified import UnifiedModel

import numpy as np
from scipy.spatial import cKDTree
import random

from tqdm import tqdm

# Set Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    # Create model
    model = UnifiedModel(
        feat_dim=args.feat_dim,
        num_joints=args.num_joints,
        transformer_name=args.transformer_name
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        model.load(args.pretrained_model)
    
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=True,
    )
    predict_output_dir = args.predict_output_dir
    print("start predicting...")
    for batch_idx, data in tqdm(enumerate(predict_loader)):
        # currently only support batch_size==1 because origin_vertices is not padded
        vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data['origin_vertices'], data['N']
        B = vertices.shape[0]
        vertices = vertices.permute(0, 2, 1)  # (B, 3, N)
        joint_pred, skin_pred = model(vertices)
        # joint_pred = joint_pred.reshape(B, -1, 3)  # 确保骨骼格式正确
        for i in range(B):
            # resample
            skin = skin_pred[i].numpy()  # (1024, 22)
            o_vertices = origin_vertices[i, :N[i]].numpy()  # (N, 3)

            tree = cKDTree(vertices[i].permute(1, 0).numpy())  # 转换回(N, 3)用于KD树
            distances, indices = tree.query(o_vertices, k=3)

            weights = 1 / (distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)

            # 计算插值后的蒙皮权重
            skin_resampled = np.zeros((o_vertices.shape[0], skin.shape[1]))  # (N, 22)
            for v in range(o_vertices.shape[0]):
                skin_resampled[v] = weights[v] @ skin[indices[v]]
            
            path = os.path.join(predict_output_dir, cls[i], str(id[i].item()))
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "predict_skeleton"), joint_pred[i])  # (22, 3)
            np.save(os.path.join(path, "predict_skin"), skin_resampled)  # (N, 22)
            np.save(os.path.join(path, "transformed_vertices"), o_vertices)  # (N, 3)
    print("finished")

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, required=True,
                        help='Path to the prediction data list file')
    
    # Model parameters
    parser.add_argument('--transformer_name', type=str, default='unified',
                        help='Name of the transformer model to use')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension size')
    parser.add_argument('--num_joints', type=int, default=22,
                        help='Number of joints')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    
    # Predict parameters
    parser.add_argument('--predict_output_dir', type=str,
                        help='Path to store prediction results')
    
    args = parser.parse_args()
    
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()