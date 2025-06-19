import jittor as jt
import numpy as np
import os
import argparse
from tqdm import tqdm

from models.twoinone import JointSkinModel
from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix

jt.flags.use_cuda = 1

def predict(args):
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    model = JointSkinModel(feat_dim=args.feat_dim, num_joints=args.num_joints)
    model.load(args.model_path)
    model.eval()
    loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=True,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("start predicting...")
    for batch_idx, batch in tqdm(enumerate(loader)):
        vertices = batch['vertices']
        cls = batch.get('cls', [f'class_{batch_idx}'])
        id_ = batch.get('id', [batch_idx])
        # 兼容 batch_size > 1
        B = vertices.shape[0]
        if jt.flags.use_cuda:
            vertices = vertices.cuda()
        pred_joints, pred_skin = model(vertices)
        pred_joints = pred_joints.numpy()  # [B, J, 3]
        pred_skin = pred_skin.numpy()      # [B, N, J]
        # 兼容 origin_vertices
        origin_vertices = batch.get('origin_vertices', None)
        N = batch.get('N', None)

        for i in range(B):
            save_path = os.path.join(output_dir, str(cls[i]), str(id_[i].item()))
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "predict_joints.npy"), pred_joints[i])
            np.save(os.path.join(save_path, "predict_skin.npy"), pred_skin[i])
            # 可选：保存原始点云
            if origin_vertices is not None and N is not None:
                o_vertices = origin_vertices[i, :N[i]].numpy()
                np.save(os.path.join(save_path, "transformed_vertices.npy"), o_vertices)
    print("finished")

def main():
    parser = argparse.ArgumentParser(description='Predict with JointSkinModel')
    parser.add_argument('--predict_data_list', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='twoinone')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--feat_dim', type=int, default=384)
    parser.add_argument('--num_joints', type=int, default=22)
    args = parser.parse_args()
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()